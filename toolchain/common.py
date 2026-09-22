# toolchain/common.py — shared constants & helpers (mirrors native/include/pcs/types.h)
import numpy as np

DOF = 7
NUM_ACT = 8
EMB_DIM = 32
NUM_PROTO = 8
PROTO_LEN = 64
NUM_CUBES = 8
NUM_COLORS = 4
NUM_PHASES = 8
MLP_IN = 24
MLP_HIDDEN = 16
MLP_OUT = 8
LSNN_N = 128
LSNN_IN = 576
LORA_RANK = 4
DEC_OUT = 48
HINGES1 = 4
L1_FEATS = MLP_IN + MLP_IN * HINGES1   # 120
L2_FEATS = MLP_HIDDEN + MLP_HIDDEN * HINGES1  # 80

PH_RESET, PH_HOME, PH_HOVER, PH_DESCEND, PH_GRASP, PH_LIFT, PH_TRANSPORT, PH_PLACE = range(8)
PHASE_NAMES = ["RESET", "HOME", "HOVER", "DESCEND", "GRASP", "LIFT", "TRANSPORT", "PLACE"]
COLOR_NAMES = ["red", "green", "blue", "yellow"]

HOME_Q = np.array([0.009, -0.211, 0.040, -1.305, -0.005, 3.063, 0.004])


def q_carry(az):
    """Safe transit posture: tcp high (0.47 m) above the target azimuth."""
    return q_fold(az, 0.40, 0.47)


def q_fold(az, r, z):
    """Closed-form fold IK (j3=j5=0, approach straight down).

    Mirrors pcs::q_fold in native/include/pcs/types.h."""
    L1, L2 = 0.3985, 0.3845
    wx, wy = r, z - 0.012
    d = np.hypot(wx, wy)
    d = np.clip(d, 0.05, L1 + L2 - 0.005)
    cb = np.clip((d * d - (L1 * L1 + L2 * L2)) / (2 * L1 * L2), -1, 1)
    beta = np.arccos(cb)
    gamma = np.arctan2(L2 * np.sin(beta), L1 + L2 * np.cos(beta))
    argW = np.arctan2(wy, wx)
    a = -(gamma + argW)     # shoulder up
    b = beta                # elbow-UP branch (elbow stays above the table)
    return np.array([az, a, 0.0, b, 0.0, np.pi / 2 - a - b, 0.0], np.float32)
GRIP_CLOSED = 0.006   # tendon length target (sum of both slides): squeeze
GRIP_PRE = 0.026      # full-close-ish reserve
GRIP_CAP = 0.025      # descent capture gap ~58 mm
GRIP_OPEN = 0.05      # tendon length target: both slides 0.025 = 95 mm opening

HOVER_Z = 0.36
GRASP_Z = 0.29
PLACE_Z0 = 0.283
TRANS_Z = 0.42
STACK_DZ = 0.052

PHASE_DURATION = {PH_RESET: 0.3, PH_HOME: 1.1, PH_HOVER: 0.7, PH_DESCEND: 1.0,
                  PH_GRASP: 0.55, PH_LIFT: 0.9, PH_TRANSPORT: 1.1, PH_PLACE: 1.0}

Q_VMAX = 1.5  # rad/s joint rate limit for the PD setpoint (safety rail)


def track(q_track, q_goal, dt, vmax=Q_VMAX):
    """Rate-limited setpoint: protects the scene from IK setpoint jumps."""
    dq = q_goal - q_track
    mx = vmax * dt
    return q_track + np.clip(dq, -mx, mx)


KP = np.array([70, 70, 60, 60, 30, 30, 22], np.float32)
KD = np.array([9, 9, 8, 8, 5, 5, 4], np.float32)


def minjerk(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


def clamp_to_limits(model, q, margin=0.02):
    out = q.copy()
    for j in range(DOF):
        lo, hi = model.jnt_range[j]
        out[j] = np.clip(q[j], lo + margin, hi - margin)
    return out


def site_jac(model, data, site_id):
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    import mujoco
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return jacp[:, :7], jacr[:, :7]


def ik_step(model, scratch, site_id, q_goal, tcp_target, iters=2, lam=1e-4):
    """Incremental damped-LS IK — mirrors glue/sim_glue.cpp.

    IMPORTANT: uses a scratch MjData so the simulated state (data.qpos) is
    never modified — same separation as the C++ glue (member q_goal_)."""
    import mujoco
    q = q_goal.copy()
    k_ori = 0.5
    for _ in range(iters):
        for j in range(DOF):
            scratch.qpos[model.jnt_qposadr[j]] = q[j]
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)   # cdof needed by mj_jacSite
        sp = scratch.site_xpos[site_id]
        R = scratch.site_xmat[site_id].reshape(3, 3)
        ep = tcp_target - sp
        ez = -R[:, 0] - np.array([0, 0, -1.0])   # approach axis -> down
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, scratch, jacp, jacr, site_id)
        J = np.vstack([jacp[:, :7], k_ori * jacr[:, :7]])
        err = np.concatenate([ep, k_ori * ez])
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), err)
        q = q + np.clip(dq, -0.1, 0.1)
        q = clamp_to_limits(model, q)
        q[2] = 0.0   # pin wrist rolls (fold family, prevents flip drift)
        q[4] = 0.0
    return q


def global_ik(model, scratch, site_id, tcp_target, iters=250, q_hint=None):
    """Multi-seed damped-LS IK — mirrors glue/sim_glue.cpp global_ik().
    Returns the best joint solution; never touches the simulated data."""
    import mujoco
    seeds = [
        np.array([0.009, -0.211, 0.040, -1.305, -0.005, 3.063, 0.004]),
        np.array([0.0, 0.35, 0.0, -1.8, 0.0, 3.02, 0.0]),
        np.array([0.0, -0.5, 0.0, -1.2, 0.0, 3.27, 0.0]),
        np.array([0.4, 0.2, 0.6, -1.9, -0.3, 3.27, 0.3]),
        np.array([-0.4, 0.1, -0.5, -2.2, 0.3, 3.62, -0.4]),
    ]
    # analytic seed: aim the base yaw at the target, elbow folded for the
    # natural grasp radius, approach pointing down (j2+j4+j6 = pi/2)
    j1 = float(np.arctan2(tcp_target[1], max(tcp_target[0], 0.05)))
    seeds.append(np.array([j1, 0.30, 0.0, -1.60, 0.0, np.pi / 2 + 1.30, 0.0]))
    seeds.append(np.array([j1, 0.10, 0.0, -1.90, 0.0, np.pi / 2 + 1.80, 0.0]))
    seeds.append(np.array([j1, 0.30, 0.0, -1.75, 0.0, np.pi / 2 + 1.45, 0.0]))
    k_ori = 0.6
    q_prev = q_hint.copy() if q_hint is not None else None
    best_q, best_err = None, 1e9     # gated best (accurate)
    near_q, near_err = None, 1e9     # among accurate: closest to hint
    fallback_q, fallback_err = None, 1e9
    for s in seeds:
        q = clamp_to_limits(model, s)
        for _ in range(iters):
            for j in range(DOF):
                scratch.qpos[model.jnt_qposadr[j]] = q[j]
            mujoco.mj_kinematics(model, scratch)
            mujoco.mj_comPos(model, scratch)
            sp = scratch.site_xpos[site_id]
            ep = tcp_target - sp
            R = scratch.site_xmat[site_id].reshape(3, 3)
            ax = R[:, 0]
            ez = -ax - np.array([0, 0, -1.0])
            jp = np.zeros((3, model.nv)); jr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, scratch, jp, jr, site_id)
            J = np.vstack([jp[:, :7], k_ori * jr[:, :7]])
            err = np.concatenate([ep, k_ori * ez])
            damp = 1e-3 * (1.0 + 50.0 * float(err @ err))
            Ainv = np.linalg.inv(J @ J.T + damp * np.eye(6))
            dq = J.T @ Ainv @ err
            # nullspace: hold the seed's branch (j3/j5/j7 rolls at 0) — without
            # this the redundancy drifts into a wrist-flipped local minimum.
            q = clamp_to_limits(model, q + np.clip(dq, -0.5, 0.5))
            q[2] = 0.0   # pin j3/j5 rolls: the down-pointing fold family needs
            q[4] = 0.0   # no wrist rolls; this prevents wrist-flip minima
        for j in range(DOF):
            scratch.qpos[model.jnt_qposadr[j]] = q[j]
        mujoco.mj_kinematics(model, scratch)
        sp = scratch.site_xpos[site_id]
        perr = float(np.linalg.norm(tcp_target - sp))
        az = float(R[2, 0])   # world-z of the approach axis (site local x)
        oerr = abs(1.0 + az)  # approach axis should point straight down
        qdist = float(np.sum((q - q_prev) ** 2)) if q_prev is not None else 0.0
        # tier 2: vertical approach, any accuracy -> min perr
        if oerr < 0.6 and perr < fallback_err:
            fallback_err, fallback_q = perr, q.copy()
        # tier 1: accurate AND vertical -> closest to hint
        if perr < 0.035 and oerr < 0.6:
            score = 0.3 * oerr + 0.5 * np.sqrt(qdist)
            if score < near_err:
                near_err, near_q = score, q.copy()
    if near_q is not None:
        return near_q
    if fallback_q is not None:
        return fallback_q
    return q_hint


def pd_torques(model, data, q_des, qd_des, kp=KP, kd=KD):
    tau = np.zeros(DOF, np.float32)
    for j in range(DOF):
        dof = model.jnt_dofadr[j]
        tau[j] = data.qfrc_bias[dof] + kp[j] * (q_des[j] - data.qpos[model.jnt_qposadr[j]]) \
                 + kd[j] * (qd_des[j] - data.qvel[dof])
        lo, hi = model.actuator_ctrlrange[j]
        tau[j] = np.clip(tau[j], lo, hi)
    return tau


def detect_grasp(model, data, n_cubes=8):
    """Both fingers touching the same cube geom — mirrors sim_glue.cpp."""
    fl = model.geom("finger_l_g").id
    fr = model.geom("finger_r_g").id
    contact_l = np.zeros(n_cubes, bool)
    contact_r = np.zeros(n_cubes, bool)
    cube_geom = [model.geom(f"geom_{COLOR_NAMES[i % 4]}_{i}").id for i in range(n_cubes)]
    for k in range(data.ncon):
        g1, g2 = data.contact[k].geom1, data.contact[k].geom2
        for ci, cg in enumerate(cube_geom):
            if (g1 == fl and g2 == cg) or (g2 == fl and g1 == cg):
                contact_l[ci] = True
            if (g1 == fr and g2 == cg) or (g2 == fr and g1 == cg):
                contact_r[ci] = True
    both = contact_l & contact_r
    return bool(both.any()), contact_l, contact_r


class ExpertTask:
    """Ground-truth task machine — mirrors task_layer.cpp but uses TRUE cube positions."""

    def __init__(self, zone_xy):
        self.zones = zone_xy  # (4,2)
        self.reset()

    def reset(self):
        self.phase = PH_RESET
        self.phase_t = 0.0
        self.zone_stack = [0, 0, 0, 0]
        self.next_color = 0
        self.in_flight = -1
        self.dead = np.zeros(NUM_CUBES, bool)
        self.cur = -1
        self.q_start = HOME_Q.copy()
        self.done = False
        self.grab_xy = None
        self.color_log = []

    def pick_next(self, cube_xy, cube_color):
        best, best_d = -1, 1e9
        for i in range(NUM_CUBES):
            if self.dead[i]:
                continue
            if cube_color[i] != self.next_color:
                continue
            d = cube_xy[i, 0] ** 2 + cube_xy[i, 1] ** 2
            if d < best_d:
                best, best_d = i, d
        if best < 0:
            for i in range(NUM_CUBES):
                if not self.dead[i]:
                    best = i  # fallback: any live cube
                    break
        self.cur = best

    def update(self, dt, cube_xy, cube_color, grasped, tcp_actual=None, q_goal_ik=None,
               q_actual=None):
        """Returns (tcp_target(3,), grip_target, s). Event-driven transitions:
        phases end when the measured tcp reached the target (or on timeout)."""
        self.phase_t += dt
        dur = PHASE_DURATION[self.phase]
        s = min(self.phase_t / dur, 1.0)
        tcp_actual = np.array([0.42, 0, 0.45]) if tcp_actual is None else tcp_actual
        def reached(t, tol=0.02):
            return np.linalg.norm(np.asarray(t, np.float32) - tcp_actual) < tol
        if self.phase not in (PH_DESCEND, PH_GRASP):
            self.grab_xy = None
        tcp = np.array([0.42, 0.0, HOVER_Z])
        grip = GRIP_OPEN
        nxt = None

        if self.phase == PH_RESET:
            nxt = PH_HOME
        elif self.phase == PH_HOME:
            tcp = np.array([0.417, 0.0, 0.517])   # home hold height
            if (s >= 1.0 and np.sum((q_goal_ik - HOME_Q) ** 2) < 0.04) or self.phase_t >= 3.0:
                self.pick_next(cube_xy, cube_color)
                if self.cur < 0:
                    self.done = True
                else:
                    nxt = PH_HOVER
        elif self.phase == PH_HOVER:
            # CARRY transit: wait until the arm reached the carry posture
            tcp = np.array([0.417, 0.0, 0.517])
            arrived = True
            if q_actual is not None and self.cur >= 0:
                az = float(np.arctan2(cube_xy[self.cur][1], cube_xy[self.cur][0]))
                arrived = np.abs(q_actual - q_carry(az)).max() < 0.15
            if (s >= 1.0 and arrived) or self.phase_t >= 3.0:
                nxt = PH_DESCEND
        elif self.phase == PH_DESCEND:
            if self.cur >= 0:
                tcp[:2] = cube_xy[self.cur]   # live tracking: centers the funnel
            tcp[2] = GRASP_Z
            # funnel: wide all the way down; close stationary in PH_GRASP
            grip = GRIP_OPEN + max(0.0, min(1.0, (s - 0.5) / 0.2)) * (0.0377 - GRIP_OPEN)
            if (s >= 1.0 and reached(tcp, 0.02)) or self.phase_t >= 4.0:
                nxt = PH_GRASP
        elif self.phase == PH_GRASP:
            tcp[2] = GRASP_Z
            if self.cur >= 0:
                tcp[:2] = cube_xy[self.cur]
            grip = GRIP_CLOSED
            if grasped or self.phase_t >= 2.0:
                if grasped:
                    self.in_flight = self.cur
                    nxt = PH_LIFT
                else:
                    self.dead[self.cur] = True
                    nxt = PH_HOME
        elif self.phase == PH_LIFT:
            if self.cur >= 0:
                tcp[:2] = cube_xy[self.cur]
            tcp[2] = TRANS_Z
            grip = GRIP_CLOSED
            if (s >= 1.0 and tcp_actual[2] > TRANS_Z - 0.025) or self.phase_t >= 3.0:
                self.next_color = cube_color[self.in_flight]
                nxt = PH_TRANSPORT
        elif self.phase == PH_TRANSPORT:
            tcp[:2] = self.zones[self.next_color]
            tcp[2] = TRANS_Z
            grip = GRIP_CLOSED
            if (s >= 1.0 and reached([tcp[0], tcp[1], tcp_actual[2]], 0.025)) or self.phase_t >= 4.0:
                nxt = PH_PLACE
        elif self.phase == PH_PLACE:
            tcp[:2] = self.zones[self.next_color]
            tcp[2] = PLACE_Z0 + self.zone_stack[self.next_color] * STACK_DZ
            grip = GRIP_CLOSED if s < 0.6 else GRIP_OPEN
            if (s >= 1.0 and reached(tcp, 0.02)) or self.phase_t >= 3.0:
                self.zone_stack[self.next_color] += 1
                if self.in_flight >= 0:
                    self.dead[self.in_flight] = True
                self.in_flight = -1
                nxt = PH_HOME

        if nxt is not None:
            self.phase = nxt
            self.phase_t = 0.0
            self.q_start_flag = True
        return tcp, grip, s
