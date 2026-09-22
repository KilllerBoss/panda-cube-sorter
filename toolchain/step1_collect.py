#!/usr/bin/env python3
"""toolchain/step1_collect.py — scripted-expert data collection with live SNN.

Runs the ground-truth task machine (common.ExpertTask) in the MuJoCo scene,
drives the arm via incremental IK + PD(+bias) torques, renders the event-camera
view through a CPU projector and computes the live 32-d SNN embedding per
cycle — exactly what the on-device pipeline does. Result: dataset.npz.
"""
import os
import sys
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from snn_py import EventCamera, LsnnState, AlifLsnn, PredCoder, EV_W, EV_H, NUM_BINS  # noqa
import common as C  # noqa

COL_RGB = {0: (235, 26, 26), 1: (26, 209, 46), 2: (38, 89, 242), 3: (242, 217, 31)}
ZONE_RGB = COL_RGB


class NumpyProjector:
    """Mirror of desktop CpuProjector: 96x72 top-down color view."""

    def __init__(self, model):
        self.cam = mujoco.MjvCamera()
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_event")
        self.pos = model.cam_pos[cam_id].astype(np.float32)
        self.right = np.array([1, 0, 0], np.float32)
        self.up = np.array([0, 1, 0], np.float32)
        self.fwd = np.array([0, 0, -1], np.float32)
        self.f = 0.5 * EV_H / np.tan(np.deg2rad(model.cam_fovy[cam_id]) / 2)

    def _project(self, pts):
        d = pts - self.pos
        z = d @ self.fwd
        u = EV_W / 2 + self.f * (d @ self.right) / z
        v = EV_H / 2 - self.f * (d @ self.up) / z
        return u, v

    def render(self, model, data, cube_body, zone_pos, pulse):
        img = np.empty((EV_H, EV_W, 3), np.uint8)
        img[:] = (115, 97, 77)
        jx, jy = (0.008, 0.004) if pulse else (0.0, 0.0)
        # zones
        for c in range(4):
            x0, y0 = zone_pos[c][0] - 0.07 + jx, zone_pos[c][1] - 0.07 + jy
            x1, y1 = zone_pos[c][0] + 0.07 + jx, zone_pos[c][1] + 0.07 + jy
            u, v = self._project(np.array([[x0, y0, 0.3545], [x1, y1, 0.3545]], np.float32))
            paint_rect(img, u[0], v[0], u[1], v[1], ZONE_RGB[c])
        # cubes (painter order: lower z first)
        order = np.argsort([data.xpos[b, 2] for b in cube_body])
        for i in order:
            p = data.xpos[cube_body[i]].astype(np.float32)
            corners = p + np.array([[dx, dy, dz] for dx in (-0.025, 0.025)
                                    for dy in (-0.025, 0.025)
                                    for dz in (-0.025, 0.025)], np.float32)
            corners[:, 0] += jx
            corners[:, 1] += jy
            u, v = self._project(corners)
            col = COL_RGB[i % 4]
            if np.all(np.isfinite(u)):
                paint_rect(img, u.min(), v.min(), u.max(), v.max(), col)
        return img


def paint_rect(img, u0, v0, u1, v1, col):
    h, w = img.shape[:2]
    x0, x1 = max(0, int(u0)), min(w - 1, int(u1))
    y0, y1 = max(0, int(v0)), min(h - 1, int(v1))
    if x1 < x0 or y1 < y0:
        return
    img[y0:y1 + 1, x0:x1 + 1] = col


def main():
    n_eps = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    scene = os.path.join(HERE, "..", "scene", "scene_8.mjb")
    model = mujoco.MjModel.from_binary_path(scene)
    data = mujoco.MjData(model)
    scratch = mujoco.MjData(model)   # IK scratch — never touches 'data'
    site_tcp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")

    init = np.load(os.path.join(HERE, "init_weights.npz"))
    W = {k: init[k].copy() for k in init.files}
    W["lambda_v"], W["rho_a"], W["beta_th"], W["vth0"] = \
        float(W["meta"][0]), float(W["meta"][1]), float(W["meta"][2]), float(W["meta"][3])
    W["pred_eta"] = float(W["meta"][5])

    cam = EventCamera()
    snn = AlifLsnn()
    coder = PredCoder()
    st = LsnnState()
    bins = np.zeros(NUM_BINS, np.float32)
    pbins = np.zeros(NUM_BINS, np.float32)   # latched last pulse snapshot
    emb = np.zeros(32, np.float32)

    # ids
    cube_jnt, cube_body = [], []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            cube_jnt.append(j)
            cube_body.append(model.jnt_bodyid[j])
    table_x = model.body_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")][0]
    zone_pos = np.array([[model.body_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                       f"zone_{C.COLOR_NAMES[c]}")][0] + table_x,
                          model.body_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                       f"zone_{C.COLOR_NAMES[c]}")][1]] for c in range(4)], np.float32)
    proj = NumpyProjector(model)

    rows = {k: [] for k in
            ["phase", "s", "q_start", "q_goal", "q_des", "q", "qd", "grip",
             "grip_target", "emb", "cube_xy", "cube_color", "grasped", "q_add_base",
             "pbins"]}
    q_goal = C.HOME_Q.copy()
    q_track = C.HOME_Q.copy()   # rate-limited PD setpoint
    q_des_prev = C.HOME_Q.copy()
    last_tcp = None

    def reset_episode(seed):
        nonlocal q_goal, q_track, q_des_prev, last_tcp
        rng = np.random.default_rng(seed)
        data.qpos[:] = 0
        data.qvel[:] = 0
        for j in range(C.DOF):
            data.qpos[model.jnt_qposadr[j]] = C.HOME_Q[j]
        for g in (7, 8):  # finger sliders (arm joints are first in the MJCF)
            data.qpos[model.jnt_qposadr[g]] = 0.025
        used = []
        for i, j in enumerate(cube_jnt):
            for _t in range(20000):
                x = rng.uniform(0.30, 0.48)
                y = rng.uniform(-0.30, 0.30)
                r = (x * x + y * y) ** 0.5
                if not (0.36 <= r <= 0.50):  # fold-IK reach band
                    continue
                if any((x - zx) ** 2 + (y - zy) ** 2 < 0.085 ** 2 for zx, zy in zone_pos):
                    continue
                if all((x - px) ** 2 + (y - py) ** 2 >= 0.06 ** 2 for px, py in used):
                    used.append((x, y))
                    break
            a = rng.uniform(0, np.pi)
            adr = model.jnt_qposadr[j]
            data.qpos[adr:adr + 3] = (x, y, 0.275)
            data.qpos[adr + 3:adr + 7] = (np.cos(a / 2), 0, 0, np.sin(a / 2))
        mujoco.mj_forward(model, data)
        data.qacc_warmstart[:] = 0
        q_goal = C.HOME_Q.copy()
        q_track = C.HOME_Q.copy()
        q_des_prev = C.HOME_Q.copy()
        last_tcp = None

    dt = model.opt.timestep * 5
    n_total = 0
    for ep in range(n_eps):
        reset_episode(1000 + ep * 7919)
        task = C.ExpertTask(zone_pos)
        cam = EventCamera()  # fresh reference frame per episode
        pbins[:] = 0
        prev_phase = C.PH_RESET
        q_start_cur = C.HOME_Q.copy()
        for cyc in range(14000):
            pulse = (cyc % 50) == 0
            # physics
            for _ in range(5):
                mujoco.mj_step(model, data)
            # true state
            cube_xy = np.array([[data.xpos[b, 0], data.xpos[b, 1]]
                                for b in cube_body], np.float32)
            cube_color = np.array([i % 4 for i in range(C.NUM_CUBES)])
            grasped, _, _ = C.detect_grasp(model, data)
            # expert task machine (event-driven via measured tcp)
            tcp_meas = data.site_xpos[site_tcp].astype(np.float32)
            q_meas = np.array([data.qpos[model.jnt_qposadr[j]] for j in range(C.DOF)], np.float32)
            tcp, grip_t, s = task.update(dt, cube_xy, cube_color, grasped,
                                         tcp_actual=tcp_meas, q_goal_ik=q_goal,
                                         q_actual=q_meas)
            if task.done:
                break
            # IK policy (mirrors sim_glue.cpp): joint-hold for carry/home,
            # otherwise the closed-form fold solution of the tcp target
            tcp_f = tcp.astype(np.float32)
            if task.phase == C.PH_HOVER and task.cur >= 0:
                q_goal = C.q_carry(float(np.arctan2(cube_xy[task.cur][1],
                                                    cube_xy[task.cur][0])))
            elif task.phase == C.PH_HOVER or tcp_f[2] > 0.45:
                q_goal = C.HOME_Q.copy()
            else:
                rr = float(np.hypot(tcp_f[0], tcp_f[1]))
                q_goal = C.q_fold(float(np.arctan2(tcp_f[1], tcp_f[0])), rr, float(tcp_f[2]))
            q_track = C.track(q_track, q_goal, dt)
            qd_des = (q_track - q_des_prev) / dt
            q_des_prev = q_track.copy()
            # perception (event cam + SNN + embedding)
            frame = proj.render(model, data, cube_body, zone_pos, pulse)
            n_ev = cam.process(frame, pulse, bins)
            n_sp = snn.step(bins, W, st)
            fe = coder.step(bins, st, W, emb)
            # record BEFORE pd write, matching C++ ordering (q after physics)
            if task.phase != prev_phase:
                # phase just transitioned this cycle: freeze q_start like start_phase()
                q_start_cur = q.copy() if len(rows["q"]) else C.HOME_Q.copy()
                # q is from previous cycle; recompute fresh current q
                q_start_cur = np.array([data.qpos[model.jnt_qposadr[j]] for j in range(C.DOF)], np.float32)
                prev_phase = task.phase
            task.q_start = q_start_cur
            q = np.array([data.qpos[model.jnt_qposadr[j]] for j in range(C.DOF)], np.float32)
            qd = np.array([data.qvel[model.jnt_dofadr[j]] for j in range(C.DOF)], np.float32)
            base = q_goal  # expert q* == IK goal (== q_des semantics in C++)
            rows["phase"].append(task.phase)
            rows["s"].append(s)
            rows["q_start"].append(task.q_start.copy())
            rows["q_goal"].append(q_goal.copy())
            rows["q_des"].append(q_track.copy())
            rows["q"].append(q)
            rows["qd"].append(qd)
            rows["grip"].append(float(data.qpos[model.jnt_qposadr[7]]))
            rows["grip_target"].append(grip_t)
            rows["emb"].append(emb.copy())
            if pulse:
                pbins[:] = bins            # latch the refresh-pulse snapshot
            rows["pbins"].append(pbins.copy())
            rows["cube_xy"].append(cube_xy.copy())
            rows["cube_color"].append(cube_color.copy())
            rows["grasped"].append(grasped)
            rows["q_add_base"].append(base - task.q_start)  # dq = q_goal-q_start
            n_total += 1
            # torques for next physics step
            tau = C.pd_torques(model, data, q_track, qd_des)
            for j in range(C.DOF):
                data.ctrl[j] = tau[j]
            data.ctrl[7] = grip_t
        print(f"[step1] episode {ep}: cycles so far={n_total}")

    out = {k: np.array(v) for k, v in rows.items()}
    np.savez_compressed(os.path.join(HERE, "dataset.npz"), **out)
    print(f"[step1] dataset.npz: {n_total} cycles, "
          f"phases={ {int(p): int(np.sum(out['phase'] == p)) for p in np.unique(out['phase'])} }")


if __name__ == "__main__":
    main()
