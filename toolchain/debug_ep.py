#!/usr/bin/env python3
"""Debug one full expert episode with per-transition diagnostics."""
import os, sys
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C

model = mujoco.MjModel.from_binary_path(os.path.join(HERE, "..", "scene", "scene_8.mjb"))
data = mujoco.MjData(model)
scratch = mujoco.MjData(model)
site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
cube_jnt = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
cube_body = [model.jnt_bodyid[j] for j in cube_jnt]

rng = np.random.default_rng(1000)
data.qpos[:] = 0; data.qvel[:] = 0
for j in range(7):
    data.qpos[model.jnt_qposadr[j]] = C.HOME_Q[j]
for g in (7, 8):
    data.qpos[model.jnt_qposadr[g]] = 0.025
used = []
for j in cube_jnt:
    for _t in range(3000):
        x = rng.uniform(0.30, 0.48); y = rng.uniform(-0.30, 0.30)
        r = (x * x + y * y) ** 0.5
        if not (0.36 <= r <= 0.50):
            continue
        if any((x - zx) ** 2 + (y - zy) ** 2 < 0.085 ** 2 for zx, zy in
               [(0.34, -0.30), (0.34, 0.30), (0.48, -0.08), (0.48, 0.08)]):
            continue
        if all((x - px) ** 2 + (y - py) ** 2 >= 0.06 ** 2 for px, py in used):
            used.append((x, y)); break
    a = rng.uniform(0, np.pi)
    adr = model.jnt_qposadr[j]
    data.qpos[adr:adr + 3] = (x, y, 0.275)
    data.qpos[adr + 3:adr + 7] = (np.cos(a / 2), 0, 0, np.sin(a / 2))
mujoco.mj_forward(model, data)
data.qacc_warmstart[:] = 0

table_x = model.body_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")][0]
zone_pos = np.array([[model.body_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                     f"zone_{C.COLOR_NAMES[c]}")][0] + table_x,
                      model.body_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                     f"zone_{C.COLOR_NAMES[c]}")][1]] for c in range(4)], np.float32)

task = C.ExpertTask(zone_pos)
q_goal = C.HOME_Q.copy(); q_track = C.HOME_Q.copy(); q_des_prev = C.HOME_Q.copy()
prev_phase = C.PH_RESET
dt = model.opt.timestep * 5

for cyc in range(30000):
    for _ in range(5):
        mujoco.mj_step(model, data)
    cube_xy = np.array([[data.xpos[b, 0], data.xpos[b, 1]] for b in cube_body], np.float32)
    grasped, cl, cr = C.detect_grasp(model, data)
    sp0 = data.site_xpos[site]
    q_meas = np.array([data.qpos[model.jnt_qposadr[j]] for j in range(7)], np.float32)
    tcp, grip_t, s = task.update(dt, cube_xy, np.array([i % 4 for i in range(8)]), grasped,
                                 tcp_actual=sp0.astype(np.float32), q_goal_ik=q_goal,
                                 q_actual=q_meas)
    if task.done:
        break
    if task.phase != prev_phase:
        cz = data.xpos[cube_body[task.cur]] if task.cur >= 0 else None
        sp0 = data.site_xpos[site]
        print(f"c={cyc:5d} -> {C.PHASE_NAMES[task.phase]:9s} cur={task.cur} grasped={grasped} "
              f"tcp=({sp0[0]:.3f},{sp0[1]:.3f},{sp0[2]:.3f})"
              + (f" cube=({cz[0]:.3f},{cz[1]:.3f})" if cz is not None else ""))
        if task.phase == C.PH_HOVER and task.cur >= 0:
            az = float(np.arctan2(cube_xy[task.cur][1], cube_xy[task.cur][0]))
            qc = C.q_carry(az)
            print(f"    carry az={az:.3f} qc={np.round(qc,2)} qact={np.round(q_meas,2)} "
                  f"maxdelta={np.abs(q_meas-qc).max():.3f}")
        prev_phase = task.phase
    tcp_f = tcp.astype(np.float32)
    if task.phase == C.PH_HOVER and task.cur >= 0:
        q_goal = C.q_carry(float(np.arctan2(cube_xy[task.cur][1], cube_xy[task.cur][0])))
    elif task.phase == C.PH_HOVER or tcp_f[2] > 0.45:
        q_goal = C.HOME_Q.copy()
    else:
        rr = float(np.hypot(tcp_f[0], tcp_f[1]))
        q_goal = C.q_fold(float(np.arctan2(tcp_f[1], tcp_f[0])), rr, float(tcp_f[2]))
    q_track = C.track(q_track, q_goal, dt)
    qd_des = (q_track - q_des_prev) / dt
    q_des_prev = q_track.copy()
    tau = C.pd_torques(model, data, q_track, qd_des)
    for j in range(7):
        data.ctrl[j] = tau[j]
    data.ctrl[7] = grip_t
    if task.phase == C.PH_DESCEND and task.cur == 4 and cyc % 60 == 0:
        sp = data.site_xpos[site]
        qa = np.array([data.qpos[model.jnt_qposadr[j]] for j in range(7)])
        tau0 = data.ctrl[0]
        print(f"   D2 c={cyc} qgoal={np.round(q_goal,2)} qact={np.round(qa,2)} "
              f"tau0={tau0:.1f} tcp=({sp[0]:.3f},{sp[1]:.3f},{sp[2]:.3f})")
    if task.phase == C.PH_DESCEND and cyc % 20 == 0:
        sp = data.site_xpos[site]
        cz = data.xpos[cube_body[task.cur]] if task.cur >= 0 else (0,0,0)
        qg = np.round(q_goal, 2)
        qt = np.round(q_track, 2)
        print(f"   D c={cyc} tcp=({sp[0]:.3f},{sp[1]:.3f},{sp[2]:.3f}) cube=({cz[0]:.3f},{cz[1]:.3f}) "
              f"qtrack-qgoal={np.abs(qt-qg).max():.2f}")
    if task.phase == C.PH_LIFT and task.cur == 0 and cyc % 10 == 0:
        sp = data.site_xpos[site]
        sl = [data.qpos[model.jnt_qposadr[g]] for g in (7, 8)]
        cz = data.xpos[cube_body[task.cur]]
        print(f"   L c={cyc} tcp_z={sp[2]:.3f} cube_z={cz[2]:.3f} slides=({sl[0]:.4f},{sl[1]:.4f}) "
              f"grasped={grasped} grip_tgt={grip_t:.3f}")
    if task.phase == C.PH_GRASP and task.cur >= 0 and cyc % 10 == 0:
        sp = data.site_xpos[site]
        sl = [data.qpos[model.jnt_qposadr[g]] for g in (7, 8)]
        cz = data.xpos[cube_body[task.cur]]
        names = set()
        for k in range(data.ncon):
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[k].geom1) or "?"
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[k].geom2) or "?"
            if "finger" in n1 or "finger" in n2:
                names.add(f"{n1}|{n2}")
        print(f"   g c={cyc} tcp=({sp[0]:.3f},{sp[1]:.3f},{sp[2]:.3f}) cube=({cz[0]:.3f},{cz[1]:.3f},{cz[2]:.3f}) "
              f"slides=({sl[0]:.3f},{sl[1]:.3f}) grasped={grasped} cons={sorted(names)[:3]}")
print(f"END: dead={task.dead.astype(int)} zone_stack={task.zone_stack}")
