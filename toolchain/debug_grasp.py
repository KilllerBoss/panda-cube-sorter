#!/usr/bin/env python3
"""Debug grasp: single cube, IK descent, close gripper, inspect contacts."""
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

# reset: home, open gripper (slides 0.025 each), cube 0 at (0.40, 0, 0.375)
for j in range(7):
    data.qpos[model.jnt_qposadr[j]] = C.HOME_Q[j]
data.qpos[model.jnt_qposadr[7]] = 0.025
data.qpos[model.jnt_qposadr[8]] = 0.025
adr = model.jnt_qposadr[cube_jnt[0]]
data.qpos[adr:adr+3] = (0.40, 0.0, 0.275)
data.qpos[adr+3:adr+7] = (1, 0, 0, 0)
mujoco.mj_forward(model, data)

q_goal = C.HOME_Q.copy()
q_track = C.HOME_Q.copy()
q_des_prev = C.HOME_Q.copy()
tcp_target = np.array([0.40, 0.0, C.HOVER_Z], np.float32)
last_tcp = None
dt = model.opt.timestep * 5

for cyc in range(300):
    if cyc == 120:
        tcp_target = np.array([0.40, 0.0, C.GRASP_Z], np.float32)  # descend
    if cyc == 220:
        tcp_target[2] = C.GRASP_Z  # hold, close below
    # IK + PD (scratch for IK; global reseed on target jumps)
    dxy2 = (tcp_target[0] - last_tcp[0]) ** 2 + (tcp_target[1] - last_tcp[1]) ** 2 \
        if last_tcp is not None else 1.0
    if last_tcp is None or cyc == 120:   # phase transition -> reseed
        q_goal = C.global_ik(model, scratch, site, tcp_target, q_hint=q_goal)
        last_tcp = tcp_target.copy()
    q_goal = C.ik_step(model, scratch, site, q_goal, tcp_target)
    q_track = C.track(q_track, q_goal, dt)
    qd_des = (q_track - q_des_prev) / dt
    q_des_prev = q_track.copy()
    tau = C.pd_torques(model, data, q_track, qd_des)
    for j in range(7):
        data.ctrl[j] = tau[j]
    data.ctrl[7] = C.GRIP_CLOSED if cyc >= 220 else (C.GRIP_PRE if cyc >= 120 else C.GRIP_OPEN)
    for _ in range(5):
        mujoco.mj_step(model, data)
    if 100 <= cyc <= 240 and cyc % 10 == 0:
        R = data.site_xmat[site].reshape(3, 3)
        ax = R[:, 0]
        print(f"  d cyc={cyc} tcp=({sp[0]:.3f},{sp[1]:.3f},{sp[2]:.3f}) "
              f"approach_z={ax[2]:.3f} cube=({cz[0]:.3f},{cz[1]:.3f}) "
              f"slides=({sl[0]:.3f},{sl[1]:.3f})")
    if cyc % 20 == 0 or cyc in (220, 221, 230, 250, 299):
        sp = data.site_xpos[site]
        sl = [data.qpos[model.jnt_qposadr[g]] for g in (7, 8)]
        con = []
        for k in range(data.ncon):
            g1, g2 = data.contact[k].geom1, data.contact[k].geom2
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1)
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2)
            con.append(f"{n1}|{n2}:{data.contact[k].dist*1000:.1f}mm")
        gr, _, _ = C.detect_grasp(model, data)
        cz = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube_red_0")]
        print(f"c={cyc:3d} tcp=({sp[0]:.3f},{sp[1]:.3f},{sp[2]:.3f}) z_cube={cz[2]:.3f} "
              f"slides=({sl[0]:.4f},{sl[1]:.4f}) grasped={gr} cube_xy=({cz[0]:.3f},{cz[1]:.3f}) ncon={data.ncon}")
        if cyc >= 220 and con:
            print("   ", "; ".join(con[:6]))
