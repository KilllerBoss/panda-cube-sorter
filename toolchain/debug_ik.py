#!/usr/bin/env python3
"""Debug IK from home: contacts at home, incremental IK tracking quality."""
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

for j in range(7):
    data.qpos[model.jnt_qposadr[j]] = C.HOME_Q[j]
data.qpos[model.jnt_qposadr[7]] = 0.025
data.qpos[model.jnt_qposadr[8]] = 0.025
mujoco.mj_forward(model, data)
sp = data.site_xpos[site]
print(f"home tcp = ({sp[0]:.3f},{sp[1]:.3f},{sp[2]:.3f})")
print(f"home contacts: ncon={data.ncon}")
for k in range(min(data.ncon, 8)):
    g1, g2 = data.contact[k].geom1, data.contact[k].geom2
    n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or "?"
    n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or "?"
    print(f"  {n1} | {n2}  dist={data.contact[k].dist*1000:.2f}mm")

# arm masses per link
print("link masses:", [f"{model.body_mass[b]:.2f}" for b in range(1, 10)])

# batch IK toward hover target with many iterations (pure kinematics)
q = C.HOME_Q.copy()
target = np.array([0.40, 0.0, C.HOVER_Z], np.float32)
for it in range(60):
    q = C.ik_step(model, scratch, site, q, target, iters=1)
    scratch.qpos[model.jnt_qposadr[7]] = 0.025
    scratch.qpos[model.jnt_qposadr[8]] = 0.025
    mujoco.mj_kinematics(model, scratch)
    if it % 10 == 0 or it == 59:
        sp = scratch.site_xpos[site]
        print(f"ik it={it:2d} tcp=({sp[0]:.3f},{sp[1]:.3f},{sp[2]:.3f}) "
              f"q={np.round(q, 2)}")
