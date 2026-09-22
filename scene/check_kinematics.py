#!/usr/bin/env python3
"""scene/check_kinematics.py — sanity-check the MJCF arm & find a good home pose.

Damped-LS IK with multiple restarts + nullspace bias toward a neutral posture.
Prints reach results for all task-relevant targets and a computed home pose.
"""
import os
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))

Q_NEUTRAL = np.array([0.0, 0.35, 0.0, -1.8, 0.0, 3.02, 0.0])  # approach down


def fk_tcp(model, data):
    mujoco.mj_forward(model, data)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    return data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy()


def jac7(model, data):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, sid)
    return jacp[:, :7], jacr[:, :7]


def clamp(model, q):
    out = q.copy()
    for i, (lo, hi) in enumerate(model.jnt_range[:7]):
        out[i] = np.clip(q[i], lo + 0.02, hi - 0.02)
    return out


def ik(model, data, q_seed, target_pos, zdir, iters=200,
       damp0=1e-2, k_ori=0.35, k_null=0.03):
    q = clamp(model, q_seed)
    best_q, best_err = q.copy(), 1e9
    rng = np.random.default_rng(3)
    for it in range(iters):
        data.qpos[:7] = q
        p, R = fk_tcp(model, data)
        ep = target_pos - p
        ez = zdir - R[:, 0]
        err = np.concatenate([ep, k_ori * ez])
        pos_err = np.linalg.norm(ep); ori_err = np.linalg.norm(ez)
        total = pos_err + 0.3 * ori_err
        if total < best_err:
            best_err, best_q = total, q.copy()
            best_pos, best_ori = pos_err, ori_err
        if pos_err < 2e-3 and ori_err < 0.05:
            break
        jp, jr = jac7(model, data)
        J = np.vstack([jp, k_ori * jr])
        JT = J.T
        damp = damp0 * (1.0 + 100.0 * total)
        A = J @ JT + damp**2 * np.eye(6)
        dq = JT @ np.linalg.solve(A, err)
        # nullspace: pull toward neutral posture
        N = np.eye(7) - JT @ np.linalg.solve(A, J)
        dq += k_null * N @ (Q_NEUTRAL - q)
        q = clamp(model, q + np.clip(dq, -0.25, 0.25))
        if it % 60 == 59:  # restart kick on stagnation
            q = clamp(model, best_q + rng.normal(0, 0.08, 7))
    return best_q, best_err, best_pos, best_ori


def main():
    xml_path = os.path.join(HERE, "scene_8.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    print(f"nq={model.nq} nv={model.nv} nu={model.nu} mass_arm="
          f"{np.sum(model.body_mass[1:9]):.2f}kg")

    zdir = np.array([0.0, 0.0, -1.0])
    # read actual zone positions from the model (table frame -> world)
    table_x = model.body_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")][0]
    targets = {}
    for zn in ("zone_red", "zone_green", "zone_blue", "zone_yellow"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, zn)
        wx, wy = model.body_pos[bid][0] + table_x, model.body_pos[bid][1]
        targets[zn + "_place"] = (wx, wy, 0.32)   # stack height 1
    for cx, cy, nm in ((0.30, -0.14, "spawn_sw"), (0.30, 0.14, "spawn_nw"),
                       (0.44, -0.14, "spawn_se"), (0.44, 0.14, "spawn_ne")):
        targets[nm + "_grasp"] = (cx, cy, 0.32)
    seeds = [Q_NEUTRAL,
             np.array([0.0, -0.5, 0.0, -1.2, 0.0, 3.27, 0.0]),
             np.array([0.4, 0.2, 0.6, -1.9, -0.3, 3.27, 0.3]),
             np.array([-0.4, 0.1, -0.5, -2.2, 0.3, 3.62, -0.4])]
    all_ok = True
    for name, t in targets.items():
        best, bq, bp, bo = 1e9, None, 1e9, 1e9
        for s in seeds:
            q, e, ep_e, eo_e = ik(model, data, s, np.array(t), zdir)
            if e < best:
                best, bq, bp, bo = e, q, ep_e, eo_e
        data.qpos[:7] = bq
        p, _ = fk_tcp(model, data)
        ok = "OK " if bp < 0.02 and bo < 0.15 else "FAIL"
        all_ok &= bp < 0.02 and bo < 0.15
        print(f"{ok} IK {name:12s} pos={bp:.4f} ori={bo:.3f} tcp={np.round(p,3)} q={np.round(bq,2)}")

    # home pose: hover over spawn area center, pointing down
    q_home, e, ep_e, eo_e = ik(model, data, Q_NEUTRAL, np.array([0.42, 0.0, 0.52]), zdir)
    print(f"\nHOME q = np.array([{', '.join(f'{v:.3f}' for v in q_home)}])  pos={ep_e:.4f} ori={eo_e:.3f}")
    print("ALL_REACHABLE" if all_ok else "SOME_FAILED")


if __name__ == "__main__":
    main()
