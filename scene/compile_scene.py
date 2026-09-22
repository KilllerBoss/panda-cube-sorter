#!/usr/bin/env python3
"""scene/compile_scene.py — MJCF template + N cubes -> binary .mjb scenes.

Replaces the <!--CUBES--> marker in scene_template.xml with N randomly-placed
(compile-time default layout) cubes, then compiles to MuJoCo's binary model
format (.mjb). MJB files are platform-independent (same endianness), so the
x86-compiled MJB runs identically on arm64-v8a — no XML parsing at app start.

Outputs:
  scene/scene_8.mjb    -> app asset "scene.mjb"      (8 cubes: 4 colors x 2)
  scene/scene_50.mjb   -> app asset "scene_50.mjb"   (stress test, 50 cubes)

Usage:  python3 compile_scene.py [--out DIR]
"""
import argparse
import json
import os
import sys

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))

CUBE_COLORS = {
    "red":    (0.92, 0.10, 0.10, 1.0),
    "green":  (0.10, 0.82, 0.18, 1.0),
    "blue":   (0.15, 0.35, 0.95, 1.0),
    "yellow": (0.95, 0.85, 0.12, 1.0),
}

# spawn region on the table (world coords), well inside arm workspace
# active scene: compact region in front of the arm; stress scene: whole table
# Closed-form fold IK (see pcs::q_fold) solves vertical top-down grasps for
# r in ~[0.30, 0.52] — cubes/zones live in that annulus in front of the arm.
# The 50-cube stress scene spawns over the whole table (physics/thermal
# benchmark only — no sorting run there).
SPAWN = {
    8:  dict(r=(0.36, 0.50), y=(-0.30, 0.30), x=(0.30, 0.48), min_dist=0.06),
    50: dict(r=(0.36, 0.54), y=(-0.34, 0.34), x=(0.26, 0.55), min_dist=0.056),
}
ZONE_XY = [(0.36, -0.26), (0.36, 0.26), (0.48, -0.12), (0.48, 0.12)]
CUBE_REST_Z = 0.25 + 0.025  # table top (0.25) + half cube


def cube_fragment(idx: int, color: str, pos, quat=None) -> str:
    rgba = CUBE_COLORS[color]
    q = "" if quat is None else ' quat="%s"' % " ".join("%.4f" % v for v in quat)
    return f'''    <body name="cube_{color}_{idx}" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}"{q}>
      <freejoint name="free_{color}_{idx}"/>
      <geom name="geom_{color}_{idx}" class="cube" rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>
      <site name="site_{color}_{idx}" pos="0 0 0" size="0.004" rgba="{rgba[0]} {rgba[1]} {rgba[2]} 0.6" type="sphere"/>
    </body>'''


def sample_positions(n: int, rng: np.random.Generator, region):
    """Random positions with minimum spacing.

    Small n: rejection sampling. Large n (stress scene): jittered hex grid
    (guaranteed collision-free packing inside the reachable radius).
    """
    if n <= 16:
        pts = []
        tries = 0
        while len(pts) < n:
            tries += 1
            assert tries < 200000, "spawn sampling failed"
            x = rng.uniform(*region["x"])
            y = rng.uniform(*region["y"])
            r = (x * x + y * y) ** 0.5
            if not (region["r"][0] <= r <= region["r"][1]):
                continue
            if any((x - zx) ** 2 + (y - zy) ** 2 < 0.085 ** 2 for zx, zy in ZONE_XY):
                continue  # keep spawn off the sorting zones
            if all((x - px) ** 2 + (y - py) ** 2 >= region["min_dist"]**2 for px, py in pts):
                pts.append((x, y))
        return pts

    # jittered hex grid centered in the spawn region
    d = region["min_dist"]
    cx = 0.5 * (region["x"][0] + region["x"][1])
    dx, dy = d, d * 0.866
    cells = []
    row = 0
    y = region["y"][0]
    while y <= region["y"][1]:
        x = cx + (d / 2 if row % 2 else 0.0) - 0.30
        while x <= region["x"][1] + 0.05:
            r = (x * x + y * y) ** 0.5
            if (region["x"][0] - 0.02 <= x <= region["x"][1] + 0.02
                    and region["y"][0] - 0.02 <= y <= region["y"][1] + 0.02
                    and region["r"][0] <= r <= region["r"][1]):
                cells.append((x, y))
            x += dx
        y += dy
        row += 1
    assert len(cells) >= n, f"hex grid too small: {len(cells)} < {n}"
    idx = rng.permutation(len(cells))[:n]
    pts = [(cells[i][0] + rng.uniform(-0.006, 0.006),
            cells[i][1] + rng.uniform(-0.006, 0.006)) for i in idx]
    # drop cells inside zone exclusion circles, top up from remaining cells
    clean = [(x, y) for (x, y) in cells
             if all((x - zx) ** 2 + (y - zy) ** 2 >= 0.085 ** 2 for zx, zy in ZONE_XY)]
    out = []
    for (x, y) in pts:
        if all((x - zx) ** 2 + (y - zy) ** 2 >= 0.085 ** 2 for zx, zy in ZONE_XY):
            out.append((x, y))
        else:
            c = clean[rng.integers(len(clean))]
            out.append((c[0] + rng.uniform(-0.006, 0.006), c[1] + rng.uniform(-0.006, 0.006)))
    return out


def build_scene(n_cubes: int, seed: int = 7) -> str:
    with open(os.path.join(HERE, "scene_template.xml"), "r", encoding="utf-8") as f:
        tpl = f.read()
    rng = np.random.default_rng(seed)
    region = SPAWN.get(n_cubes, SPAWN[50])
    pts = sample_positions(n_cubes, rng, region)
    colors = list(CUBE_COLORS.keys())
    frags = []
    for i in range(n_cubes):
        color = colors[i % len(colors)]
        frags.append(cube_fragment(i, color, (pts[i][0], pts[i][1], CUBE_REST_Z)))
    return tpl.replace("    <!--CUBES-->", "\n".join(frags))


def compile_scene(xml_text: str, out_mjb: str, name: str):
    model = mujoco.MjModel.from_xml_string(xml_text)
    mujoco.mj_saveModel(model, out_mjb, None)
    total_mass = float(np.sum(model.body_mass))
    info = dict(
        name=name,
        file=os.path.basename(out_mjb),
        nq=int(model.nq), nv=int(model.nv), nu=int(model.nu),
        nbody=int(model.nbody), ngeom=int(model.ngeom),
        total_mass_kg=round(total_mass, 2),
        timestep=float(model.opt.timestep),
    )
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=HERE)
    args = ap.parse_args()
    infos = []
    for n, fname in ((8, "scene_8.mjb"), (50, "scene_50.mjb")):
        xml = build_scene(n, seed=7)
        out = os.path.join(args.out, fname)
        info = compile_scene(xml, out, fname)
        infos.append(info)
        print(f"[compile_scene] {fname}: nq={info['nq']} nv={info['nv']} nu={info['nu']} "
              f"bodies={info['nbody']} mass={info['total_mass_kg']}kg")
        # keep the expanded XML for reference / debugging
        with open(os.path.join(args.out, fname.replace(".mjb", ".xml")), "w") as f:
            f.write(xml)
    with open(os.path.join(args.out, "scene_info.json"), "w") as f:
        json.dump(infos, f, indent=2)
    print("[compile_scene] done ->", args.out)


if __name__ == "__main__":
    main()
