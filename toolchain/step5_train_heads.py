#!/usr/bin/env python3
"""toolchain/step5_train_heads.py — train perception heads on the SNN embedding.

Decoder:  emb(32) -> [x_off(8), y(8), color_logits(8x4)]   (ridge)
Soft-MoE: emb(32) -> mixture-logit bias (8)                (ridge on residual
           logits so softmax(KAN_logits + bias) reproduces the expert w*)
Outputs: heads.npz {dec_w, dec_b, smoe_w, smoe_b}
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa


def ridge(F, Y, lam=1e-3):
    d = F.shape[1]
    A = F.T @ F + lam * np.eye(d, dtype=np.float64)
    Cf = np.linalg.solve(A, F.T @ Y)
    b = Y.mean(0) - F.mean(0) @ Cf
    return Cf.astype(np.float32), b.astype(np.float32)


def main():
    ds = dict(np.load(os.path.join(HERE, "dataset.npz")))
    km = np.load(os.path.join(HERE, "kan_model.npz"))
    emb = ds["emb"].astype(np.float32)
    n = len(emb)
    print(f"[step5] {n} embedding samples")

    # ---------------- decoder ----------------
    cube_xy = ds["cube_xy"]  # (n, 8, 2)
    cube_color = ds["cube_color"]  # (n, 8)
    Y = np.zeros((n, C.DEC_OUT), np.float32)
    Y[:, 0:8] = cube_xy[:, :, 0] - 0.38
    Y[:, 8:16] = cube_xy[:, :, 1]
    for c in range(8):
        for k in range(4):
            Y[:, 16 + k * 8 + c] = np.where(cube_color[:, c] == k, 1.5, -1.0)
    tr = np.arange(0, n, 2)   # even = train, odd = val
    va = np.arange(1, n, 2)
    dec_w, dec_b = ridge(emb[tr], Y[tr])
    pred = emb[va] @ dec_w + dec_b
    pos_err = np.abs(pred[:, 0:8] - Y[va, 0:8]).mean() + \
              np.abs(pred[:, 8:16] - Y[va, 8:16]).mean()
    col_hit = (np.argmax(pred[:, 16:].reshape(-1, 8, 4), 2) == cube_color[va]).mean()
    print(f"[step5] decoder val: mean pos err {pos_err * 100:.2f} cm, color acc {col_hit:.2%}")

    # ---------------- Soft-MoE bias ----------------
    # expert w* (same as step3)
    from step3_kan_train import wstar_targets, build_inputs  # noqa
    P = np.load(os.path.join(HERE, "prototypes.npz"))["P"]
    Wstar = wstar_targets(ds, P)
    # KAN logits without MoE bias (training data, runtime-style)
    X = build_inputs(ds)
    Xs = X * km["in_scale"][None, :]
    from step3_kan_train import hinge_basis  # noqa
    h = hinge_basis(Xs, km["knots1"]) @ km["coef1"] + km["b1"]
    logits = hinge_basis(h, km["knots2"]) @ km["coef2"] + km["b2"]
    bias_t = np.log(np.maximum(Wstar, 1e-4)) - logits
    bias_t = np.clip(bias_t, -1.5, 1.5)
    smoe_w, smoe_b = ridge(emb[tr], bias_t[tr])
    print(f"[step5] softmoe bias fit: val |bias err| "
          f"{np.abs(emb[va] @ smoe_w + smoe_b - bias_t[va]).mean():.4f}")

    np.savez(os.path.join(HERE, "heads.npz"), dec_w=dec_w, dec_b=dec_b,
             smoe_w=smoe_w, smoe_b=smoe_b)
    print("[step5] heads.npz written")


if __name__ == "__main__":
    main()
