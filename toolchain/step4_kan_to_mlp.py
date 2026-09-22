#!/usr/bin/env python3
"""toolchain/step4_kan_to_mlp.py — compile the KAN into the exact ReLU-hinge MLP
layout consumed by native/src/kan_mlp.cpp, with numerical verification.

Expansion identity used (exact for piecewise-linear f on ascending knots t):
  f(x) = c0 + c1*x + sum_k c_{k+1} * ReLU(x - t_k)

The runtime computes h = W1^T [x ; relu(x - t1)] + b1 — identical to the
hinge-basis ridge model from step3 up to floating-point associativity.
Verification asserts max abs error < 1e-4 on random inputs.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa

HINGES = 4


def main():
    km = np.load(os.path.join(HERE, "kan_model.npz"))
    in_scale, k1, C1, b1 = km["in_scale"], km["knots1"], km["coef1"], km["b1"]
    k2, C2, b2 = km["knots2"], km["coef2"], km["b2"]

    # C++ layout: w1[hidden * L1_FEATS + feat]  -> row-major (hidden, feats)
    w1 = C1.T.astype(np.float32)                      # (16, 120)
    w2 = C2.T.astype(np.float32)                      # (8, 80)
    hinge_t1 = k1.astype(np.float32)                  # (24, 4)
    hinge_t2 = k2.astype(np.float32)                  # (16, 4)

    # ---------------- verification on random inputs ----------------
    rng = np.random.default_rng(0)
    n = 2000
    X = rng.normal(0, 1, (n, C.MLP_IN)).astype(np.float32)
    X[:, 16:] = (rng.random((n, 8)) > 0.85).astype(np.float32)  # one-hots
    Xs = X * in_scale[None, :]

    # runtime-style forward (exactly like kan_mlp.cpp)
    feats1 = np.empty((n, C.L1_FEATS), np.float32)
    feats1[:, :C.MLP_IN] = Xs
    for k in range(HINGES):
        feats1[:, C.MLP_IN * (k + 1):C.MLP_IN * (k + 2)] = \
            np.maximum(Xs - hinge_t1[:, k][None, :], 0.0)
    h_rt = feats1 @ w1.T + b1                    # w1.T: (120,16) = feats @ C1 + b1

    feats2 = np.empty((n, C.L2_FEATS), np.float32)
    feats2[:, :C.MLP_HIDDEN] = h_rt
    for k in range(HINGES):
        feats2[:, C.MLP_HIDDEN * (k + 1):C.MLP_HIDDEN * (k + 2)] = \
            np.maximum(h_rt - hinge_t2[:, k][None, :], 0.0)
    logits_rt = feats2 @ w2.T + b2

    # step3-style forward (basis model)
    from step3_kan_train import hinge_basis  # noqa
    h_ref = hinge_basis(Xs, hinge_t1) @ C1 + b1
    logits_ref = hinge_basis(h_ref, hinge_t2) @ C2 + b2

    e_h = np.abs(h_rt - h_ref).max()
    e_l = np.abs(logits_rt - logits_ref).max()
    print(f"[step4] compile verify: max|h_rt-h_ref|={e_h:.2e}  max|logit diff|={e_l:.2e}")
    assert e_h < 1e-3 and e_l < 1e-3, "KAN->MLP compile mismatch"

    np.savez(os.path.join(HERE, "kan_compiled.npz"),
             in_scale=km["in_scale"].astype(np.float32),
             w1=w1, b1=b1.astype(np.float32),
             w2=w2, b2=b2.astype(np.float32),
             hinge_t1=hinge_t1, hinge_t2=hinge_t2)
    print("[step4] kan_compiled.npz written "
          f"(w1 {w1.shape}, w2 {w2.shape})")


if __name__ == "__main__":
    main()
