#!/usr/bin/env python3
"""toolchain/step3_kan_train.py — train the 2-layer KAN as additive
univariate-spline models via ridge regression on hinge bases.

Layer 1: h_j = sum_i f_ij(x_i)            (16 additive splines over 24 inputs)
Layer 2: logit_o = sum_j g_oj(h_j)        (8 additive splines over 16 hidden)

Targets w* (simplex) come from projecting the expert residual at each cycle
onto the NMF prototype bank sampled at that cycle's progress s.
Outputs: kan_model.npz {knots1, coef1, b1, knots2, coef2, b2, in_scale}
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa

HINGES = 4
QTS = (0.15, 0.40, 0.65, 0.90)


def hinge_basis(X, knots):
    """X (n,d), knots (d,HINGES) -> [X, relu(X - t)] (n, d*(1+HINGES))"""
    n, d = X.shape
    feats = np.empty((n, d * (1 + HINGES)), np.float32)
    feats[:, :d] = X
    for k in range(HINGES):
        feats[:, d * (k + 1):d * (k + 2)] = np.maximum(X - knots[:, k][None, :], 0.0)
    return feats


def fit_knots(Xs):
    d = Xs.shape[1]
    knots = np.empty((d, HINGES), np.float32)
    for i in range(d):
        q = np.quantile(Xs[:, i], QTS)
        if np.allclose(q, q[0]):  # degenerate (e.g. one-hot) -> tiny spread
            q = np.linspace(q[0] - 1e-3, q[0] + 1e-3, HINGES)
        knots[i] = q
    return knots


def ridge(F, Y, lam=1e-4):
    d = F.shape[1]
    A = F.T @ F + lam * np.eye(d, dtype=np.float64)
    B = F.T @ Y
    C = np.linalg.solve(A, B)
    return C.astype(np.float32), (Y.mean(0) - F.mean(0) @ C).astype(np.float32)


def build_inputs(ds):
    n = len(ds["phase"])
    X = np.empty((n, C.MLP_IN), np.float32)
    dq = ds["q_goal"] - ds["q_start"]
    X[:, 0:7] = dq
    X[:, 7:14] = ds["q_start"]
    X[:, 14] = ds["s"]
    X[:, 15] = ds["grip_target"]
    for p in range(C.NUM_PHASES):
        X[:, 16 + p] = (ds["phase"] == p).astype(np.float32)
    return X


def proto_sample(P, s):
    """P (8,7,64) sampled at scalar s -> (7,8) matrix of basis values."""
    x = np.clip(s, 0.0, 1.0) * (C.PROTO_LEN - 1)
    i0 = int(x)
    i1 = min(i0 + 1, C.PROTO_LEN - 1)
    fr = x - i0
    return (1 - fr) * P[:, :, i0] + fr * P[:, :, i1]  # (7? no: (8,7)) -> transpose


def wstar_targets(ds, P):
    """Instantaneous projection of expert residual on the sampled bank."""
    n = len(ds["phase"])
    W = np.zeros((n, C.NUM_PROTO), np.float32)
    q_des, q_start, q_goal, s, phase = ds["q_des"], ds["q_start"], ds["q_goal"], ds["s"], ds["phase"]
    for i in range(n):
        if phase[i] < C.PH_HOVER and phase[i] != C.PH_HOME:
            continue
        base = q_start[i] + (q_goal[i] - q_start[i]) * C.minjerk(s[i])
        r = np.maximum(q_des[i] - base, 0.0)  # positive envelope (bank is nn)
        B = proto_sample(P, s[i]).T           # (7, 8)
        # non-negative least squares approximation
        w = np.linalg.lstsq(B, r, rcond=None)[0]
        w = np.maximum(w, 0.0)
        tot = w.sum()
        W[i] = w / tot if tot > 1e-6 else 1.0 / C.NUM_PROTO
    # smooth within segments (EMA over consecutive samples)
    for i in range(1, n):
        W[i] = 0.6 * W[i] + 0.4 * W[i - 1]
    # renormalize rows
    W /= np.maximum(W.sum(1, keepdims=True), 1e-6)
    return W


def main():
    ds = dict(np.load(os.path.join(HERE, "dataset.npz")))
    prot = np.load(os.path.join(HERE, "prototypes.npz"))
    P = prot["P"]
    X = build_inputs(ds)
    in_scale = X.std(0)
    in_scale[14:16] = max(in_scale[14], 0.05), max(in_scale[15], 0.05)
    in_scale[16:] = 1.0
    in_scale = np.maximum(in_scale, 1e-3).astype(np.float32)
    Xs = X / in_scale[None, :]
    Wstar = wstar_targets(ds, P)
    print(f"[step3] {len(X)} samples, w* mean {Wstar.mean(0).round(3)}")

    # split train/val 90/10
    rng = np.random.default_rng(5)
    idx = rng.permutation(len(Xs))
    ntr = int(0.9 * len(Xs))
    tr, va = idx[:ntr], idx[ntr:]

    # ---- layer 1 (additive splines -> 16 hidden) ----
    k1 = fit_knots(Xs)
    F1 = hinge_basis(Xs, k1)
    C1, b1 = ridge(F1[tr], Wstar[tr])           # direct head: predicts w*
    h = F1 @ C1 + b1                            # hidden (pre-logits)
    print(f"[step3] layer1 val R2: "
          f"{1 - np.var((F1[va] @ C1 + b1 - Wstar[va]), 0).mean() / np.var(Wstar[va], 0).mean():.3f}")

    # ---- layer 2 (additive splines over hidden -> logits target) ----
    # target logits: log(w*) so that softmax(logit) ~ w*
    logits_t = np.log(np.maximum(Wstar, 1e-4))
    k2 = fit_knots(h)
    F2 = hinge_basis(h, k2)
    C2, b2 = ridge(F2[tr], logits_t[tr])
    logits_va = F2[va] @ C2 + b2
    w_va = np.exp(logits_va - logits_va.max(1, keepdims=True))
    w_va /= w_va.sum(1, keepdims=True)
    err = np.abs(w_va - Wstar[va]).mean()
    print(f"[step3] layer2 val mean |w - w*| = {err:.4f}")

    np.savez(os.path.join(HERE, "kan_model.npz"),
             in_scale=in_scale, knots1=k1, coef1=C1, b1=b1,
             knots2=k2, coef2=C2, b2=b2, hidden=h.mean(0) * 0)
    print("[step3] kan_model.npz written")


if __name__ == "__main__":
    main()
