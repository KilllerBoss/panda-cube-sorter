#!/usr/bin/env python3
"""toolchain/step2_nmf.py — extract motion prototypes from expert residuals.

Residual r(t) = q*_expert(t) - [q_start + (q_goal-q_start)*minjerk(s)]
per motion segment, resampled to 64 samples. Non-negative matrix
factorization (multiplicative updates) rank 8 -> prototype bank P (8x7x64)
and per-window activations H. Saved to prototypes.npz.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa

PROTO_LEN = 64
RANK = 8


def segment_residuals(ds):
    """Group consecutive cycles of motion phases into segments; resample to 64."""
    phase = ds["phase"]
    q_des = ds["q_des"]
    q_start = ds["q_start"]
    q_goal = ds["q_goal"]
    s = ds["s"]
    n = len(phase)
    motion = (phase >= C.PH_HOVER) | (phase == C.PH_HOME)
    windows = []
    i = 0
    while i < n:
        if not motion[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and motion[j + 1]:
            j += 1
        seg = np.arange(i, j + 1)
        if len(seg) >= 25:  # >= 0.25 s
            # residual over the segment
            base = q_start[seg] + (q_goal[seg] - q_start[seg]) * C.minjerk(s[seg])[:, None]
            res = q_des[seg] - base
            # resample segment to 64 samples
            x = np.linspace(0, 1, len(seg))
            xi = np.linspace(0, 1, PROTO_LEN)
            res64 = np.stack([np.interp(xi, x, res[:, d]) for d in range(7)], axis=1)
            windows.append(res64.reshape(-1))  # 7*64
        i = j + 1
    return np.array(windows, np.float32)


def nmf(V, rank, iters=400, seed=0):
    rng = np.random.default_rng(seed)
    n, m = V.shape
    W = np.abs(rng.normal(1, 0.1, (n, rank))).astype(np.float32) + 0.01
    H = np.abs(rng.normal(1, 0.1, (rank, m))).astype(np.float32) + 0.01
    eps = 1e-6
    for it in range(iters):
        H *= (W.T @ V) / (W.T @ W @ H + eps)
        W *= (V @ H.T) / (W @ H @ H.T + eps)
        if it % 100 == 0:
            r = np.linalg.norm(V - W @ H) / (np.linalg.norm(V) + eps)
            print(f"  nmf iter {it}: rel resid {r:.4f}")
    return W, H


def main():
    ds = dict(np.load(os.path.join(HERE, "dataset.npz")))
    V = segment_residuals(ds)
    print(f"[step2] {V.shape[0]} motion windows, dim {V.shape[1]}")
    # shift to non-negative (residuals are signed): NMF on the positive
    # envelope keeps the prototype bank additive (runtime blend is a
    # non-negative weighted sum).
    Vn = np.maximum(V, 0.0)
    W_act, P = nmf(Vn, RANK)      # W_act: windows x rank, P: rank x 448
    P = P.reshape(RANK, 7, PROTO_LEN)
    np.savez(os.path.join(HERE, "prototypes.npz"), P=P, act=W_act, V=Vn)
    print(f"[step2] prototypes.npz: P {P.shape}, mean |residual| "
          f"{np.abs(Vn).mean():.4f} rad")


if __name__ == "__main__":
    main()
