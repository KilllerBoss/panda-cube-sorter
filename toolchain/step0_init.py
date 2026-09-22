#!/usr/bin/env python3
"""toolchain/step0_init.py — initialize the random/fixed parts of the weight blob.

Generates: LSNN input projection, biases, predictive-coding feedback init,
fixed embedding projection, LoRA zeros, META scalars. Saved as init_weights.npz
(seed 1234 — reproducible; the same values are embedded in weights.bin).
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    rng = np.random.default_rng(1234)
    LSNN_IN, LSNN_N, NUM_BINS, EMB_DIM = 576, 128, 576, 32
    w = {}
    # sparse-ish random input projection (event bins -> neurons)
    lw = rng.normal(0, 0.09, (LSNN_IN, LSNN_N)).astype(np.float32)
    lw[rng.random(lw.shape) < 0.35] = 0.0
    w["lsnn_w"] = lw
    w["lsnn_b"] = (-0.02 + 0.01 * rng.normal(0, 1, LSNN_N)).astype(np.float32)
    w["lsnn_gain"] = np.ones(LSNN_N, np.float32)
    # predictive coding feedback (state -> predicted input), small init, adapts online
    w["pred_fb"] = (0.02 * rng.normal(0, 1, (LSNN_N, NUM_BINS))).astype(np.float32)
    # fixed embedding projection [err(288) ; a(128)] -> 32
    w["emb_proj"] = (0.045 * rng.normal(0, 1, (NUM_BINS + LSNN_N, EMB_DIM))).astype(np.float32)
    # LoRA init = zeros (adapter grows purely online)
    w["lora_a"] = np.zeros((8, 4), np.float32)
    w["lora_b"] = np.zeros((4, 8), np.float32)
    # META scalars: lambda_v, rho_a, beta_th, vth0, snn_eta, pred_eta, out_scale
    w["meta"] = np.array([0.85, 0.92, 0.35, 0.35, 0.0, 0.002, 1.0, 0.0], np.float32)
    np.savez(os.path.join(HERE, "init_weights.npz"), **w)
    print("[step0] init_weights.npz written:",
          {k: v.shape for k, v in w.items()})


if __name__ == "__main__":
    main()
