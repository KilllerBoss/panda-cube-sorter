// pcs/kan_mlp.h — compiled Kolmogorov-Arnold Network (Phase 3)
// Each univariate B-spline of the KAN was exactly translated into piecewise
// linear ReLU hinges by toolchain/step4_kan_to_mlp.py:
//   f(x) = a0 + a1*x + sum_i c_i * ReLU(x - t_i)
// so the whole network runs as plain dense matrix multiplies (GEMV) — ideal
// for NEON/XNNPACK; no spline evaluation at runtime.
// Forward:  h1 = act(W1^T * [x ; hinge1(x)]) ; logits = W2^T * [h1 ; hinge2(h1)] + LoRA
#pragma once
#include "pcs/types.h"
#include "pcs/weights.h"

namespace pcs {

struct KanMlpState {
  float x[kMlpIn];
  float h1[kMlpHidden];
  float l1_feats[kL1Feats];
  float l2_feats[kL2Feats];
  float logits[kMlpOut];
  float w[kMlpOut];         // after softmax
};

class KanMlp {
 public:
  // forward pass; lora_delta (kMlpOut) may be nullptr.
  // If use_bias_in provided, it is added to logits (Soft-MoE vision bias).
  void forward(const float* x_in, const Weights& w, const float* moe_bias,
               KanMlpState& st) const;
};

}  // namespace pcs
