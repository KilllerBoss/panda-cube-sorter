// pcs/soft_moe.h — Soft Mixture-of-Experts gating (Phase 3)
// Vision embedding modulates the prototype mixture: bias = W_sm * emb + b_sm,
// added to the KAN-MLP output logits before softmax (continuous blending of
// approach/grasp/sort motion prototypes).
#pragma once
#include "pcs/types.h"
#include "pcs/weights.h"

namespace pcs {

void softmoe_bias(const Weights& w, const float* emb, float* bias_out /*kMlpOut*/);

}  // namespace pcs
