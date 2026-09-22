// pcs/soft_moe.cpp — Soft-MoE vision gating (Phase 3)
#include "pcs/soft_moe.h"
#include "pcs/weights.h"

namespace pcs {

void softmoe_bias(const Weights& w, const float* emb, float* bias_out) {
  for (int o = 0; o < kMlpOut; ++o) {
    float acc = w.smoe_b[o];
    for (int d = 0; d < kEmbDim; ++d) acc += emb[d] * w.smoe_w[(size_t)d * kMlpOut + o];
    bias_out[o] = 1.5f * tanhf(acc);   // bounded modulation of mixture logits
  }
}

}  // namespace pcs
