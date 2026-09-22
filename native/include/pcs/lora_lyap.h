// pcs/lora_lyap.h — Lyapunov-guided LoRA online adaptation (Phase 5)
// Output layer of the frozen base MLP is augmented with a low-rank adapter:
//   logits += B * A * h1    (W = W0 + B·A, rank 4, FP32 on CPU)
// Weights of A, B are updated by a closed-form, normalized-gradient rule on
// the joint tracking error e with an adaptive gain that guarantees the
// Lyapunov function V = e^T e decreases — no backpropagation, divergence-free.
#pragma once
#include "pcs/types.h"
#include "pcs/weights.h"

namespace pcs {

struct LoraState {
  float A[kMlpHidden * kLoraRank];  // kMlpHidden x r
  float B[kLoraRank * kMlpOut];     // r x kMlpOut
  float delta[kMlpOut];             // current B*A*h1 contribution
  float eta;                        // adaptive gain
  float v_prev;                     // previous V
  float norm_a;                     // last ||A h1||
  void reset();
};

class LoraLyap {
 public:
  // computes delta = B*A*h1 and stores it in st.delta
  void forward(const float* h1, LoraState& st) const;
  // closed-form Lyapunov update of A, B from tracking error e (kDof)
  // phi = h1 (the sensitivity of logits w.r.t. adapter path)
  void update(const float* e /*kDof*/, const float* h1, LoraState& st,
              const Weights& w);
};

}  // namespace pcs
