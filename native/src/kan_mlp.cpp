// pcs/kan_mlp.cpp — compiled KAN forward pass (Phase 3)
#include "pcs/kan_mlp.h"
#include "pcs/weights.h"

namespace pcs {

void KanMlp::forward(const float* x_in, const Weights& w, const float* moe_bias,
                     KanMlpState& st) const {
  // input scaling
  for (int i = 0; i < kMlpIn; ++i) st.x[i] = x_in[i] * w.in_scale[i];

  // layer 1 features: [x ; ReLU(x - t_ij)]  (exact spline-hinge expansion)
  for (int i = 0; i < kMlpIn; ++i) st.l1_feats[i] = st.x[i];
  for (int i = 0; i < kMlpIn; ++i) {
    for (int k = 0; k < kHinges1; ++k) {
      const float t = w.hinge_t1[(size_t)i * kHinges1 + k];
      const float h = st.x[i] - t;
      st.l1_feats[kMlpIn + (size_t)i * kHinges1 + k] = h > 0.f ? h : 0.f;
    }
  }
  // h1 = act(W1^T * f1 + b1)  (tanh keeps the KAN in its trained range)
  for (int j = 0; j < kMlpHidden; ++j) {
    float acc = w.b1[j];
    const float* col = &w.w1[(size_t)j * kL1Feats];  // column j, row-major W1 [feats x hidden]
    for (int f = 0; f < kL1Feats; ++f) acc += st.l1_feats[f] * col[f];
    st.h1[j] = tanhf(acc);
  }

  // layer 2 features: [h1 ; ReLU(h1 - t2_jk)]
  for (int i = 0; i < kMlpHidden; ++i) st.l2_feats[i] = st.h1[i];
  for (int i = 0; i < kMlpHidden; ++i) {
    for (int k = 0; k < kHinges1; ++k) {
      const float t = w.hinge_t2[(size_t)i * kHinges1 + k];
      const float h = st.h1[i] - t;
      st.l2_feats[kMlpHidden + (size_t)i * kHinges1 + k] = h > 0.f ? h : 0.f;
    }
  }
  // logits = W2^T * f2 + b2 (+ Soft-MoE bias + LoRA delta applied by caller via bias ptr)
  for (int o = 0; o < kMlpOut; ++o) {
    float acc = w.b2[o];
    const float* col = &w.w2[(size_t)o * kL2Feats];
    for (int f = 0; f < kL2Feats; ++f) acc += st.l2_feats[f] * col[f];
    if (moe_bias) acc += moe_bias[o];
    st.logits[o] = acc;
  }
  // softmax over prototypes
  float mx = st.logits[0];
  for (int o = 1; o < kMlpOut; ++o) if (st.logits[o] > mx) mx = st.logits[o];
  float sum = 0.f;
  for (int o = 0; o < kMlpOut; ++o) {
    st.w[o] = expf(st.logits[o] - mx);
    sum += st.w[o];
  }
  const float inv = 1.f / sum;
  for (int o = 0; o < kMlpOut; ++o) st.w[o] *= inv;
}

}  // namespace pcs
