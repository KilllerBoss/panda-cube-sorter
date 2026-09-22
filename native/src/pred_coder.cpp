// pcs/pred_coder.cpp — unsupervised predictive coding (FEP) -> 32-d embedding
#include "pcs/pred_coder.h"
#include "pcs/weights.h"

namespace pcs {

void PredCoder::reset() {
  memset(u_hat_, 0, sizeof(u_hat_));
  memset(err_, 0, sizeof(err_));
}

void PredCoder::step(const float* u, const LsnnState& st, const Weights& w,
                     float* emb_out, float& free_energy) {
  // 1) predict next input from adaptive state: u_hat = PFB^T * a
  for (int j = 0; j < kNumBins; ++j) u_hat_[j] = 0.f;
  for (int i = 0; i < kLsnnN; ++i) {
    const float ai = st.a[i];
    if (ai <= 0.001f) continue;
    const float* row = &w.pred_fb[(size_t)i * kNumBins];
    for (int j = 0; j < kNumBins; ++j) u_hat_[j] += ai * row[j];
  }
  // 2) prediction error (only this propagates — free energy minimization)
  float fe = 0.f;
  for (int j = 0; j < kNumBins; ++j) {
    err_[j] = u[j] - u_hat_[j];
    fe += 0.5f * err_[j] * err_[j];
  }
  free_energy = fe;
  // 3) local Hebbian update of the feedback predictor (bounded, normalized)
  if (w.pred_eta > 0.f) {
    const float eps = 1e-4f;
    for (int i = 0; i < kLsnnN; ++i) {
      const float ai = st.a[i];
      if (ai <= 0.001f) continue;
      const float g = w.pred_eta * ai / (1.f + ai * ai);
      float* row = const_cast<float*>(&w.pred_fb[(size_t)i * kNumBins]);
      for (int j = 0; j < kNumBins; ++j) {
        row[j] += g * err_[j];
        // bound weights for stability
        if (row[j] > 0.5f) row[j] = 0.5f; else if (row[j] < -0.5f) row[j] = -0.5f;
      }
      (void)eps;
    }
  }
  // 4) fixed projection [err ; a] -> 32-d embedding, ReLU
  for (int d = 0; d < kEmbDim; ++d) {
    const float* row = &w.emb_proj[(size_t)d * (kNumBins + kLsnnN)];
    float acc = 0.f;
    for (int j = 0; j < kNumBins; ++j) acc += err_[j] * row[j];
    for (int i = 0; i < kLsnnN; ++i) acc += st.a[i] * row[kNumBins + i];
    emb_out[d] = acc > 0.f ? acc : 0.f;
  }
}

}  // namespace pcs
