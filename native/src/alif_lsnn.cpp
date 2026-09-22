// pcs/alif_lsnn.cpp — Adaptive LIF spiking network (Phase 2)
#include "pcs/alif_lsnn.h"
#include "pcs/weights.h"

namespace pcs {

void AlifLsnn::reset() {}

void AlifLsnn::step(const float* bins, const Weights& w, LsnnState& st) {
  // u_i = b_e * W[e,i] ; only bins with activity contribute (event-driven sparsity)
  for (int i = 0; i < kLsnnN; ++i) {
    float u = w.lsnn_b[i];
    st.v[i] = w.lambda_v * st.v[i] + u;  // leaky integrate (bias pre-folded)
  }
  for (int e = 0; e < kNumBins; ++e) {
    const float be = bins[e];
    if (be <= 0.001f) continue;
    const float* row = &w.lsnn_w[(size_t)e * kLsnnN];
    for (int i = 0; i < kLsnnN; ++i) st.v[i] += be * row[i];
  }
  // fire with adaptive threshold
  uint32_t nsp = 0;
  for (int i = 0; i < kLsnnN; ++i) {
    const float th = w.vth0 * (1.f + w.beta_th * st.a[i]);
    uint8_t s = st.v[i] > th ? 1u : 0u;
    if (s) { st.v[i] -= th; nsp++; }
    st.spike[i] = s;
    st.a[i] = w.rho_a * st.a[i] + (float)s;   // ALIF adaptive trace
  }
  st.n_spikes = nsp;
}

}  // namespace pcs
