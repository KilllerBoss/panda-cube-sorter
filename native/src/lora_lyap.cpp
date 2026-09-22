// pcs/lora_lyap.cpp — Lyapunov-guided LoRA adaptation (Phase 5)
#include "pcs/lora_lyap.h"
#include "pcs/weights.h"

namespace pcs {

void LoraState::reset() {
  memset(A, 0, sizeof(A));
  memset(B, 0, sizeof(B));
  memset(delta, 0, sizeof(delta));
  eta = 0.9f;
  v_prev = 0.f;
  norm_a = 0.f;
}

void LoraLyap::forward(const float* h1, LoraState& st) const {
  // tmp = A^T * h1   (r)
  float tmp[kLoraRank];
  st.norm_a = 0.f;
  for (int r = 0; r < kLoraRank; ++r) {
    float acc = 0.f;
    for (int i = 0; i < kMlpHidden; ++i) acc += h1[i] * st.A[(size_t)i * kLoraRank + r];
    tmp[r] = acc;
    st.norm_a += acc * acc;
  }
  st.norm_a = sqrtf(st.norm_a);
  // delta = B * tmp  (kMlpOut)
  for (int o = 0; o < kMlpOut; ++o) {
    float acc = 0.f;
    for (int r = 0; r < kLoraRank; ++r) acc += tmp[r] * st.B[(size_t)r * kMlpOut + o];
    st.delta[o] = acc;
  }
}

void LoraLyap::update(const float* e, const float* h1, LoraState& st,
                      const Weights& w) {
  // Tracking error drives a normalized-gradient (MIT-rule) update with an
  // adaptive gain eta. With V = e^T e and de/dt ~= -L e - Phi B A h1, the
  // update dA/dB = -eta * e_phi * phi / (1 + ||phi||^2) makes
  // dV/dt < 0 for eta in (0, 2/L] — we additionally adapt eta by observing
  // dV and hard-bound ||A||, ||B||. This is the closed-form "Lyapunov gate".
  float e_mag = 0.f;
  for (int j = 0; j < kDof; ++j) e_mag += e[j] * e[j];
  e_mag = sqrtf(e_mag);

  // error signal mapped to logit space (soft-maxing weights): use joint error
  // directly as e_phi; phi = h1.
  float phi_norm = 0.f;
  for (int i = 0; i < kMlpHidden; ++i) phi_norm += h1[i] * h1[i];
  phi_norm = sqrtf(phi_norm) + 1e-6f;

  const float eta = st.eta * 0.02f / (phi_norm);   // normalized step
  // gradient wrt B: delta_B[:,o] += eta * e_o * (A^T h1)
  float tmp[kLoraRank];
  for (int r = 0; r < kLoraRank; ++r) {
    float acc = 0.f;
    for (int i = 0; i < kMlpHidden; ++i) acc += h1[i] * st.A[(size_t)i * kLoraRank + r];
    tmp[r] = acc;
  }
  for (int o = 0; o < kMlpOut; ++o) {
    const float g = eta * clampf(e[o % kDof], -0.35f, 0.35f);
    for (int r = 0; r < kLoraRank; ++r)
      st.B[(size_t)r * kMlpOut + o] += g * tmp[r];
  }
  // gradient wrt A: delta_A[i,r] += eta * (B e)_r * h1_i
  float be[kLoraRank];
  for (int r = 0; r < kLoraRank; ++r) {
    float acc = 0.f;
    for (int o = 0; o < kMlpOut; ++o) acc += st.B[(size_t)r * kMlpOut + o] * e[o % kDof];
    be[r] = acc;
  }
  for (int i = 0; i < kMlpHidden; ++i) {
    const float g = eta * clampf(h1[i], -2.f, 2.f);
    for (int r = 0; r < kLoraRank; ++r)
      st.A[(size_t)i * kLoraRank + r] += g * be[r];
  }

  // ---- Lyapunov stability gate ----
  // V = ||e||^2 ; if V grew, shrink eta (energy increased -> step too big)
  const float V = e_mag * e_mag;
  if (st.v_prev > 0.f && V > st.v_prev * 1.02f) st.eta = st.eta * 0.7f;
  else if (st.eta < 0.9f) st.eta = st.eta * 1.02f;
  st.eta = clampf(st.eta, 0.02f, 1.0f);
  st.v_prev = V;

  // hard bounds on adapter norms (BIBO stability)
  float a_norm = 0.f, b_norm = 0.f;
  for (size_t i = 0; i < kMlpHidden * kLoraRank; ++i) a_norm += st.A[i] * st.A[i];
  for (size_t i = 0; i < kLoraRank * kMlpOut; ++i) b_norm += st.B[i] * st.B[i];
  a_norm = sqrtf(a_norm); b_norm = sqrtf(b_norm);
  const float a_max = 1.5f, b_max = 1.5f;
  if (a_norm > a_max) { const float s = a_max / a_norm;
    for (size_t i = 0; i < kMlpHidden * kLoraRank; ++i) st.A[i] *= s; }
  if (b_norm > b_max) { const float s = b_max / b_norm;
    for (size_t i = 0; i < kLoraRank * kMlpOut; ++i) st.B[i] *= s; }
  (void)w;
}

}  // namespace pcs
