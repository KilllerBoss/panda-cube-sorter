// pcs/rl_policy.cpp — episodic PPO-lite over the grasp-skill parameters
#include "pcs/rl_policy.h"
#include <cmath>
#include <algorithm>
#include <random>

namespace pcs {

void skill_defaults(SkillParams* p) {
  p->align_z     = 0.030f;   // stage-1 funnel stop 30 mm above grasp plane
  // v1.6.0 RECALIBRATION for the fixed (hand-x-down) approach: pads span
  // [tcp-0.010, tcp+0.074] vertically; fingertips at tcp-0.010 must clear
  // the table (0.25) and the pad span must contain the cube mid (0.275)
  // => grasp tcp in [0.26, 0.285]; default tcp = 0.270.
  p->grasp_z_off = 0.005f;
  p->grip_pre    = 0.052f;   // 53 mm Anfahrtsspalt (funnel)
  p->descend_t   = 2.4f;
  p->grasp_t     = 0.9f;
  p->lift_t      = 1.3f;
  p->xy_gate     = 0.008f;
  p->z_gate      = 0.012f;
  p->wiggle_amp  = 0.012f;
  p->iir_alpha   = 0.15f;
  p->rate_cap    = 0.006f;   // 0.6 rad/s
  p->kp_scale    = 1.0f;
  p->assist_k    = 260.f;
  p->release_s   = 0.60f;
}

const RlParamInfo* rl_param_table() {
  static const RlParamInfo T[kRlDim] = {
    {"ANFAHRT", 0.016f, 0.050f, 0.030f},   // 0 align_z
    {"TIEFE",  -0.010f, 0.015f, 0.005f},   // 1 grasp_z_off (tcp = mid - off)
    {"SPALT",   0.040f, 0.062f, 0.052f},   // 2 grip_pre
    {"T_ABSEN", 1.4f,   3.6f,   2.4f  },   // 3 descend_t
    {"T_GREIF", 0.5f,   1.6f,   0.9f  },   // 4 grasp_t
    {"T_HEBEN", 0.8f,   2.2f,   1.3f  },   // 5 lift_t
    {"TOR_XY",  0.004f, 0.020f, 0.008f},   // 6 xy_gate
    {"TOR_Z",   0.006f, 0.030f, 0.012f},   // 7 z_gate
    {"KORRIG",  0.004f, 0.020f, 0.012f},   // 8 wiggle_amp
    {"GLATT",   0.06f,  0.30f,  0.15f },   // 9 iir_alpha
    {"TEMPO",   0.003f, 0.012f, 0.006f},   // 10 rate_cap
    {"KRAFT",   0.70f,  1.40f,  1.00f },   // 11 kp_scale
    {"FEDER",   120.f,  420.f,  260.f },   // 12 assist_k
    {"ABGABE",  0.35f,  0.80f,  0.60f },   // 13 release_s
  };
  return T;
}

float skill_get(const SkillParams& p, int i) {
  const float* v = &p.align_z;
  return v[i];
}
void skill_set(SkillParams* p, int i, float v) {
  float* d = &p->align_z;
  d[i] = v;
}

void RlPolicy::reset() {
  skill_defaults(&mu_);
  skill_defaults(&best_);
  const RlParamInfo* T = rl_param_table();
  for (int i = 0; i < kRlDim; ++i) sig_[i] = 0.10f;   // gentle start: mu IS the working default
  st_ = RlStats{};
  recent_n_ = 0;
  fail_streak_ = 0;
  loaded_ = false;
  (void)T;
}

// z = normalized parameter (0..1)
static void to_z(const SkillParams& p, float z[kRlDim]) {
  const RlParamInfo* T = rl_param_table();
  for (int i = 0; i < kRlDim; ++i) {
    const float v = skill_get(p, i);
    z[i] = clampf((v - T[i].lo) / (T[i].hi - T[i].lo), 0.f, 1.f);
  }
}
static void from_z(const float z[kRlDim], SkillParams* p) {
  const RlParamInfo* T = rl_param_table();
  for (int i = 0; i < kRlDim; ++i)
    skill_set(p, i, T[i].lo + clampf(z[i], 0.f, 1.f) * (T[i].hi - T[i].lo));
}

float RlPolicy::log_prob(const float z[kRlDim]) const {
  float mu_z[kRlDim];
  to_z(mu_, mu_z);
  float lp = 0.f;
  for (int i = 0; i < kRlDim; ++i) {
    const float d = z[i] - mu_z[i];
    lp += -0.5f * d * d / (sig_[i] * sig_[i]) - logf(sig_[i]);
  }
  return lp;
}

void RlPolicy::sample(uint64_t seed, SkillParams* theta) {
  std::mt19937_64 rng(seed);
  float mu_z[kRlDim];
  to_z(mu_, mu_z);
  for (int i = 0; i < kRlDim; ++i) {
    std::normal_distribution<float> nd(mu_z[i], sig_[i]);
    last_z_[i] = clampf(nd(rng), 0.f, 1.f);
  }
  has_sample_ = true;
  from_z(last_z_, theta);
}

void RlPolicy::observe(float reward, bool success) {
  if (!has_sample_) return;   // no episode running -> nothing to learn from
  update_from_z(last_z_, reward, success);
  has_sample_ = false;
}

void RlPolicy::update_from_z(const float z[kRlDim], float reward,
                             bool success) {
  const float eps = 0.2f;         // PPO clip range
  const float lr_mu = 0.22f;      // mu step (normalized space)
  const float lr_sig = 0.06f;     // sigma step
  const float lp_old = log_prob(z);

  // ---- best-ever tracking (from the SAMPLED theta, not mu) ----
  if (reward > st_.reward_best) {
    st_.reward_best = reward;
    from_z(z, &best_);
  }

  // running baseline (value function proxy)
  st_.baseline += 0.05f * (reward - st_.baseline);
  float A = clampf(reward - st_.baseline, -2.f, 2.f);

  const SkillParams mu_old = mu_;
  float mu_z[kRlDim];
  to_z(mu_, mu_z);
  for (int i = 0; i < kRlDim; ++i) {
    const float d = z[i] - mu_z[i];
    // grad log pi wrt mu_i: (z-mu)/sigma^2  (positive-A reinforces)
    mu_z[i] += lr_mu * A * d / (sig_[i] * sig_[i]);
    mu_z[i] = clampf(mu_z[i], 0.f, 1.f);
    // grad log pi wrt sigma_i: ((z-mu)^2/sigma^2 - 1)/sigma
    const float gs = (d * d / (sig_[i] * sig_[i]) - 1.f) / sig_[i];
    sig_[i] *= expf(clampf(lr_sig * A * gs, -0.35f, 0.35f));
    sig_[i] = clampf(sig_[i], 0.035f, 0.25f);
  }
  from_z(mu_z, &mu_);

  // PPO-lite clip: if the sampled theta's log-prob under the NEW parameters
  // left the trust region against the advantage sign, revert mu (the update
  // was too aggressive for one step; sigma keeps its entropy adjustment)
  const float lp_new = log_prob(z);
  const float ratio = expf(lp_new - lp_old);
  if ((A >= 0.f && ratio < 1.f - eps) || (A < 0.f && ratio > 1.f + eps)) {
    mu_ = mu_old;
  }

  // bookkeeping
  st_.episodes++;
  if (success) st_.successes++;
  recent_[recent_n_ % kRecentWin] = success ? 1.f : 0.f;
  recent_n_++;
  int n = std::min(recent_n_, kRecentWin);
  float s = 0.f;
  for (int i = 0; i < n; ++i) s += recent_[i];
  st_.rate_recent = s / (float)n;
  st_.reward_last = reward;
  fail_streak_ = success ? 0 : fail_streak_ + 1;
}

void RlPolicy::apply_mean(SkillParams* p) const { *p = mu_; }

void RlPolicy::apply_best(SkillParams* p) {
  *p = best_;
  // re-center the policy on the best parameters, keep some exploration
  mu_ = best_;
  for (int i = 0; i < kRlDim; ++i) sig_[i] = std::max(0.06f, sig_[i] * 0.7f);
  if (p) *p = mu_;
}

float RlPolicy::mu_norm(int i) const {
  const RlParamInfo* T = rl_param_table();
  const float v = skill_get(mu_, i);
  return clampf((v - T[i].lo) / (T[i].hi - T[i].lo), 0.f, 1.f);
}

bool RlPolicy::save(const char* path) const {
  FILE* f = fopen(path, "wb");
  if (!f) return false;
  const char magic[8] = "PCSRL01";
  fwrite(magic, 1, 8, f);
  fwrite(&mu_, sizeof(SkillParams), 1, f);
  fwrite(sig_, sizeof(float), kRlDim, f);
  fwrite(&best_, sizeof(SkillParams), 1, f);
  fwrite(&st_, sizeof(RlStats), 1, f);
  fwrite(recent_, sizeof(float), kRecentWin, f);
  fwrite(&recent_n_, sizeof(int), 1, f);
  fwrite(&fail_streak_, sizeof(int), 1, f);
  fclose(f);
  return true;
}

bool RlPolicy::load(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) return false;
  char magic[8] = {0};
  if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "PCSRL01", 8) != 0) {
    fclose(f);
    return false;
  }
  bool ok = true;
  ok &= fread(&mu_, sizeof(SkillParams), 1, f) == 1;
  ok &= fread(sig_, sizeof(float), kRlDim, f) == kRlDim;
  ok &= fread(&best_, sizeof(SkillParams), 1, f) == 1;
  ok &= fread(&st_, sizeof(RlStats), 1, f) == 1;
  ok &= fread(recent_, sizeof(float), kRecentWin, f) == kRecentWin;
  ok &= fread(&recent_n_, sizeof(int), 1, f) == 1;
  ok &= fread(&fail_streak_, sizeof(int), 1, f) == 1;
  fclose(f);
  if (!ok) return false;
  loaded_ = true;
  return true;
}

}  // namespace pcs
