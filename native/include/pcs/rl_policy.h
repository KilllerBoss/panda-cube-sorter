// pcs/rl_policy.h — on-device reinforcement learning for the grasp skill
//
// v1.6.0 answers "Erfolgsquote soll bei 100% sein / finetunen mit RL":
// the hand-tuned constants of the v1.5.0 grasp-place skill (approach heights,
// gates, phase durations, goal-filter, PD scale, carry spring) become a
// 14-dimensional action vector theta. An EPISODIC PPO-LITE policy
//   pi(theta) = N(mu, diag sigma^2)      (normalized z-space per parameter)
// samples one theta per training episode, the episode runs in the MuJoCo
// simulation (fast mode: perception stages skipped, only physics + task +
// IK), and the return
//   r = 2*sorted/total + 0.5*stacked/total + 0.5*grasp_rate
//       - 0.4*fail_rate - 0.6*time_frac
// drives the clipped policy-gradient update with a running baseline:
//   mu  += lr_mu  * A * (z - mu)/sigma^2          (per dim)
//   sigma += lr_sig * A * ((z-mu)^2/sigma^2 - 1)/sigma
//   ratio clip (PPO): the update is dropped when the sampled log-prob under
//   the new parameters leaves [1-eps, 1+eps] against the advantage sign.
// sigma anneals to a floor as measured success climbs (entropy shrink).
// The best-ever theta is tracked; BEST copies it into mu (deterministic
// evaluation always runs mu).
//
// Engine-agnostic (no MuJoCo include) so the desktop harness trains with
// the exact same code that ships in the APK.
#pragma once
#include "pcs/types.h"
#include <cstdio>
#include <cstdint>

namespace pcs {

// Tunable skill parameters. Defaults = the v1.5.0 hand-tuned constants.
struct SkillParams {
  float align_z;      // stage-1 stop height above grasp plane (m)
  float grasp_z_off;  // tcp depth below cube mid-height at grasp (m)
  float grip_pre;     // pre-close tendon gap (m)
  float descend_t;    // stage-1 duration (s)
  float grasp_t;      // stage-2 duration (s)
  float lift_t;       // lift duration (s)
  float xy_gate;      // stage-1 XY tightness gate (m)
  float z_gate;       // stage-1 Z gate (m)
  float wiggle_amp;   // grasp-wiggle re-approach offset (m)
  float iir_alpha;    // goal low-pass alpha (per 10 ms cycle)
  float rate_cap;     // joint rate cap (rad per 10 ms cycle)
  float kp_scale;     // PD position gain scale (damping unchanged)
  float assist_k;     // carry-assist spring K (N/m)
  float release_s;    // PLACE release progress (0..1)
};

// v1.5.0 hand-tuned defaults (identical behavior before any training)
void skill_defaults(SkillParams* p);

// parameter table: names + bounds (also drives the UI bars)
static constexpr int kRlDim = 14;
struct RlParamInfo {
  const char* name;   // short label for the UI
  float lo, hi;       // hard bounds (theta is always clamped inside)
  float def;          // v1.5.0 default
};
const RlParamInfo* rl_param_table();

float skill_get(const SkillParams& p, int i);
void  skill_set(SkillParams* p, int i, float v);

struct RlStats {
  int   episodes = 0;        // training episodes completed
  int   successes = 0;       // episodes with sorted == total
  float rate_recent = 0.f;   // success rate over the last window
  float reward_last = 0.f;
  float reward_best = -1e9f;
  float baseline = 0.f;
};

// The policy. mu is ALWAYS what the running skill uses; training samples
// around it. All state is POD-ish so save/load is a straight fwrite.
class RlPolicy {
 public:
  void reset();                       // mu = defaults, sigma = sigma0
  // sample theta ~ N(mu, sigma^2) (clipped into bounds) — training episode.
  // The sampled z is remembered; observe() reuses it for the update.
  void sample(uint64_t seed, SkillParams* theta);
  // one episode finished: reward + success -> PPO-lite update on the
  // theta that sample() drew for this episode
  void observe(float reward, bool success);
  // deterministic evaluation: copy mu into p
  void apply_mean(SkillParams* p) const;
  // BEST action: copy the best-ever theta into mu (and re-center sigma)
  void apply_best(SkillParams* p);

  const SkillParams& mu() const { return mu_; }
  const RlStats& stats() const { return st_; }
  float sigma(int i) const { return sig_[i]; }
  bool  trained() const { return st_.episodes > 0 || loaded_; }
  void  mark_loaded() { loaded_ = true; }

  // persistence (device: internalDataPath/rl_policy.bin)
  bool save(const char* path) const;
  bool load(const char* path);

  // normalized 0..1 value of mu[i] for the UI bars
  float mu_norm(int i) const;

  static constexpr int kRecentWin = 20;   // "letzte N Episoden" window

 private:
  void update_from_z(const float z[kRlDim], float reward, bool success);
  float log_prob(const float z[kRlDim]) const;   // log N(z; mu, sigma^2) sum

  SkillParams mu_;
  float sig_[kRlDim];
  SkillParams best_;
  RlStats st_;
  float last_z_[kRlDim] = {0};       // z of the episode currently running
  bool  has_sample_ = false;
  float recent_[kRecentWin] = {0};   // 1 = success, 0 = failure
  int   recent_n_ = 0;
  int   fail_streak_ = 0;
  bool  loaded_ = false;
};

}  // namespace pcs
