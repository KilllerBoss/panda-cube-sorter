// pcs/types.h — core constants & small helpers (engine-agnostic, no MuJoCo here)
// Panda Cube Sorter — native perception/control core.
#pragma once
#include <cstdint>
#include <cmath>
#include <cstring>

namespace pcs {

constexpr int kDof        = 7;    // arm joints
constexpr int kNumAct     = 8;    // 7 arm torques + 1 gripper position target
constexpr int kEmbDim     = 32;   // SNN embedding dimension (Phase 2 output)
constexpr int kNumProto   = 8;    // motion prototypes (NMF rank)
constexpr int kProtoLen   = 64;   // samples per prototype curve
constexpr int kNumCubes   = 8;    // active cubes in the sorting scene
constexpr int kNumColors  = 4;
constexpr int kNumPhases  = 8;    // task state machine phases
constexpr int kMlpIn      = 24;   // [dq_goal(7) q_start(7) s(1) grip(1) phase(8)]
constexpr int kMlpHidden  = 8;    // hidden width of the compiled KAN
constexpr int kMlpOut     = 8;    // mixture weights over prototypes (softmax)
constexpr int kLsnnN      = 128;  // ALIF neurons
constexpr int kLsnnIn     = 576;  // 12x8 grid x (2 polarity + 4 hue) = 576 bins
constexpr int kLoraRank   = 4;    // LoRA rank on MLP output layer
constexpr int kDecOut     = 48;   // decoder: x_off(8) y(8) color_logits(8x4=32)
constexpr int kDecIn      = kEmbDim + kLsnnIn;  // [embedding ; event bins] head input
constexpr int kHinges1    = 4;    // ReLU hinges per univariate spline (K=5 knots)
constexpr int kL1Feats    = kMlpIn + kMlpIn * kHinges1;        // 24 + 96 = 120
constexpr int kL2Feats    = kMlpHidden + kMlpHidden * kHinges1; // 16 + 64 = 80

// task phase ids
enum Phase : int {
  PH_RESET = 0, PH_HOME, PH_HOVER, PH_DESCEND,
  PH_GRASP, PH_LIFT, PH_TRANSPORT, PH_PLACE
};
static const char* kPhaseName[kNumPhases] = {
  "RESET", "HOME", "HOVER", "DESCEND", "GRASP", "LIFT", "TRANSPORT", "PLACE" };

// home posture (computed via scene/check_kinematics.py)
constexpr float kHomeQ[kDof] = {0.009f, -0.211f, 0.040f, -1.305f, -0.005f, 3.063f, 0.004f};
// gripper TENDON-LENGTH targets (tendon = sum of both finger slides; each
// slide in [0, 0.025]). Open: 0.05 -> 95 mm. Closed: 0.004 -> servo squeezes
// the 50 mm cube (contact at sum 0.017) with ~5 N clamp force.
constexpr float kGripClosed = 0.006f;
constexpr float kGripPre    = 0.022f;  // pre-close during descent (~55 mm gap)
constexpr float kGripOpen   = 0.05f;

// zone colors (order matches zone_red/green/blue/yellow bodies & cube colors)
enum ColorId : int { COL_RED = 0, COL_GREEN, COL_BLUE, COL_YELLOW };

// ---------- tiny math helpers (avoid Eigen dependency on device) ----------
inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }
// minimum-jerk interpolation profile g(s), s in [0,1]
inline float minjerk(float s) {
  s = clampf(s, 0.f, 1.f);
  return s * s * s * (10.f - 15.f * s + 6.f * s * s);
}
inline float minjerk_d(float s) {  // derivative g'(s) (per unit s)
  s = clampf(s, 0.f, 1.f);
  return 30.f * s * s * (s - 1.f) * (s - 1.f);
}

// Closed-form fold IK for the Panda-class chain with j3=j5=0 (no wrist rolls)
// and the approach axis pointing straight down (j2+j4+j6 = pi/2).
// Chain: L1 = 0.3985 (j2->j4), L2 = 0.3845 (j4->j6), L3 = 0.283 (j6->tcp).
// Planar equations (arm plane after yaw j1):
//   r       = L1 cos(a) + L2 cos(a+b)
//   z - 0.012 = -(L1 sin(a) + L2 sin(a+b)),  a=j2, b=j4
//   (0.012 = shoulder 0.333 - j6->tcp 0.321, tcp at the fingertip plane)
// => exact 2-link solution; j6 = pi/2 - a - b. Valid for r in ~[0.30, 0.52].
inline void q_fold(float az, float r, float z, float* q) {
  const float L1 = 0.3985f, L2 = 0.3845f;
  const float kPi2 = 1.5707963f;
  const float wx = r;
  const float wy = z - 0.012f;             // z minus shoulder-to-tcp base offset
  float d = sqrtf(wx * wx + wy * wy);
  d = clampf(d, 0.05f, L1 + L2 - 0.005f);
  float cb = (d * d - (L1 * L1 + L2 * L2)) / (2.f * L1 * L2);
  cb = clampf(cb, -1.f, 1.f);
  const float beta = acosf(cb);            // elbow fold (>0)
  const float gamma = atan2f(L2 * sinf(beta), L1 + L2 * cosf(beta));
  const float argW = atan2f(wy, wx);
  const float a = -(gamma + argW);         // j2 (negative: shoulder up)
  const float b = beta;                    // j4 (positive: elbow-UP branch —
                                           // keeps the elbow above the table)
  q[0] = az;
  q[1] = a;
  q[2] = 0.0f;
  q[3] = b;
  q[4] = 0.0f;
  q[5] = kPi2 - a - b;                     // j6
  q[6] = 0.0f;
}

// safe transit ("carry") posture family: base yaw aimed at azimuth az,
// arm folded so the tcp rides high with the approach axis pointing down.
inline void q_carry(float az, float* q) {
  q_fold(az, 0.40f, 0.47f, q);
}

// 100 Hz fixed-cycle stats: per-stage microseconds + derived medians
struct CycleStats {
  // per-stage durations (us), filled every cycle
  uint32_t t_phys, t_event, t_snn, t_moe, t_mlp, t_lora, t_task, t_total;
  uint32_t cycles;
  // events & spikes of last cycle (for HUD)
  uint32_t n_events, n_spikes;
  float    cpu_budget_frac;   // t_total / 10000us
  void reset() { memset(this, 0, sizeof(*this)); }
};

}  // namespace pcs
