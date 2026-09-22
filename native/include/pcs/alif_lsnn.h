// pcs/alif_lsnn.h — Adaptive Leaky Integrate-and-Fire LSNN (Phase 2)
// 128 neurons; adaptive threshold v_th_i(t) = vth0 * (1 + beta * a_i) where
// a_i is a slowly decaying spike trace => short-term memory of motion history.
#pragma once
#include "pcs/types.h"
#include "pcs/weights.h"
#include "pcs/event_camera.h"

namespace pcs {

struct LsnnState {
  float v[kLsnnN];
  float a[kLsnnN];      // adaptive trace (threshold modifier)
  uint8_t spike[kLsnnN];
  uint32_t n_spikes;
  void reset() { memset(v, 0, sizeof(v)); memset(a, 0, sizeof(a));
                 memset(spike, 0, sizeof(spike)); n_spikes = 0; }
};

class AlifLsnn {
 public:
  void reset();
  // bins: kNumBins activity values (EventFrame::bins)
  void step(const float* bins, const Weights& w, LsnnState& st);
};

}  // namespace pcs
