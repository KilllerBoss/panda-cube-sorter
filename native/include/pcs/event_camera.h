// pcs/event_camera.h — simulated event camera from a low-res RGB frame (Phase 2)
// Computes per-pixel log-intensity change between consecutive frames and bins
// bipolar events (+1/-1 polarity) plus hue-activity into a 12x8 grid.
#pragma once
#include "pcs/types.h"

namespace pcs {

constexpr int kEvW = 96, kEvH = 72;          // event-camera resolution
constexpr int kBinCols = 12, kBinRows = 8;   // spatial binning
constexpr int kBinsPerCell = 6;              // 2 polarity (lum) + 4 hue
constexpr int kNumBins = kBinCols * kBinRows * kBinsPerCell;  // = 576 = kLsnnIn

struct EventFrame {
  uint32_t n_events;          // events in this frame delta
  float    bins[kNumBins];    // normalized per-bin activity -> SNN input
};

class EventCamera {
 public:
  // feed an RGB888 buffer (w*h*3); internally converts to float luma + hue.
  // `jitter_refresh` marks a refresh pulse (camera micro-jitter) — during a
  // pulse the whole static scene re-emits events (solves "static = no events").
  void process(const uint8_t* rgb, int w, int h, bool jitter_refresh, EventFrame& out);

 private:
  float luma_[kEvW * kEvH];
  float hue_[kEvW * kEvH];        // dominant hue bucket 0..3 (r,g,b,y), -1 = gray
  bool  first_ = true;
};

}  // namespace pcs
