// pcs/event_camera.cpp — simulated DVS-style event stream (Phase 2)
#include "pcs/event_camera.h"

namespace pcs {

void EventCamera::process(const uint8_t* rgb, int w, int h, bool jitter_refresh,
                          EventFrame& out) {
  // resample (nearest) to kEvW x kEvH luma + hue buckets
  static float luma[kEvW * kEvH];
  static float hue[kEvW * kEvH];
  const float sx = (float)w / kEvW, sy = (float)h / kEvH;
  for (int y = 0; y < kEvH; ++y) {
    const int sy0 = (int)(y * sy);
    for (int x = 0; x < kEvW; ++x) {
      const int sx0 = (int)(x * sx);
      const uint8_t* px = rgb + (sy0 * w + sx0) * 3;
      const float r = px[0] * (1.f / 255.f), g = px[1] * (1.f / 255.f), b = px[2] * (1.f / 255.f);
      const float L = 0.2126f * r + 0.7152f * g + 0.0722f * b;
      luma[y * kEvW + x] = L;
      // hue bucket: 0=r 1=g 2=b 3=yellow, -1 if achromatic/dark
      float hb = -1.f;
      const float mx = (r > g ? (r > b ? r : b) : (g > b ? g : b));
      const float mn = (r < g ? (r < b ? r : b) : (g < b ? g : b));
      if (mx - mn > 0.18f && mx > 0.15f) {
        if (mx == r && g > b + 0.12f) hb = 3.f;        // yellow-ish
        else if (mx == r) hb = 0.f;
        else if (mx == g) hb = 1.f;
        else hb = 2.f;
      }
      hue[y * kEvW + x] = hb;
    }
  }

  memset(out.bins, 0, sizeof(out.bins));
  out.n_events = 0;
  if (first_) {
    memcpy(luma_, luma, sizeof(luma_));
    memcpy(hue_, hue, sizeof(hue_));
    first_ = false;
    return;   // no reference frame yet
  }

  const float thr = 0.045f;                       // log-ish intensity change threshold
  const float inv_area = 1.f / (float)((kEvW / kBinCols) * (kEvH / kBinRows));
  for (int y = 0; y < kEvH; ++y) {
    const int br = y / (kEvH / kBinRows);
    for (int x = 0; x < kEvW; ++x) {
      const int idx = y * kEvW + x;
      const float dL = luma[idx] - luma_[idx];
      if (dL > thr || dL < -thr) {
        const int bc = x / (kEvW / kBinCols);
        const int cell = br * kBinCols + bc;
        out.bins[cell * kBinsPerCell + (dL > 0 ? 0 : 1)] += 1.f;
        out.n_events++;
        // hue evidence: color of the pixel that changed
        if (hue[idx] >= 0.f) {
          out.bins[cell * kBinsPerCell + 2 + (int)hue[idx]] += 1.f;
        }
      }
      luma_[idx] = luma[idx];
      hue_[idx] = hue[idx];
    }
  }
  if (jitter_refresh) {
    // refresh pulse: static scene re-emitted events this frame (already counted);
    // scale bins up slightly so a pulse dominates stale state.
    for (int i = 0; i < kNumBins; ++i) out.bins[i] *= 1.35f;
  }
  // normalize bins to [0, ~2]
  for (int i = 0; i < kNumBins; ++i) {
    out.bins[i] *= inv_area;
    if (out.bins[i] > 2.f) out.bins[i] = 2.f;
  }
}

}  // namespace pcs
