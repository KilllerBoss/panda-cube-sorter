// pcs/proto_bank.cpp — NMF prototype bank sampling (Phase 3)
#include "pcs/proto_bank.h"
#include "pcs/weights.h"

namespace pcs {

void ProtoBank::sample(const float* w, float s, const Weights& wts,
                       float* out_q_add) const {
  // map s in [0,1] to sample index with linear interpolation
  const float x = clampf(s, 0.f, 1.f) * (kProtoLen - 1);
  const int i0 = (int)x;
  const int i1 = i0 + 1 < kProtoLen ? i0 + 1 : kProtoLen - 1;
  const float fr = x - (float)i0;
  for (int j = 0; j < kDof; ++j) out_q_add[j] = 0.f;
  for (int r = 0; r < kNumProto; ++r) {
    const float wr = w[r];
    if (wr < 1e-4f) continue;
    const float* P = &wts.proto[((size_t)r * kDof + 0) * kProtoLen + 0];
    for (int j = 0; j < kDof; ++j) {
      const float* pj = P + (size_t)j * kProtoLen;
      out_q_add[j] += wr * lerpf(pj[i0], pj[i1], fr);
    }
  }
}

}  // namespace pcs
