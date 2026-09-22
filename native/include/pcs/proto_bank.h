// pcs/proto_bank.h — NMF motion-prototype bank (Phase 3)
// kNumProto residual trajectories per joint, kProtoLen samples each,
// extracted by toolchain/step2_nmf.py from scripted expert demonstrations.
// At runtime the bank is sampled at progress s and blended with the
// Soft-MoE/KAN mixture weights.
#pragma once
#include "pcs/types.h"
#include "pcs/weights.h"

namespace pcs {

class ProtoBank {
 public:
  // out_q_add[kDof] = sum_r w[r] * P[r][j][idx(s)]   (linear interp in s)
  void sample(const float* w /*kNumProto*/, float s, const Weights& wts,
              float* out_q_add) const;
};

}  // namespace pcs
