// pcs/pred_coder.h — unsupervised predictive coding (Free Energy Principle, Phase 2)
// The network predicts the next input bin pattern from its internal state;
// only the prediction error propagates. Error + adaptive state are projected
// through a fixed random projection to a compact 32-d embedding.
#pragma once
#include "pcs/types.h"
#include "pcs/alif_lsnn.h"

namespace pcs {

class PredCoder {
 public:
  void reset();
  // u = binned input (kNumBins), st = current SNN state
  // emb_out: kEmbDim embedding; also returns prediction error energy (FEP free energy proxy)
  void step(const float* u, const LsnnState& st, const Weights& w,
            float* emb_out, float& free_energy);

 private:
  float u_hat_[kNumBins];   // predicted next input
  float err_[kNumBins];     // prediction error (propagated)
};

}  // namespace pcs
