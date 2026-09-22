// pcs/weights.h — binary weight-blob loader (weights.bin, produced by toolchain/step6_export.py)
#pragma once
#include "pcs/types.h"
#include <string>
#include <vector>

namespace pcs {

// All matrices are row-major float32, little-endian.
struct Weights {
  // ---- prototype bank: residual curves P[r][dof][s] ----
  std::vector<float> proto;          // kNumProto x kDof x kProtoLen
  // ---- ALIF-LSNN ----
  std::vector<float> lsnn_w;         // kLsnnIn x kLsnnN  (input projection)
  std::vector<float> lsnn_b;         // kLsnnN
  std::vector<float> lsnn_gain;      // kLsnnN  (output feature scale)
  // scalars (from META floats)
  float lambda_v, rho_a, beta_th, vth0, snn_eta;
  // ---- predictive coding feedback (state->input prediction) ----
  std::vector<float> pred_fb;        // kLsnnN x kLsnnIn  (a -> u_hat)
  float pred_eta;                    // local learning rate
  // ---- embedding projection: [error(288) ; a(128)] -> 32 ----
  std::vector<float> emb_proj;       // (kLsnnIn + kLsnnN) x kEmbDim
  // ---- Soft-MoE vision gating ----
  std::vector<float> smoe_w;         // kEmbDim x kMlpOut
  std::vector<float> smoe_b;         // kMlpOut
  // ---- compiled KAN->MLP (exact ReLU-hinge expansion) ----
  std::vector<float> w1, b1;         // w1: kL1Feats x kMlpHidden ; b1: kMlpHidden
  std::vector<float> w2, b2;         // w2: kL2Feats x kMlpOut    ; b2: kMlpOut
  std::vector<float> hinge_t1;       // kMlpIn     x kHinges1 (knots, ascending)
  std::vector<float> hinge_t2;       // kMlpHidden x kHinges1
  // ---- decoder: embedding -> cube slots (x,y,color logits) ----
  std::vector<float> dec_w, dec_b;   // kDecIn x kDecOut, kDecOut
  // ---- LoRA init (usually zeros) ----
  std::vector<float> lora_a, lora_b; // kMlpHidden x kLoraRank, kLoraRank x kMlpOut
  // ---- input/output scaling ----
  std::vector<float> in_scale;       // kMlpIn
  float out_scale;

  bool load_from_memory(const uint8_t* data, size_t size);
  bool load_from_file(const char* path);
  std::string last_error;
};

}  // namespace pcs
