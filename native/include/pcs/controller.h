// pcs/controller.h — per-cycle orchestration of the full pipeline (100 Hz)
//   events -> ALIF-LSNN -> predictive coding -> 32-d embedding
//          -> decoder (cube map) + Soft-MoE bias
//          -> compiled KAN-MLP (+ LoRA) -> mixture weights
//          -> prototype blending -> q_des -> (glue: torque -> mjData->ctrl)
#pragma once
#include "pcs/types.h"
#include "pcs/weights.h"
#include "pcs/event_camera.h"
#include "pcs/alif_lsnn.h"
#include "pcs/pred_coder.h"
#include "pcs/soft_moe.h"
#include "pcs/kan_mlp.h"
#include "pcs/lora_lyap.h"
#include "pcs/proto_bank.h"
#include "pcs/task_layer.h"

namespace pcs {

struct ControllerInput {
  float q[kDof];
  float qd[kDof];
  float grip;
  bool  grasped, contact_l, contact_r;
  float tcp_actual[3];        // measured tcp (site_xpos)
  float q_goal_ik[kDof];      // glue IK solution of the previous cycle
  const uint8_t* frame_rgb;   // event-camera input (w x h x 3)
  int   frame_w, frame_h;
  bool  refresh_pulse;        // camera micro-jitter flag this cycle
  float dt;
};

struct ControllerOutput {
  float q_des[kDof];
  float q_add[kDof];        // prototype-blend texture (baseline added by glue)
  float grip_target;
  TaskOutput task;            // includes counters for HUD/eval
  // perception snapshot for HUD/debug
  float emb[kEmbDim];
  EventFrame events;
  LsnnState snn;
  KanMlpState mlp;
  float free_energy;
  CycleStats stats;
};

class Controller {
 public:
  bool load_weights(const uint8_t* blob, size_t size);
  bool load_weights_file(const char* path);
  void reset();

  void cycle(const ControllerInput& in, ControllerOutput& out);

  Weights w;
  EventCamera cam;
  AlifLsnn snn;
  PredCoder coder;
  KanMlp mlp;
  LoraLyap lora;
  ProtoBank proto;
  TaskLayer task;
  LoraState lora_st;
};

}  // namespace pcs
