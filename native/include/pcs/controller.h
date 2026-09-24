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
  // v1.5.0 "privileged perception": the simulator's true cube slots. When
  // non-null the task layer targets THESE instead of the analytic detection
  // head — the old head produced phantom slots (all score <= 0.03 on the real
  // scene), the arm grasped empty space, every slot died and the episode
  // ended without a single grasp ("Warum kann der Roboter nichts").
  // The event/LSNN/embedding/MoE pipeline still runs every cycle and is shown
  // live in the NN window; only the SLOT SOURCE changes.
  const struct CubeSlot* truth_slots;
  int   truth_n;
  // v1.6.0 RL fast mode: skip the perception/network stages (event camera,
  // LSNN, predictive coding, MoE, MLP, LoRA, prototypes). The task layer is
  // driven purely by the truth slots — exactly what RL training needs, at
  // ~10x realtime. skill = active skill parameter set (null = defaults).
  bool  fast = false;
  const struct SkillParams* skill = nullptr;
};

struct ControllerOutput {
  float q_des[kDof];
  float q_add[kDof];        // prototype-blend texture (baseline added by glue)
  float grip_target;
  TaskOutput task;            // includes counters for HUD/eval
  // perception snapshot for HUD/debug
  float emb[kEmbDim];
  CubeSlot dbg_slots[kNumCubes];   // decoded slots (debug/eval)
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

  // Finetune button: run a short LoRA burst on the last stored tracking
  // error (same analytic Lyapunov update, elevated iteration count).
  void finetune(int iters);

  Weights w;
  EventCamera cam;
  AlifLsnn snn;
  PredCoder coder;
  KanMlp mlp;
  LoraLyap lora;
  ProtoBank proto;
  TaskLayer task;
  LoraState lora_st;
  float pulse_bins_[576] = {0};   // latched refresh-pulse event snapshot

 private:
  // last cycle's adaptation sample (kept for the finetune burst)
  float last_e_[kDof] = {0};
  float last_h1_[kMlpHidden] = {0};
  bool has_sample_ = false;
};

}  // namespace pcs
