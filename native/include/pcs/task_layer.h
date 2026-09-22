// pcs/task_layer.h — supervisory task state machine (sorting & stacking)
// Uses the decoded cube map (from the SNN embedding) to pick cubes, plan
// hover/descend/grasp/place targets, and provides phase + progress to the
// KAN-MLP. Contact feedback from the physics glue enables the grasp-wiggle
// correction. Cartesian targets are turned into joint goals by an incremental
// damped-LS IK inside the glue layer.
#pragma once
#include "pcs/types.h"

namespace pcs {

struct CubeSlot {          // decoded perception output per slot
  float x, y;              // table-plane position estimate (m)
  float color_logits[kNumColors];
  int   color;             // argmax
  float score;             // slot activity (confidence)
};

struct TaskInput {
  float q[kDof];           // current joint positions
  float grip;              // current gripper slide (m)
  bool  grasped;           // contact-based grasp flag (from glue)
  bool  contact_l, contact_r;
  float dt;                // seconds since last update (0.01)
  float tcp_actual[3];     // measured tcp position (site_xpos)
  float q_goal_ik[kDof];   // glue IK solution (previous cycle)
  CubeSlot cubes[kNumCubes];
  int   n_cubes;
};

struct TaskOutput {
  int   phase;             // Phase enum
  float s;                 // phase progress 0..1
  float phase_t;           // raw seconds in phase (for event timeouts)
  float q_start[kDof];     // posture frozen at phase start
  float q_goal[kDof];      // joint goal for this phase (IK from cartesian)
  float grip_target;       // 0..0.025 slide target
  float tcp_target[3];     // cartesian target (for glue IK)
  bool  ik_needed;         // phase transition: glue reseeds the IK here
  bool  joint_hold;        // task provides q_goal directly (carry/home hold)
  bool  episode_done;
  // HUD extras
  int   active_color;      // ColorId currently being sorted
  int   remaining;         // unprocessed cubes left
  int   in_flight;         // slot index of carried cube or -1
};

class TaskLayer {
 public:
  void reset();
  void update(const TaskInput& in, TaskOutput& out);

  float zones[4][2];               // zone centers xy (world), set by glue
  int   zone_stack[4] = {0,0,0,0}; // cubes placed per color
  int   next_color_ = 0;           // color currently being sorted
  bool  in_flight_ = false;
  int   flight_slot_ = -1;
  float phase_t_ = 0.f;
  float wiggle_dx_ = 0.f, wiggle_dy_ = 0.f;
  int   wiggle_tries_ = 0;
  int   slot_tries_[8] = {0,0,0,0,0,0,0,0};
  bool  slot_dead_[8] = {false,false,false,false,false,false,false,false};
  float phase_dur_ = 1.f;
  int   phase_ = PH_RESET;
  bool  done_ = false;
  int   cur_slot_ = -1;
  float q_start_[kDof] = {0};
  float grab_xy_[2] = {0};              // cube xy frozen at descent start
  int   prev_phase_ = PH_RESET;
  bool  have_grab_ = false;

 private:
  void start_phase(int p, const TaskInput& in, TaskOutput& out);
  void pick_next_slot(const TaskInput& in);
};

}  // namespace pcs
