// glue/sim_glue.h — MuJoCo glue: model loading, reset, physics stepping,
// incremental IK, torque computation (PD + qfrc_bias), grasp/eval logic.
// Shared by the Android app (NDK) and the desktop harness.
#pragma once
#include <mujoco/mujoco.h>
#include <cstdint>
#include <vector>
#include "pcs/controller.h"

namespace pcs {

class SimGlue {
 public:
  ~SimGlue();

  bool load_mjb_file(const char* path);
  bool load_mjb_memory(const uint8_t* data, size_t size);
  const char* last_error() const { return err_.c_str(); }

  mjModel* model() { return m_; }
  mjData*  data()  { return d_; }

  void reset_episode(uint64_t seed);          // random cubes + home pose
  void randomize_cubes(uint64_t seed);        // cubes only (stress benchmark)

  // STOP button: freeze the arm at the current configuration, open gripper.
  // While halted, step_cycle keeps physics alive but no longer moves the arm.
  void set_halt(bool h) { halted_ = h; }
  bool halted() const { return halted_; }

  // one 10 ms control cycle: mj_step x substeps, then perception+control,
  // then torque write to ctrl. `frame` = event-camera RGB input.
  // v1.6.0: fast=true skips the perception/network stages (RL training:
  // physics + task layer + IK only, ~10x faster than realtime).
  void step_cycle(Controller& c, ControllerOutput& out,
                  const uint8_t* frame, int fw, int fh, bool refresh,
                  bool fast = false);

  // grasp / eval state
  bool grasped() const { return grasped_; }
  bool contact_l() const { return contact_l_; }
  bool contact_r() const { return contact_r_; }
  int  sorted_count() const { return sorted_; }
  int  stacked_count() const { return stacked_; }
  int  total_cubes() const { return n_cubes_; }
  int  cube_color(int i) const { return cube_color_[i]; }
  const float* cube_pos(int i) const { return &cube_pos_[3 * i]; }
  const float* zone_pos(int c) const { return &zone_pos_[2 * c]; }
  float q_goal(int j) const { return q_goal_[j]; }

  // camera intrinsics for the CPU projector (event camera: fixed scene camera)
  void cam_event_pose(float pos[3], float fwd[3], float up[3], float right[3],
                      float* fovy_deg) const;

  // v1.3.0 motion manager: stable read of the current arm state (actual
  // joint positions, gripper slide, tcp position) — used by the 100 Hz loop
  // to record motion clips without racing mjData
  void arm_state(float q[7], float* grip, float tcp[3]) const;

  // ---------------- v1.6.0 RL skill parameters + episode counters -------
  // skill() is the ACTIVE parameter set the skill reads every cycle (RL mu
  // or the sampled theta of the running training episode). The trainer on
  // the loop thread writes it between episodes — single-writer, no races.
  SkillParams& skill() { return skill_; }
  const SkillParams& skill() const { return skill_; }
  int ep_grasps() const { return ep_grasps_; }   // PH_GRASP -> PH_LIFT events
  int ep_fails() const { return ep_fails_; }     // grasp fails + drops

 private:
  void resolve_ids();
  void refresh_eval();
  // warm-started damped-least-squares IK (approach axis down) for the real
  // Menagerie Panda chain; runs on a private mjData, never on d_.
  // v1.5.0: optional 7th task row — hand-Y yaw (face alignment for flank
  // grips). yaw_valid=false leaves the wrist yaw free (nullspace).
  void solve_ik_down(float tx, float ty, float tz, float yaw, bool yaw_valid,
                     float* q_out);

  mjModel* m_ = nullptr;
  mjData*  d_ = nullptr;
  mjData*  d_ik_ = nullptr;   // IK scratch (kinematics only)
  std::string err_;

  // ids
  int site_tcp_ = -1;
  int hand_body_ = -1;
  int jnt_arm_[7] = {0}, act_arm_[7] = {0};
  int act_grip_ = -1;
  int jnt_grip_[2] = {-1, -1};
  int n_cubes_ = 0;
  int dbg_flight_ = 0;        // v1.5.0 debug: current flight/target slot
  int dbg_phase_ = 0;         // v1.5.0 debug: previous cycle's phase
  float dbg_s_ = 0.f;         // v1.5.0 debug: previous cycle's phase progress
  float dbg_tgt_[3] = {0};    // v1.5.0 debug: previous cycle's tcp target
  int cube_jnt_[64], cube_body_[64], cube_geom_[64], cube_color_[64];
  int zone_body_[4] = {-1, -1, -1, -1};
  int cam_event_ = -1;
  int finger_geom_[2] = {-1, -1};

  // scratch jacobian buffers (3 x nv), sized in resolve_ids()
  std::vector<mjtNum> jacp_, jacr_;

  // state
  float q_goal_[7] = {0}, q_des_prev_[7] = {0};
  float q_goal_smooth_[7] = {0};   // v1.5.0: IIR-smoothed IK goal (nullspace)
  float last_tgt_[3] = {0};
  bool  have_tgt_ = false;
  float cube_pos_[64 * 3] = {0};
  float zone_pos_[4 * 2] = {0};
  float kp_[7] = {120, 120, 100, 100, 60, 60, 40};
  float kd_[7] = {20, 20, 17, 17, 11, 11, 9};
  bool  grasped_ = false, contact_l_ = false, contact_r_ = false;
  int   sorted_ = 0, stacked_ = 0;
  int   substeps_ = 5;
  bool  halted_ = false;
  // v1.5.0 grasp assist (soft attachment): while a clamped cube is carried,
  // a critically damped spring pulls it to its recorded hand-relative pose.
  // The long Panda fingers + minjerk-free servoing leave a few mm of residual
  // error; without the assist that error ejects the cube during the lift.
  // The GRASP ITSELF (both pads in contact) stays a real, physics-detected
  // event — only the carry is assisted.
  bool  assist_ = false;
  int   assist_cube_ = -1;
  float assist_rel_[3] = {0};     // cube offset in hand frame at grasp time
  float assist_q_[4] = {1, 0, 0, 0};  // hand quat at grasp time
  // v1.6.0: RL skill parameters + per-episode grasp counters
  SkillParams skill_ = {};
  int ep_grasps_ = 0, ep_fails_ = 0;
  int ep_prev_phase_ = PH_RESET;
};

}  // namespace pcs
