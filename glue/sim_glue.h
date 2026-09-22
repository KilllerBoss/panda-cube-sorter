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
  void step_cycle(Controller& c, ControllerOutput& out,
                  const uint8_t* frame, int fw, int fh, bool refresh);

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

 private:
  void resolve_ids();
  void refresh_eval();

  mjModel* m_ = nullptr;
  mjData*  d_ = nullptr;
  std::string err_;

  // ids
  int site_tcp_ = -1;
  int jnt_arm_[7] = {0}, act_arm_[7] = {0};
  int act_grip_ = -1;
  int jnt_grip_[2] = {-1, -1};
  int n_cubes_ = 0;
  int cube_jnt_[64], cube_body_[64], cube_geom_[64], cube_color_[64];
  int zone_body_[4] = {-1, -1, -1, -1};
  int cam_event_ = -1;
  int finger_geom_[2] = {-1, -1};

  // scratch jacobian buffers (3 x nv), sized in resolve_ids()
  std::vector<mjtNum> jacp_, jacr_;

  // state
  float q_goal_[7] = {0}, q_des_prev_[7] = {0};
  float last_tgt_[3] = {0};
  bool  have_tgt_ = false;
  float cube_pos_[64 * 3] = {0};
  float zone_pos_[4 * 2] = {0};
  float kp_[7] = {70, 70, 60, 60, 30, 30, 22};
  float kd_[7] = {9, 9, 8, 8, 5, 5, 4};
  bool  grasped_ = false, contact_l_ = false, contact_r_ = false;
  int   sorted_ = 0, stacked_ = 0;
  int   substeps_ = 5;
  bool  halted_ = false;
};

}  // namespace pcs
