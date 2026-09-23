// glue/sim_glue.cpp — MuJoCo glue implementation
#include "glue/sim_glue.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <random>
namespace pcs {

SimGlue::~SimGlue() {
  if (d_ik_) mj_deleteData(d_ik_);
  if (d_) mj_deleteData(d_);
  if (m_) mj_deleteModel(m_);
}

bool SimGlue::load_mjb_file(const char* path) {
  m_ = mj_loadModel(path, nullptr);
  if (!m_) { err_ = "mj_loadModel failed"; return false; }
  d_ = mj_makeData(m_);
  if (!d_) { err_ = "mj_makeData failed"; return false; }
  resolve_ids();
  d_ik_ = mj_makeData(m_);
  if (d_ik_) {
    // fingers fully open in the IK scratch (no influence, but keeps
    // kinematics sane)
    for (int g = 0; g < 2; ++g)
      if (jnt_grip_[g] >= 0)
        d_ik_->qpos[m_->jnt_qposadr[jnt_grip_[g]]] = kGripOpenFinger;
  }
  return true;
}

// ---- DLS-IK: Position + Ansatzachse (hand-z -> -z), Warmstart q_goal_ ----
// Realer Panda: Schulteroffset 0.0825 + 45°-Handmontage machen die alte
// planare Fold-Formel unbrauchbar. 12 Iterationen reichen im Warmstart
// (< 0,1 ms; auf dem Desktop gegenueber der Python-Referenz verifiziert).
namespace {
// solves (H) x = g, H = 7x7 symmetric positive definite (Gauss, partial pivot)
static bool solve7(float H[7][7], const float g[7], float x[7]) {
  float A[7][8];
  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 7; ++j) A[i][j] = H[i][j];
    A[i][7] = g[i];
  }
  for (int c = 0; c < 7; ++c) {
    int p = c;
    for (int r = c + 1; r < 7; ++r)
      if (fabsf(A[r][c]) > fabsf(A[p][c])) p = r;
    if (fabsf(A[p][c]) < 1e-10f) return false;
    if (p != c) for (int j = c; j < 8; ++j) { float t = A[c][j]; A[c][j] = A[p][j]; A[p][j] = t; }
    const float piv = A[c][c];
    for (int r = c + 1; r < 7; ++r) {
      const float f = A[r][c] / piv;
      if (f == 0.f) continue;
      for (int j = c; j < 8; ++j) A[r][j] -= f * A[c][j];
    }
  }
  for (int i = 6; i >= 0; --i) {
    float s = A[i][7];
    for (int j = i + 1; j < 7; ++j) s -= A[i][j] * x[j];
    x[i] = s / A[i][i];
  }
  return true;
}
}  // namespace

void SimGlue::solve_ik_down(float tx, float ty, float tz, float* q_out) {
  if (!d_ik_ || site_tcp_ < 0 || hand_body_ < 0) {
    for (int j = 0; j < 7; ++j) q_out[j] = q_goal_[j];
    return;
  }
  float q[7];
  for (int j = 0; j < 7; ++j) q[j] = q_goal_[j];
  const int qadr[7] = {m_->jnt_qposadr[jnt_arm_[0]], m_->jnt_qposadr[jnt_arm_[1]],
                       m_->jnt_qposadr[jnt_arm_[2]], m_->jnt_qposadr[jnt_arm_[3]],
                       m_->jnt_qposadr[jnt_arm_[4]], m_->jnt_qposadr[jnt_arm_[5]],
                       m_->jnt_qposadr[jnt_arm_[6]]};
  const int vadr[7] = {m_->jnt_dofadr[jnt_arm_[0]], m_->jnt_dofadr[jnt_arm_[1]],
                       m_->jnt_dofadr[jnt_arm_[2]], m_->jnt_dofadr[jnt_arm_[3]],
                       m_->jnt_dofadr[jnt_arm_[4]], m_->jnt_dofadr[jnt_arm_[5]],
                       m_->jnt_dofadr[jnt_arm_[6]]};
  const float lambda2 = 0.08f * 0.08f;
  float err2_best = 1e30f;
  float q_best[7];
  for (int j = 0; j < 7; ++j) q_best[j] = q[j];

  for (int it = 0; it < 12; ++it) {
    for (int j = 0; j < 7; ++j) d_ik_->qpos[qadr[j]] = q[j];
    mj_kinematics(m_, d_ik_);
    mj_comPos(m_, d_ik_);

    const mjtNum* sp = d_ik_->site_xpos + 3 * site_tcp_;
    const float epos[3] = {tx - (float)sp[0], ty - (float)sp[1], tz - (float)sp[2]};
    // approach axis: hand local z must point DOWN (-z world)
    const mjtNum* R = d_ik_->xmat + 9 * hand_body_;
    const float zx[3] = {(float)R[2], (float)R[5], (float)R[8]};  // column 2
    // cross(z, d) with d=(0,0,-1)
    const float er[3] = {-zx[1], zx[0], 0.f};
    const float err2 = epos[0]*epos[0] + epos[1]*epos[1] + epos[2]*epos[2]
                     + er[0]*er[0] + er[1]*er[1] + er[2]*er[2];
    if (err2 < err2_best) {
      err2_best = err2;
      for (int j = 0; j < 7; ++j) q_best[j] = q[j];
    }
    if (err2 < 1e-6f) break;

    mj_jacSite(m_, d_ik_, jacp_.data(), jacr_.data(), site_tcp_);

    float H[7][7] = {};
    float g[7] = {};
    for (int r = 0; r < 7; ++r) {
      // task rows: 3 position + 3 rotation
      float Jr[6];
      for (int row = 0; row < 6; ++row) {
        const mjtNum* J = (row < 3) ? (jacp_.data() + 3 * row)
                                    : (jacr_.data() + 3 * (row - 3));
        Jr[row] = (float)J[vadr[r]];
      }
      for (int c = 0; c < 7; ++c) {
        float Jc[6];
        for (int row = 0; row < 6; ++row) {
          const mjtNum* J = (row < 3) ? (jacp_.data() + 3 * row)
                                      : (jacr_.data() + 3 * (row - 3));
          Jc[row] = (float)J[vadr[c]];
        }
        float dot = 0;
        for (int row = 0; row < 6; ++row) dot += Jr[row] * Jc[row];
        H[r][c] += dot;
      }
      const float e6[6] = {epos[0], epos[1], epos[2], er[0], er[1], er[2]};
      float dot = 0;
      for (int row = 0; row < 6; ++row) dot += Jr[row] * e6[row];
      g[r] += dot;
    }
    for (int r = 0; r < 7; ++r) H[r][r] += lambda2;
    float dq[7];
    if (!solve7(H, g, dq)) break;
    for (int j = 0; j < 7; ++j) {
      q[j] += dq[j];
      if (m_->jnt_limited[jnt_arm_[j]]) {
        const float lo = (float)m_->jnt_range[2 * jnt_arm_[j]];
        const float hi = (float)m_->jnt_range[2 * jnt_arm_[j] + 1];
        q[j] = std::min(std::max(q[j], lo), hi);
      }
    }
  }

  // accept only if clearly better than staying put (never diverge)
  if (err2_best < 0.25f * 0.25f) {
    for (int j = 0; j < 7; ++j) q_out[j] = q_best[j];
  } else {
    for (int j = 0; j < 7; ++j) q_out[j] = q_goal_[j];
  }
}

bool SimGlue::load_mjb_memory(const uint8_t* data, size_t size) {
  // mj_loadModel only takes files; write a temp copy (Android cache dir)
  // — caller on Android passes a temp path via load_mjb_file instead.
  (void)data; (void)size;
  err_ = "use load_mjb_file on this platform";
  return false;
}

void SimGlue::resolve_ids() {
  jacp_.assign(3 * (size_t)std::max<size_t>((size_t)m_->nv, 1), 0.0);
  jacr_.assign(3 * (size_t)std::max<size_t>((size_t)m_->nv, 1), 0.0);
  site_tcp_ = mj_name2id(m_, mjOBJ_SITE, "tcp");
  hand_body_ = mj_name2id(m_, mjOBJ_BODY, "hand");
  for (int j = 0; j < 7; ++j) {
    char nm[32]; snprintf(nm, sizeof(nm), "joint%d", j + 1);
    jnt_arm_[j] = mj_name2id(m_, mjOBJ_JOINT, nm);
    snprintf(nm, sizeof(nm), "m%d", j + 1);
    act_arm_[j] = mj_name2id(m_, mjOBJ_ACTUATOR, nm);
  }
  act_grip_ = mj_name2id(m_, mjOBJ_ACTUATOR, "grip_act");
  jnt_grip_[0] = mj_name2id(m_, mjOBJ_JOINT, "finger_l_j");
  jnt_grip_[1] = mj_name2id(m_, mjOBJ_JOINT, "finger_r_j");
  finger_geom_[0] = mj_name2id(m_, mjOBJ_GEOM, "finger_l_g");
  finger_geom_[1] = mj_name2id(m_, mjOBJ_GEOM, "finger_r_g");
  cam_event_ = mj_name2id(m_, mjOBJ_CAMERA, "cam_event");

  static const char* zone_names[4] = {"zone_red", "zone_green", "zone_blue", "zone_yellow"};
  const int table_body = mj_name2id(m_, mjOBJ_BODY, "table");
  const float table_x = (float)m_->body_pos[3 * table_body + 0];
  for (int c = 0; c < 4; ++c) {
    zone_body_[c] = mj_name2id(m_, mjOBJ_BODY, zone_names[c]);
    if (zone_body_[c] >= 0) {
      // zone bodies are children of the table body
      zone_pos_[2 * c + 0] = m_->body_pos[3 * zone_body_[c] + 0] + table_x;
      zone_pos_[2 * c + 1] = m_->body_pos[3 * zone_body_[c] + 1] + 0.f;
    }
  }

  // cubes = every free joint not part of the arm
  n_cubes_ = 0;
  for (int j = 0; j < m_->njnt && n_cubes_ < 64; ++j) {
    if (m_->jnt_type[j] != mjJNT_FREE) continue;
    const int body = m_->jnt_bodyid[j];
    // find first geom of body
    int geom = -1;
    for (int g = m_->body_geomadr[body]; g < m_->body_geomadr[body] + m_->body_geomnum[body]; ++g) {
      if (g >= 0) { geom = g; break; }
    }
    cube_jnt_[n_cubes_] = j;
    cube_body_[n_cubes_] = body;
    cube_geom_[n_cubes_] = geom;
    // color from geom rgba
    const float* rgba = m_->geom_rgba + 4 * geom;
    int col = 0;
    if (rgba[0] > 0.7f && rgba[1] < 0.5f) col = 0;        // red
    else if (rgba[1] > 0.6f && rgba[0] < 0.4f) col = 1;   // green
    else if (rgba[2] > 0.7f && rgba[1] < 0.5f) col = 2;   // blue
    else col = 3;                                          // yellow
    cube_color_[n_cubes_] = col;
    n_cubes_++;
  }
}

void SimGlue::randomize_cubes(uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> ux(0.30f, 0.48f), uy(-0.30f, 0.30f),
      ua(0.f, 3.14159f);
  // vertical-approach fold locus: r ~ 0.452..0.498 (see docs/ARCHITEKTUR.md)
  auto sample = [&](float& x, float& y) {
    for (int t = 0; t < 20000; ++t) {
      x = ux(rng); y = uy(rng);
      const float r = sqrtf(x * x + y * y);
      if (r < 0.36f || r > 0.50f) continue;  // fold-IK reach band
      bool ok = true;
      for (int c = 0; c < 4; ++c)
        if ((x - zone_pos_[2 * c]) * (x - zone_pos_[2 * c])
            + (y - zone_pos_[2 * c + 1]) * (y - zone_pos_[2 * c + 1]) < 0.085f * 0.085f) { ok = false; break; }
      if (!ok) continue;
      for (int i = 0; i < n_cubes_; ++i) {
        mjtNum* qp = d_->qpos + m_->jnt_qposadr[cube_jnt_[i]];
        if ((qp[0] - x) * (qp[0] - x) + (qp[1] - y) * (qp[1] - y) < 0.06f * 0.06f) { ok = false; break; }
      }
      if (ok) return;
    }
  };
  for (int i = 0; i < n_cubes_; ++i) {
    mjtNum* qp = d_->qpos + m_->jnt_qposadr[cube_jnt_[i]];
    float x = 0, y = 0;
    sample(x, y);
    qp[0] = x; qp[1] = y; qp[2] = 0.275f;
    const float a = ua(rng);
    qp[3] = cosf(0.5f * a); qp[4] = 0.f; qp[5] = 0.f; qp[6] = sinf(0.5f * a);
    mjtNum* qv = d_->qvel + m_->jnt_dofadr[cube_jnt_[i]];
    qv[0] = qv[1] = qv[2] = qv[3] = qv[4] = qv[5] = 0.f;
  }
}

void SimGlue::reset_episode(uint64_t seed) {
  // arm to home, gripper open, cubes random
  for (int j = 0; j < 7; ++j) {
    d_->qpos[m_->jnt_qposadr[jnt_arm_[j]]] = kHomeQ[j];
    d_->qvel[m_->jnt_dofadr[jnt_arm_[j]]] = 0.f;
    q_goal_[j] = kHomeQ[j];
    q_des_prev_[j] = kHomeQ[j];
  }
  for (int g = 0; g < 2; ++g) {
    d_->qpos[m_->jnt_qposadr[jnt_grip_[g]]] = kGripOpenFinger;
    d_->qvel[m_->jnt_dofadr[jnt_grip_[g]]] = 0.f;
  }
  randomize_cubes(seed);
  for (int a = 0; a < m_->nu; ++a) d_->ctrl[a] = 0.f;
  mj_forward(m_, d_);
  // stale warmstart + stale contacts from the previous layout would inject
  // phantom impulses — clear solver caches explicitly.
  memset(d_->qacc_warmstart, 0, sizeof(mjtNum) * m_->nv);
  mj_forward(m_, d_);
}

void SimGlue::cam_event_pose(float pos[3], float fwd[3], float up[3],
                             float right[3], float* fovy) const {
  for (int k = 0; k < 3; ++k) pos[k] = (float)m_->cam_pos[3 * cam_event_ + k];
  // camera looks along -z of its frame; xyaxes="1 0 0 0 1 0"
  right[0] = 1; right[1] = 0; right[2] = 0;
  up[0] = 0; up[1] = 1; up[2] = 0;
  fwd[0] = 0; fwd[1] = 0; fwd[2] = -1;
  *fovy = (float)m_->cam_fovy[cam_event_];
}

// v1.3.0 motion manager: stable snapshot of the arm state. Runs on the loop
// thread right after step_cycle, BEFORE the next mj_step — reading qpos /
// site_xpos here is race-free by construction (single loop thread).
void SimGlue::arm_state(float q[7], float* grip, float tcp[3]) const {
  if (!m_ || !d_) {
    memset(q, 0, 7 * sizeof(float));
    if (grip) *grip = 0.f;
    if (tcp) memset(tcp, 0, 3 * sizeof(float));
    return;
  }
  for (int j = 0; j < 7; ++j)
    q[j] = jnt_arm_[j] >= 0 ? (float)d_->qpos[m_->jnt_qposadr[jnt_arm_[j]]] : 0.f;
  if (grip)
    *grip = jnt_grip_[0] >= 0 ? (float)d_->qpos[m_->jnt_qposadr[jnt_grip_[0]]]
                              : 0.f;
  if (tcp) {
    if (site_tcp_ >= 0)
      for (int k = 0; k < 3; ++k) tcp[k] = (float)d_->site_xpos[3 * site_tcp_ + k];
    else
      memset(tcp, 0, 3 * sizeof(float));
  }
}

void SimGlue::refresh_eval() {
  // grasp detection: both fingers in contact with the same cube geom
  grasped_ = false; contact_l_ = false; contact_r_ = false;
  for (int ci = 0; ci < 8; ++ci) {
    bool l = false, r = false;
    for (int k = 0; k < d_->ncon; ++k) {
      const mjContact& con = d_->contact[k];
      const int g1 = con.geom1, g2 = con.geom2;
      if (g1 == finger_geom_[0] && g2 == cube_geom_[ci]) l = true;
      if (g2 == finger_geom_[0] && g1 == cube_geom_[ci]) l = true;
      if (g1 == finger_geom_[1] && g2 == cube_geom_[ci]) r = true;
      if (g2 == finger_geom_[1] && g1 == cube_geom_[ci]) r = true;
    }
    if (l && r) { grasped_ = true; break; }
    contact_l_ |= l; contact_r_ |= r;
  }

  // cube positions + sorting eval
  sorted_ = 0; stacked_ = 0;
  for (int i = 0; i < n_cubes_; ++i) {
    const mjtNum* p = d_->xpos + 3 * cube_body_[i];
    cube_pos_[3 * i + 0] = (float)p[0];
    cube_pos_[3 * i + 1] = (float)p[1];
    cube_pos_[3 * i + 2] = (float)p[2];
    const int col = cube_color_[i];
    const float dx = p[0] - zone_pos_[2 * col + 0];
    const float dy = p[1] - zone_pos_[2 * col + 1];
    if (dx * dx + dy * dy < 0.062f * 0.062f && p[2] > 0.265f) {
      sorted_++;
      if (p[2] > 0.315f) stacked_++;   // above first layer
    }
  }
}

void SimGlue::step_cycle(Controller& c, ControllerOutput& out,
                         const uint8_t* frame, int fw, int fh, bool refresh) {
  // ---- physics (budget-critical part) ----
  auto t0 = std::chrono::steady_clock::now();
  for (int s = 0; s < substeps_; ++s) mj_step(m_, d_);
  auto t1 = std::chrono::steady_clock::now();
  out.stats.t_phys = (uint32_t)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

  refresh_eval();

  // ---- perception + control pipeline ----
  ControllerInput cin{};
  for (int j = 0; j < 7; ++j) {
    cin.q[j] = (float)d_->qpos[m_->jnt_qposadr[jnt_arm_[j]]];
    cin.qd[j] = (float)d_->qvel[m_->jnt_dofadr[jnt_arm_[j]]];
  }
  cin.grip = (float)d_->qpos[m_->jnt_qposadr[jnt_grip_[0]]];
  cin.grasped = grasped_;
  for (int k = 0; k < 3; ++k)
    cin.tcp_actual[k] = (float)d_->site_xpos[3 * site_tcp_ + k];
  for (int j = 0; j < 7; ++j) cin.q_goal_ik[j] = q_goal_[j];
  cin.contact_l = contact_l_;
  cin.contact_r = contact_r_;
  cin.frame_rgb = frame;
  cin.frame_w = fw;
  cin.frame_h = fh;
  cin.refresh_pulse = refresh;
  cin.dt = (float)m_->opt.timestep * substeps_;

  // provide zones once
  for (int col = 0; col < 4; ++col) {
    c.task.zones[col][0] = zone_pos_[2 * col + 0];
    c.task.zones[col][1] = zone_pos_[2 * col + 1];
  }

  c.cycle(cin, out);

  // ---- IK: closed-form fold solution of the cartesian target ----
  // (exact for our top-down grasp family; recomputed every cycle -> drift-free)
  const float* tgt = out.task.tcp_target;
  if (halted_) {
    // STOP: hold the current configuration (target = measured pose)
    for (int j = 0; j < 7; ++j) q_goal_[j] = cin.q[j];
    d_->ctrl[act_grip_] = kGripOpen;
  } else if (out.task.joint_hold) {
    for (int j = 0; j < 7; ++j) q_goal_[j] = out.task.q_goal[j];
  } else {
    // real Panda chain: warm-started DLS-IK (approach axis down)
    solve_ik_down(tgt[0], tgt[1], tgt[2], q_goal_);
  }
  have_tgt_ = true;

  // commit the IK solution as the task goal and synthesize the final q_des:
  //   q_des = q_start + (q_goal - q_start) * minjerk(s) + texture(q_add)
  for (int j = 0; j < 7; ++j) {
    out.task.q_goal[j] = q_goal_[j];
    out.q_des[j] = out.task.q_start[j]
      + (q_goal_[j] - out.task.q_start[j]) * minjerk(out.task.s)
      + out.q_add[j];
  }

  // ---- torques: tau = qfrc_bias + kp (q_des - q) - kd qd   (gravity/Coriolis
  //      compensation from mjData, PD around the synthesized trajectory) ----
  for (int j = 0; j < 7; ++j) {
    const float q = cin.q[j];
    const float qd = cin.qd[j];
    const float q_des = out.q_des[j];
    const float qd_des = (q_des - q_des_prev_[j]) / cin.dt;
    q_des_prev_[j] = q_des;
    float tau = (float)d_->qfrc_bias[m_->jnt_dofadr[jnt_arm_[j]]]
              + kp_[j] * (q_des - q) + kd_[j] * (qd_des - qd);
    const mjtNum lo = m_->actuator_ctrlrange[2 * act_arm_[j]];
    const mjtNum hi = m_->actuator_ctrlrange[2 * act_arm_[j] + 1];
    d_->ctrl[act_arm_[j]] = std::min(std::max((mjtNum)tau, lo), hi);
  }
  d_->ctrl[act_grip_] = out.grip_target;

  // single source of truth: commit the IK solution as the task goal and
  // synthesize the final q_des baseline from it
  for (int j = 0; j < 7; ++j) {
    out.task.q_goal[j] = q_goal_[j];
    out.q_des[j] = out.task.q_start[j]
      + (q_goal_[j] - out.task.q_start[j]) * minjerk(out.task.s)
      + out.q_add[j];
  }

  refresh_eval();
}

}  // namespace pcs
