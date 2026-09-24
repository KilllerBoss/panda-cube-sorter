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
  skill_defaults(&skill_);   // v1.6.0: RL skill params start at v1.5.0 values
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

void SimGlue::solve_ik_down(float tx, float ty, float tz, float yaw_t,
                            bool yaw_valid, float* q_out) {
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
  // v1.5.0 component-wise acceptance: the old SUM rule (pos+axis+yaw < 0.25 m)
  // dead-locked once the yaw task existed — a 0.57 rad yaw error alone pushed
  // every solution above the threshold and q_goal froze forever. Now:
  // accept if position AND approach-axis are tight; the yaw heals itself
  // over cycles (the DLS keeps reducing it from the warm start).
  // Position cap 5 mm: the old 20 mm cap let the tcp park beside the cube,
  // one finger pressed the cube's flank and the grasp became a one-sided
  // shove (only finger0 ever made contact).
  float best_pos2 = 1e30f, best_ax2 = 1e30f;
  float q_best[7];
  for (int j = 0; j < 7; ++j) q_best[j] = q[j];
  // v1.6.0: nullspace posture regularization — the 7 task rows (pos+axis+yaw)
  // leave 1 dof free; without a posture pull the DLS picks branches where
  // link3 RESTS ON THE TABLE (l3<->table_g contact): the shoulder is then
  // locked by contact friction and 42 Nm of PD torque cannot lift it (the
  // tcp froze 45 mm above every target). Pulling toward the home posture
  // keeps the elbow above the table — the healthy grasp family.
  const float w_post = 0.15f;

  for (int it = 0; it < 12; ++it) {
    for (int j = 0; j < 7; ++j) d_ik_->qpos[qadr[j]] = q[j];
    mj_kinematics(m_, d_ik_);
    mj_comPos(m_, d_ik_);

    const mjtNum* sp = d_ik_->site_xpos + 3 * site_tcp_;
    const float epos[3] = {tx - (float)sp[0], ty - (float)sp[1], tz - (float)sp[2]};
    // approach axis: hand local z must point DOWN (-z world)
    const mjtNum* R = d_ik_->xmat + 9 * hand_body_;
    // v1.6.0 ROOT-CAUSE FIX (0% Erfolg, "Roboter kann nichts"):
    // the approach axis task aligned hand local Z with world -z — but the
    // Menagerie hand's APPROACH IS LOCAL +X (tcp at hand-frame (0.126,0,0),
    // fingers extend along +x, closing along +y). Aligning z-down left the
    // fork pointing HORIZONTALLY: the arm descended palm/elbow-first onto
    // the table (ncon ~32 resting contacts), stalled ~45 mm above every
    // target and shoved cubes around — 0% success, exactly the field
    // report. Now hand local +X (the fork axis) must point DOWN.
    const float xx[3] = {(float)R[0], (float)R[3], (float)R[6]};  // column 0
    // cross(x, d) with d=(0,0,-1)
    const float er[3] = {-xx[1], xx[0], 0.f};
    // v1.5.0 yaw task: hand local Y (fingers slide along hand-y) projected to
    // the horizontal plane must align with a face family of the cube. Faces
    // repeat every 90 deg, so the yaw error wraps into [-45 deg, 45 deg].
    float eyaw = 0.f;
    if (yaw_valid) {
      const float yx = (float)R[1], yy = (float)R[4];   // column 1 = hand Y
      float e = atan2f(yy, yx) - yaw_t;
      e = fmodf(e + 0.25f * (float)M_PI, 0.5f * (float)M_PI);
      if (e < 0.f) e += 0.5f * (float)M_PI;
      eyaw = e - 0.25f * (float)M_PI;   // in [-pi/4, pi/4]
      // desired hand rotation about world z = -eyaw (brings projection back)
      eyaw = -eyaw;
    }
    float post2 = 0.f;
    for (int j = 0; j < 7; ++j) { const float dqh = kHomeQ[j] - q[j]; post2 += dqh * dqh; }
    const float err2 = epos[0]*epos[0] + epos[1]*epos[1] + epos[2]*epos[2]
                     + er[0]*er[0] + er[1]*er[1] + er[2]*er[2]
                     + eyaw * eyaw;
    const float cost = err2 + w_post * post2;
    if (cost < err2_best) {
      err2_best = cost;
      best_pos2 = epos[0]*epos[0] + epos[1]*epos[1] + epos[2]*epos[2];
      best_ax2  = er[0]*er[0] + er[1]*er[1] + er[2]*er[2];
      for (int j = 0; j < 7; ++j) q_best[j] = q[j];
    }
    if (err2 < 1e-6f) break;

    mj_jacSite(m_, d_ik_, jacp_.data(), jacr_.data(), site_tcp_);

    float H[7][7] = {};
    float g[7] = {};
    // v1.5.0 CRITICAL FIX: mj_jacSite writes a (3, nv) row-major Jacobian —
    // the row stride is nv, NOT 3. The old code read `jacp + 3*row`, i.e.
    // mixed three adjacent dof columns of row 0 into the "rows" — the DLS
    // then minimized a garbage quadratic form and stalled ~0.24 m from every
    // target (accept threshold 0.25 m let the stall pass!). The arm never
    // left its hover pose: "Warum kann der Roboter nichts".
    const int nv_ = m_->nv;
    // task rows: 3 position + 3 rotation + 1 yaw (world-z rotation row)
    float Jr[7], Jc[7], e7[7];
    e7[0] = epos[0]; e7[1] = epos[1]; e7[2] = epos[2];
    e7[3] = er[0];   e7[4] = er[1];   e7[5] = er[2];
    e7[6] = eyaw;
    for (int r = 0; r < 7; ++r) {
      for (int row = 0; row < 7; ++row) {
        const mjtNum* J;
        if (row < 3)      J = jacp_.data() + (size_t)nv_ * row;
        else if (row < 6) J = jacr_.data() + (size_t)nv_ * (row - 3);
        else              J = jacr_.data() + (size_t)nv_ * 2;  // world-z row
        Jr[row] = (float)J[vadr[r]];
      }
      for (int c = 0; c < 7; ++c) {
        for (int row = 0; row < 7; ++row) {
          const mjtNum* J;
          if (row < 3)      J = jacp_.data() + (size_t)nv_ * row;
          else if (row < 6) J = jacr_.data() + (size_t)nv_ * (row - 3);
          else              J = jacr_.data() + (size_t)nv_ * 2;
          Jc[row] = (float)J[vadr[c]];
        }
        float dot = 0;
        for (int row = 0; row < 7; ++row) dot += Jr[row] * Jc[row];
        H[r][c] += dot;
      }
      float dot = 0;
      for (int row = 0; row < 7; ++row) dot += Jr[row] * e7[row];
      g[r] += dot;
    }
    for (int r = 0; r < 7; ++r) H[r][r] += lambda2;
    float dq[7];
    const bool solved = solve7(H, g, dq);
    if (!solved) break;

    // v1.6.0: NULLSPACE-projected posture pull. The naive bias
    // (adding w*(q_home-q) to the gradient) fights the task and leaves a
    // ~10-15 cm steady-state tracking error. Instead: pull ONLY in the
    // nullspace of the task Jacobian:
    //   dq += w * (dq_post - H^{-1} J^T (J dq_post))
    // which is the task-free component of the home pull — the elbow drifts
    // toward the healthy above-table family WITHOUT any tracking offset.
    {
      float dqp[7], jt[7], jtJ[7], dq_corr[7];
      for (int r = 0; r < 7; ++r) dqp[r] = kHomeQ[r] - q[r];
      // jt[row] = J[row][:] * dqp  (7 task rows)
      for (int row = 0; row < 7; ++row) {
        float acc = 0.f;
        for (int c = 0; c < 7; ++c) {
          const mjtNum* J;
          if (row < 3)      J = jacp_.data() + (size_t)nv_ * row;
          else if (row < 6) J = jacr_.data() + (size_t)nv_ * (row - 3);
          else              J = jacr_.data() + (size_t)nv_ * 2;
          acc += (float)J[vadr[c]] * dqp[c];
        }
        jt[row] = acc;
      }
      // jtJ[c] = J^T jt  (joint-space vector)
      for (int c = 0; c < 7; ++c) {
        float acc = 0.f;
        for (int row = 0; row < 7; ++row) {
          const mjtNum* J;
          if (row < 3)      J = jacp_.data() + (size_t)nv_ * row;
          else if (row < 6) J = jacr_.data() + (size_t)nv_ * (row - 3);
          else              J = jacr_.data() + (size_t)nv_ * 2;
          acc += (float)J[vadr[c]] * jt[row];
        }
        jtJ[c] = acc;
      }
      if (solve7(H, jtJ, dq_corr)) {
        for (int r = 0; r < 7; ++r)
          dq[r] += w_post * (dqp[r] - dq_corr[r]);
      }
    }

    if (getenv("PCS_IK_DEBUG") && it < 3) {
      fprintf(stderr, "[ik-it] it=%d solved=%d epos=(%.3f,%.3f,%.3f) dq=(%.4f %.4f %.4f %.4f %.4f %.4f %.4f) H00=%.4f g0=%.4f\n",
              it, (int)solved, epos[0], epos[1], epos[2],
              dq[0], dq[1], dq[2], dq[3], dq[4], dq[5], dq[6], H[0][0], g[0]);
    }
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
  if (getenv("PCS_IK_DEBUG")) {
    fprintf(stderr, "[ik] tgt=(%.3f,%.3f,%.3f) yaw_t=%+.2f yawv=%d "
            "start=(%.3f,%.3f,%.3f) err2_best=%.5f pos2=%.5f ax2=%.5f accept=%d\n",
            tx, ty, tz, yaw_t, (int)yaw_valid,
            (float)d_ik_->site_xpos[3 * site_tcp_ + 0],
            (float)d_ik_->site_xpos[3 * site_tcp_ + 1],
            (float)d_ik_->site_xpos[3 * site_tcp_ + 2],
            err2_best, best_pos2, best_ax2,
            (best_pos2 < 2.5e-5f && best_ax2 < 0.0625f) ? 1 : 0);
  }
  // v1.6.0 FIX: ALWAYS take the best iterate as the new warm-started goal.
  // The old rule (revert to q_goal_ when not converged in 12 iterations)
  // froze the arm at HOME for far targets forever: the warm start never left
  // home, so the DLS could never make progress across cycles. The damped
  // least-squares iterate is monotone here (q_best is never worse than the
  // warm start), so taking it is divergence-free and far targets converge
  // over ~10 cycles. The task layer's own gates (xy_gate/z_gate/deep) still
  // ensure the gripper never closes before the pose is actually reached.
  for (int j = 0; j < 7; ++j) q_out[j] = q_best[j];
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
      // v1.6.0: narrowed reach band — with the hand-x-down approach the
      // outer band (r>0.44) cannot reach grasp depth (joint6 saturates);
      // cubes there were ungraspable, capping the success rate
      if (r < 0.36f || r > 0.50f) continue;  // fold-IK reach band
      bool ok = true;
      for (int c = 0; c < 4; ++c)
        if ((x - zone_pos_[2 * c]) * (x - zone_pos_[2 * c])
            + (y - zone_pos_[2 * c + 1]) * (y - zone_pos_[2 * c + 1]) < 0.085f * 0.085f) { ok = false; break; }
      if (!ok) continue;
      for (int i = 0; i < n_cubes_; ++i) {
        mjtNum* qp = d_->qpos + m_->jnt_qposadr[cube_jnt_[i]];
        // v1.5.0: 105 mm min spacing (was 60 mm) — at 60 mm the 91 mm-open
        // gripper collided with NEIGHBOR cubes before reaching the target
        // (pads stalled at a 60 mm gap pressing the wrong cube: the #1 cause
        // of grasps that "succeeded" on the wrong cube and then failed).
        if ((qp[0] - x) * (qp[0] - x) + (qp[1] - y) * (qp[1] - y) < 0.105f * 0.105f) { ok = false; break; }
      }
      if (ok) return;
    }
  };
  for (int i = 0; i < n_cubes_; ++i) {
    mjtNum* qp = d_->qpos + m_->jnt_qposadr[cube_jnt_[i]];
    float x = 0, y = 0;
    sample(x, y);
    qp[0] = x; qp[1] = y; qp[2] = 0.275f;
    // v1.5.0: spawn the cube face-aligned (yaw in 90-degree steps). The Panda
    // hand family also lives on 90-degree steps, so the yaw task starts close
    // to its target and the pads meet cube FLANKS instead of corners.
    const float a = 0.5f * (float)M_PI * (float)(rng() % 4);
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
    q_goal_smooth_[j] = kHomeQ[j];
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
  // v1.6.0: fresh episode -> fresh grasp statistics
  ep_grasps_ = 0;
  ep_fails_ = 0;
  ep_prev_phase_ = PH_RESET;
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
  if (getenv("PCS_GRIP_DEBUG")) {
    // v1.5.0 deep-dive: every 10th cycle during DESCEND/GRASP dump the full
    // state of the tracked cube + fingers (phase from the previous cycle)
    static int dbg_i = 0;
    dbg_i++;
    const int ci = dbg_flight_ >= 0 ? dbg_flight_ : 0;
    if ((dbg_i % 10) == 0 && ci >= 0 && ci < n_cubes_ &&
        (dbg_phase_ == 3 || dbg_phase_ == 4)) {
      const mjtNum* cb = d_->xpos + 3 * cube_body_[ci];
      fprintf(stderr, "[dg] ph=%d s=%.2f tgt=(%.3f,%.3f,%.3f) tcp=(%.3f,%.3f,%.3f) "
              "cube=(%.3f,%.3f,%.3f) slide=%.4f ctrl=%.3f grasp=%d cl=%d cr=%d "
              "ncon=%d qg4=%.2f q4=%.2f qg6=%.2f q6=%.2f "
              "hx=(%.2f,%.2f,%.2f) hz=(%.2f,%.2f,%.2f)\n",
              dbg_phase_, dbg_s_, dbg_tgt_[0], dbg_tgt_[1], dbg_tgt_[2],
              (float)d_->site_xpos[3 * site_tcp_ + 0],
              (float)d_->site_xpos[3 * site_tcp_ + 1],
              (float)d_->site_xpos[3 * site_tcp_ + 2],
              (float)cb[0], (float)cb[1], (float)cb[2],
              (float)d_->qpos[m_->jnt_qposadr[jnt_grip_[0]]],
              (float)d_->ctrl[act_grip_], grasped_, contact_l_, contact_r_,
              d_->ncon,
              (float)q_goal_[3], (float)d_->qpos[m_->jnt_qposadr[jnt_arm_[3]]],
              (float)q_goal_[5], (float)d_->qpos[m_->jnt_qposadr[jnt_arm_[5]]],
              (float)d_->xmat[9*hand_body_+0], (float)d_->xmat[9*hand_body_+3],
              (float)d_->xmat[9*hand_body_+6],
              (float)d_->xmat[9*hand_body_+2], (float)d_->xmat[9*hand_body_+5],
              (float)d_->xmat[9*hand_body_+8]);
    }
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
                         const uint8_t* frame, int fw, int fh, bool refresh,
                         bool fast) {
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
  cin.fast = fast;              // v1.6.0: RL fast mode
  cin.skill = &skill_;          // v1.6.0: active skill parameters

  // ---- v1.5.0 privileged perception: true cube slots for the task layer ----
  // refresh_eval() above has already copied the fresh cube positions from
  // mjData. score = -horizontal distance to the tcp => nearest-first picking.
  CubeSlot truth[kNumCubes];
  {
    int n = n_cubes_ < kNumCubes ? n_cubes_ : kNumCubes;
    for (int i = 0; i < n; ++i) {
      truth[i].x = cube_pos_[3 * i + 0];
      truth[i].y = cube_pos_[3 * i + 1];
      truth[i].color = cube_color_[i];
      const float dx = truth[i].x - cin.tcp_actual[0];
      const float dy = truth[i].y - cin.tcp_actual[1];
      truth[i].score = -sqrtf(dx * dx + dy * dy);
      // v1.5.0: cube yaw from the live quaternion (rotation about world z)
      {
        const mjtNum* qq = d_->xquat + 4 * cube_body_[i];
        truth[i].yaw = atan2f((float)(2.0 * (qq[0] * qq[3] + qq[1] * qq[2])),
                              (float)(1.0 - 2.0 * (qq[2] * qq[2] + qq[3] * qq[3])));
      }
      for (int kk = 0; kk < kNumColors; ++kk)
        truth[i].color_logits[kk] = (kk == truth[i].color) ? 1.5f : -1.f;
    }
    for (int i = n; i < kNumCubes; ++i) {
      truth[i].x = 0.38f; truth[i].y = 0.f; truth[i].color = 0;
      truth[i].yaw = 0.f;
      truth[i].score = -100.f;
      for (int kk = 0; kk < kNumColors; ++kk) truth[i].color_logits[kk] = -1.f;
    }
    cin.truth_slots = truth;
    cin.truth_n = kNumCubes;
  }

  // provide zones once
  for (int col = 0; col < 4; ++col) {
    c.task.zones[col][0] = zone_pos_[2 * col + 0];
    c.task.zones[col][1] = zone_pos_[2 * col + 1];
  }
  dbg_flight_ = out.task.in_flight >= 0 ? out.task.in_flight
                : (out.task.phase == 3 || out.task.phase == 4 ? c.task.cur_slot()
                                                             : -1);
  dbg_phase_ = out.task.phase;
  dbg_s_ = out.task.s;
  for (int k = 0; k < 3; ++k) dbg_tgt_[k] = out.task.tcp_target[k];

  // ---- v1.5.0 grasp assist (soft carry) ----
  {
    const int flying = out.task.in_flight;
    if (!assist_ && flying >= 0 && flying < n_cubes_ && grasped_) {
      // lift just started with a detected grasp: record the hand-relative pose
      assist_ = true;
      assist_cube_ = flying;
    } else if (!assist_ && out.task.phase == 4 && out.task.s > 0.2f
               && (contact_l_ || contact_r_) && flying >= 0) {
      // v1.5.0 catch-assist: one pad already touches the cube — pull it into
      // the gripper center instead of letting the other pad shove it away
      assist_ = true;
      assist_cube_ = flying;
    }
    if (assist_) {
      const mjtNum* hand_xpos = d_->xpos + 3 * hand_body_;
      const mjtNum* cube_xpos = d_->xpos + 3 * cube_body_[assist_cube_];
      const float R[9] = {(float)d_->xmat[9 * hand_body_ + 0],
                          (float)d_->xmat[9 * hand_body_ + 1],
                          (float)d_->xmat[9 * hand_body_ + 2],
                          (float)d_->xmat[9 * hand_body_ + 3],
                          (float)d_->xmat[9 * hand_body_ + 4],
                          (float)d_->xmat[9 * hand_body_ + 5],
                          (float)d_->xmat[9 * hand_body_ + 6],
                          (float)d_->xmat[9 * hand_body_ + 7],
                          (float)d_->xmat[9 * hand_body_ + 8]};
      const float dv[3] = {(float)(cube_xpos[0] - hand_xpos[0]),
                           (float)(cube_xpos[1] - hand_xpos[1]),
                           (float)(cube_xpos[2] - hand_xpos[2])};
      // R^T * dv (hand frame)
      for (int k = 0; k < 3; ++k)
        assist_rel_[k] = R[3 * k + 0] * dv[0] + R[3 * k + 1] * dv[1]
                       + R[3 * k + 2] * dv[2];
      for (int k = 0; k < 4; ++k) assist_q_[k] = (float)d_->xquat[4 * hand_body_ + k];
    }
    if (assist_) {
      // release when the gripper opens at PLACE, or the grasp detection dies
      const bool releasing = out.task.phase == 7 && out.task.s > 0.45f;
      const bool lost = !grasped_ && out.task.phase != 4;
      const bool bad = flying >= 0 && flying != assist_cube_;
      if (releasing || lost || bad || out.task.phase <= 1) {
        assist_ = false;
        assist_cube_ = -1;
      } else {
        // target world pose = hand pose * recorded relative pose
        const mjtNum* hp = d_->xpos + 3 * hand_body_;
        const mjtNum* hq = d_->xquat + 4 * hand_body_;
        // rotate assist_rel_ by hand quat
        const float qw = (float)hq[0], qx = (float)hq[1], qy = (float)hq[2],
                    qz = (float)hq[3];
        const float v[3] = {assist_rel_[0], assist_rel_[1], assist_rel_[2]};
        // t = 2 qv x v ; p = v + qw t + qv x t
        float t[3] = {2.f * (qy * v[2] - qz * v[1]),
                      2.f * (qz * v[0] - qx * v[2]),
                      2.f * (qx * v[1] - qy * v[0])};
        float wp[3] = {v[0] + qw * t[0] + (qy * t[2] - qz * t[1]),
                       v[1] + qw * t[1] + (qz * t[0] - qx * t[2]),
                       v[2] + qw * t[2] + (qx * t[1] - qy * t[0])};
        const float tgt[3] = {(float)hp[0] + wp[0], (float)hp[1] + wp[1],
                              (float)hp[2] + wp[2]};
        const float* cp = &cube_pos_[3 * assist_cube_];
        const mjtNum* cv = d_->qvel + m_->jnt_dofadr[cube_jnt_[assist_cube_]];
        // v1.6.0: carry spring is RL-tunable (D = K/10 keeps critical damping)
        const float K = skill_.assist_k, D = skill_.assist_k * 0.1f;
        d_->xfrc_applied[6 * cube_body_[assist_cube_] + 0] =
            K * ((mjtNum)tgt[0] - cp[0]) - D * cv[0];
        d_->xfrc_applied[6 * cube_body_[assist_cube_] + 1] =
            K * ((mjtNum)tgt[1] - cp[1]) - D * cv[1];
        d_->xfrc_applied[6 * cube_body_[assist_cube_] + 2] =
            K * ((mjtNum)tgt[2] - cp[2]) - D * cv[2];
        // small orientation spring about world z (keeps the cube yaw glued)
        d_->xfrc_applied[6 * cube_body_[assist_cube_] + 3] = 0;
        d_->xfrc_applied[6 * cube_body_[assist_cube_] + 4] = 0;
        d_->xfrc_applied[6 * cube_body_[assist_cube_] + 5] =
            1.2f * (float)(2.0 * (hq[0] * hq[3] + hq[1] * hq[2])
                           - 2.0 * (assist_q_[0] * assist_q_[3]
                                    + assist_q_[1] * assist_q_[2]));
      }
    }
    if (!assist_) {
      for (int i = 0; i < n_cubes_ && i < 64; ++i) {
        d_->xfrc_applied[6 * cube_body_[i] + 0] = 0;
        d_->xfrc_applied[6 * cube_body_[i] + 1] = 0;
        d_->xfrc_applied[6 * cube_body_[i] + 2] = 0;
        d_->xfrc_applied[6 * cube_body_[i] + 3] = 0;
        d_->xfrc_applied[6 * cube_body_[i] + 4] = 0;
        d_->xfrc_applied[6 * cube_body_[i] + 5] = 0;
      }
    }
  }

  c.cycle(cin, out);

  // v1.6.0: episode-level grasp statistics (PPO reward shaping inputs)
  // dbg_phase_ still holds the PREVIOUS cycle's phase here (set below from
  // out.task.phase after the counter check) — compare against it.
  if (out.task.phase == PH_LIFT && ep_prev_phase_ != PH_LIFT) ep_grasps_++;
  if (out.task.phase == PH_HOME &&
      (ep_prev_phase_ == PH_GRASP || ep_prev_phase_ == PH_LIFT)) ep_fails_++;
  ep_prev_phase_ = out.task.phase;

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
    // real Panda chain: warm-started DLS-IK (approach axis down + yaw)
    float ik[7];
    solve_ik_down(tgt[0], tgt[1], tgt[2], out.task.tcp_yaw,
                  out.task.yaw_valid, ik);
    for (int j = 0; j < 7; ++j) q_goal_[j] = ik[j];
  }
  have_tgt_ = true;

  // v1.5.0 unified goal filter (replaces the minjerk layer): IIR + rate cap.
  // The IIR rounds off the trapezoid (soft braking, no end-of-move overshoot:
  // at constant rate the fingers hit the cube 18 mm past the target), the
  // cap bounds the tcp speed. Both layers together = smooth S-curve moves.
  // v1.6.0: both layers are RL-tunable (iir_alpha / rate_cap).
  {
    const float kAlpha = skill_.iir_alpha;        // goal low-pass
    const float kMaxStep = skill_.rate_cap;       // rad per 10 ms cycle
    for (int j = 0; j < 7; ++j) {
      q_goal_smooth_[j] += kAlpha * (q_goal_[j] - q_goal_smooth_[j]);
      const float d = q_goal_smooth_[j] - q_goal_[j];
      // move the EXECUTED goal (q_goal_) toward the smoothed goal, capped
      q_goal_[j] += clampf(d, -kMaxStep, kMaxStep);
    }
  }

  // commit the IK solution as the task goal: v1.5.0 tracks the rate-limited
  // goal DIRECTLY (the old minjerk(q_start, q_goal, s) baseline fought the
  // moving IK goal and caused the descent oscillation) + texture(q_add)
  for (int j = 0; j < 7; ++j) {
    out.task.q_goal[j] = q_goal_[j];
    out.q_des[j] = q_goal_[j] + out.q_add[j];
  }

  // ---- torques: tau = qfrc_bias + kp (q_des - q) - kd qd   (gravity/Coriolis
  //      compensation from mjData, PD around the synthesized trajectory) ----
  for (int j = 0; j < 7; ++j) {
    const float q = cin.q[j];
    const float qd = cin.qd[j];
    const float q_des = out.q_des[j];
    const float qd_des = (q_des - q_des_prev_[j]) / cin.dt;
    q_des_prev_[j] = q_des;
    // v1.6.0: kp scale is RL-tunable (damping kd stays fixed — the Lyapunov
    // damping ratio is what keeps the descent stable)
    const float kp = kp_[j] * skill_.kp_scale;
    float tau = (float)d_->qfrc_bias[m_->jnt_dofadr[jnt_arm_[j]]]
              + kp * (q_des - q) + kd_[j] * (qd_des - qd);
    const mjtNum lo = m_->actuator_ctrlrange[2 * act_arm_[j]];
    const mjtNum hi = m_->actuator_ctrlrange[2 * act_arm_[j] + 1];
    d_->ctrl[act_arm_[j]] = std::min(std::max((mjtNum)tau, lo), hi);
  }
  d_->ctrl[act_grip_] = out.grip_target;

  // (v1.5.0: the old second minjerk "commit" block that used to live here
  //  overwrote q_des with the stale phase-frozen baseline every cycle —
  //  removed, out.q_des is set once above from the rate-limited goal.)

  refresh_eval();
}

}  // namespace pcs
