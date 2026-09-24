// pcs/task_layer.cpp — sorting/stacking state machine
#include "pcs/task_layer.h"
#include <cstdlib>
#include <cstdio>
#include <algorithm>

namespace pcs {

// tcp sits at the fingertip plane of the REAL Panda hand. v1.6.0: the
// approach axis task now points the hand's LOCAL +X down (the fork axis),
// so the pads span [tcp-0.010, tcp+0.074] vertically and carried cubes ride
// ~5 mm above the tcp. Heights are RL-tunable via SkillParams.
static constexpr float kCubeMid = 0.275f;    // cube mid-height on the table
static constexpr float kTransZ  = 0.42f;
static constexpr float kStackDz  = 0.052f;   // cube height + clearance
static constexpr float kSlotScoreMin = -10.f;  // relative selection only

// grasp-plane tcp height + stage-1 stop height from the skill parameters
// v1.6.0: with the corrected hand-x-down approach the pads span
// [tcp-0.010, tcp+0.074]; default grasp tcp = 0.270 (mid 0.275 - 0.005)
inline float grasp_z_of(const SkillParams* P) {
  const float off = P ? P->grasp_z_off : 0.005f;
  return kCubeMid - off;
}
inline float align_z_of(const SkillParams* P) {
  const float a = P ? P->align_z : 0.030f;
  return grasp_z_of(P) + a;
}

void TaskLayer::reset() {
  next_color_ = 0; in_flight_ = false; flight_slot_ = -1; phase_t_ = 0.f;
  wiggle_dx_ = wiggle_dy_ = 0.f; wiggle_tries_ = 0; done_ = false; cur_slot_ = -1;
  for (int i = 0; i < 4; ++i) zone_stack[i] = 0;
  for (int i = 0; i < 8; ++i) { slot_tries_[i] = 0; slot_dead_[i] = false; }
  phase_ = PH_RESET;
}

// phase durations: v1.6.0 DESCEND/GRASP/LIFT are RL-tunable
static float phase_duration(int p, const SkillParams* P) {
  switch (p) {
    case PH_RESET: return 0.3f;
    case PH_HOME: return 1.1f;
    case PH_HOVER: return 1.8f;   // gated on REACHED (transit to the cube)
    case PH_DESCEND: return P ? P->descend_t : 3.0f;
    case PH_GRASP: return P ? P->grasp_t : 0.9f;
    case PH_LIFT: return P ? P->lift_t : 1.3f;
    case PH_TRANSPORT: return 1.3f;
    case PH_PLACE: return 1.2f;
  }
  return 1.f;
}

void TaskLayer::start_phase(int p, const TaskInput& in, TaskOutput& out) {
  if (getenv("PCS_TRACE")) {
    fprintf(stderr, "[task] %s -> %s slot=%d grasped=%d grab=(%.3f,%.3f) tcp=(%.3f,%.3f,%.3f) q=(%.2f %.2f %.2f %.2f %.2f %.2f %.2f)\n",
            kPhaseName[phase_], kPhaseName[p], cur_slot_, (int)in.grasped,
            grab_xy_[0], grab_xy_[1], in.tcp_actual[0], in.tcp_actual[1], in.tcp_actual[2],
            in.q[0], in.q[1], in.q[2], in.q[3], in.q[4], in.q[5], in.q[6]);
  }
  prev_phase_ = phase_;
  phase_ = p;
  phase_t_ = 0.f;
  phase_dur_ = phase_duration(p, in.skill);
  for (int j = 0; j < kDof; ++j) { out.q_start[j] = in.q[j]; q_start_[j] = in.q[j]; }
  out.ik_needed = true;
}

void TaskLayer::pick_next_slot(const TaskInput& in) {
  // highest-scoring live slot of the color we are currently collecting
  int best = -1; float best_sc = kSlotScoreMin - 1.f;
  for (int round = 0; round < 2 && best < 0; ++round) {
    for (int i = 0; i < in.n_cubes; ++i) {
      if (slot_dead_[i]) continue;
      if (round == 0 && in.cubes[i].color != next_color_) continue;
      if (in.cubes[i].score > best_sc) { best_sc = in.cubes[i].score; best = i; }
    }
  }
  cur_slot_ = best;   // may be -1 if nothing left
}

namespace {
inline float dist3(const float* a, const float* b) {
  const float dx = a[0]-b[0], dy = a[1]-b[1], dz = a[2]-b[2];
  return sqrtf(dx*dx + dy*dy + dz*dz);
}
inline float dist2xy(const float* a, const float* b) {
  const float dx = a[0]-b[0], dy = a[1]-b[1];
  return sqrtf(dx*dx + dy*dy);
}
}  // namespace

void TaskLayer::update(const TaskInput& in, TaskOutput& out) {
  // v1.6.0: the active skill parameter set (RL mean or sampled theta)
  const SkillParams* P = in.skill;
  out.phase = phase_;
  out.ik_needed = false;
  out.episode_done = done_;
  out.in_flight = in_flight_ ? flight_slot_ : -1;
  out.active_color = next_color_;
  out.grip_target = kGripOpen;
  out.yaw_valid = false;
  out.tcp_yaw = 0.f;
  for (int j = 0; j < kDof; ++j) out.q_start[j] = q_start_[j];
  out.tcp_target[0] = 0.42f; out.tcp_target[1] = 0.f; out.tcp_target[2] = 0.43f;
  out.joint_hold = false;
  int remaining = 0;
  for (int i = 0; i < in.n_cubes; ++i) if (!slot_dead_[i]) remaining++;
  out.remaining = remaining;

  if (done_) { out.phase = PH_HOME; out.s = 1.f; out.grip_target = kGripOpen;
    for (int j = 0; j < kDof; ++j) out.q_goal[j] = kHomeQ[j];
    out.joint_hold = true;   // home hold (above the scene)
    out.ik_needed = false; return; }

  phase_t_ += in.dt;

  switch (phase_) {
    case PH_RESET:
      out.s = phase_t_ / phase_dur_;
      start_phase(PH_HOME, in, out);
      break;

    case PH_HOME: {
      out.s = phase_t_ / phase_dur_;
      out.ik_needed = false;
      out.joint_hold = true;   // home hold: no IK, joints fixed
      for (int j = 0; j < kDof; ++j) out.q_goal[j] = kHomeQ[j];
      out.tcp_target[0] = 0.417f; out.tcp_target[1] = 0.f; out.tcp_target[2] = 0.517f;
      out.grip_target = kGripOpen;
      if (out.s >= 1.f) {
        pick_next_slot(in);
        if (cur_slot_ < 0) {
          // try next color batch
          bool found = false;
          for (int c = 0; c < kNumColors && !found; ++c) {
            next_color_ = (next_color_ + 1) % kNumColors;
            pick_next_slot(in);
            if (cur_slot_ >= 0) found = true;
          }
          if (!found) { done_ = true; out.episode_done = true; break; }
        }
        start_phase(PH_HOVER, in, out);
      }
      break;
    }

    case PH_HOVER: {
      have_grab_ = false;
      // CARRY transit: warm-started IK drives the tcp above the target cube
      // at safe height (real Panda chain — the old planar q_carry is gone)
      out.s = phase_t_ / phase_dur_;
      out.ik_needed = false;
      out.joint_hold = false;
      if (cur_slot_ >= 0 && cur_slot_ < in.n_cubes) {
        const CubeSlot& c = in.cubes[cur_slot_];
        out.tcp_target[0] = c.x;
        out.tcp_target[1] = c.y;
        // v1.5.0: rotate the hand to the cube's face family WHILE hovering —
        // flank grips (50 mm) are stable; corner grips slip during LIFT
        out.tcp_yaw = c.yaw;
        out.yaw_valid = true;
      }
      out.tcp_target[2] = kTransZ;
      out.grip_target = kGripOpen;
      // v1.6.0: leave HOVER only when the hover point is actually REACHED
      // (near the cube xy at transit height). The old pure-time exit fired
      // while the arm was still at home; DESCEND then swept the elbow
      // diagonally through the cube field (link3 plowed into cubes/table and
      // the shoulder locked). Timeout fallback stays for unreachable slots.
      const bool hover_ok = cur_slot_ >= 0 && cur_slot_ < in.n_cubes
          && dist2xy(in.tcp_actual, out.tcp_target) < 0.03f
          && in.tcp_actual[2] > kTransZ - 0.06f;
      if (hover_ok || out.s >= 2.f) {
        // v1.5.0: FREEZE the grasp point NOW. Tracking the live cube position
        // during DESCEND/GRASP made the gripper chase (and push) the sliding
        // cube across the table — the classic "schieben statt greifen".
        if (cur_slot_ >= 0 && cur_slot_ < in.n_cubes) {
          grab_xy_[0] = in.cubes[cur_slot_].x;
          grab_xy_[1] = in.cubes[cur_slot_].y;
          grab_yaw_ = in.cubes[cur_slot_].yaw;
          have_grab_ = true;
        }
        start_phase(PH_DESCEND, in, out);
      }
      break;
    }

    case PH_DESCEND: {
      // v1.5.0 two-stage approach: stage 1 stops ABOVE the cube (pads ~28 mm
      // over cube mid-height) while the pads pre-close to just over both the
      // flank (50 mm) and diagonal (71 mm) envelopes — the funnel aligns a
      // rotated cube without pushing it.
      // v1.6.0: stop heights + gates are RL-tunable
      const float align_z = align_z_of(P);
      const float xy_gate = P ? P->xy_gate : 0.008f;
      const float z_gate = P ? P->z_gate : 0.012f;
      out.s = phase_t_ / phase_dur_;
      // v1.6.0 COLLISION-FREE PATH: high transit FIRST, then vertical
      // descent. The old direct approach from home swept the elbow through
      // the cube field (link3-vs-cube/table contact locked the shoulder).
      // Stage A: above the cube at transit height. Stage B: straight down.
      // NOTE: the switch is purely horizontal — a z-based switch would
      // deadlock (the stage-A target IS the transit height).
      const float dxy_grab = dist2xy(in.tcp_actual, grab_xy_);
      const bool high = dxy_grab > 0.06f;
      out.tcp_target[0] = grab_xy_[0] + wiggle_dx_;
      out.tcp_target[1] = grab_xy_[1] + wiggle_dy_;
      out.tcp_target[2] = high ? kTransZ : align_z;
      out.tcp_yaw = grab_yaw_;
      out.yaw_valid = true;
      float g;
      const float grip_pre = P ? P->grip_pre : kGripPre;
      // v1.6.0 ROOT-CAUSE FIX (0% Erfolg): the old progress-based pre-close
      // (ramp at s in [0.30, 0.75]) started closing while the pads were
      // still ABOVE the cube — the gap (63 mm) is narrower than the cube
      // diagonal (70.7 mm), so the pads WEDGED on the top corners and the
      // arm stalled ~47 mm above the target ("pads ride the cube top").
      // Now the pads stay OPEN until the tcp is beside the cube (below the
      // cube top, tcp_z < cube_mid + 12 mm) and only then funnel close.
      // The funnel still aligns a rotated cube — but from the flank, not
      // from the top.
      const float pre_z = kCubeMid + 0.012f;
      if (in.tcp_actual[2] > pre_z) {
        g = kGripOpen;
      } else {
        const float t = clampf((pre_z - in.tcp_actual[2]) / 0.02f, 0.f, 1.f);
        g = kGripOpen + (grip_pre - kGripOpen) * t;
      }
      out.grip_target = g;
      // v1.5.0: stage-1 exit requires the XY error to be TIGHT (8 mm) — the
      // old combined 3D 25 mm gate let the gripper descend with a lateral
      // offset that pressed one finger into the cube's flank and shoved it.
      const float pxy[3] = {out.tcp_target[0], out.tcp_target[1],
                            in.tcp_actual[2]};
      const float txy[3] = {out.tcp_target[0], out.tcp_target[1],
                            out.tcp_target[2]};
      const bool xy_ok = !high && dist2xy(in.tcp_actual, pxy) < xy_gate;
      const bool z_ok = !high && dist3(in.tcp_actual, txy) < z_gate;
      // one-sided touch while descending: klemme sofort, statt zu schieben
      if (!high && in.contact_l != in.contact_r && out.s > 0.3f) {
        start_phase(PH_GRASP, in, out);
        break;
      }
      if ((out.s >= 1.f && xy_ok && z_ok) || phase_t_ >= 10.f)
        start_phase(PH_GRASP, in, out);
      break;
    }

    case PH_GRASP: {
      // stage 2: descend the last ~23 mm WHILE closing pre -> closed; the
      // cube is gripped around its flanks instead of pushed sideways
      // v1.6.0: grasp depth + wiggle amplitude are RL-tunable
      const float grasp_z = grasp_z_of(P);
      const float wig = P ? P->wiggle_amp : 0.012f;
      out.s = phase_t_ / phase_dur_;
      out.tcp_target[0] = grab_xy_[0] + wiggle_dx_;
      out.tcp_target[1] = grab_xy_[1] + wiggle_dy_;
      // v1.6.0: HOLD the height when exactly ONE pad already touches the
      // cube. The old rule kept descending while a plate edge was hooked on
      // the cube's top corner — the fork then plowed the cube across (and
      // off) the table ("Wuerfel wird uebern Tisch geschossen"). Holding and
      // closing lets the funnel realign the cube and complete the grip.
      if (in.contact_l != in.contact_r)
        out.tcp_target[2] = std::max(in.tcp_actual[2], grasp_z);
      else
        out.tcp_target[2] = grasp_z;
      out.tcp_yaw = grab_yaw_;
      out.yaw_valid = true;
      float g;
      if (out.s < 0.10f) g = kGripPre;
      else if (out.s < 0.60f)
        g = kGripPre + (kGripClosed - kGripPre) * (out.s - 0.10f) / 0.5f;
      else g = kGripClosed;
      out.grip_target = g;
      // v1.5.0: leave the grasp only when the pads sit at cube MID-height
      // (deep) AND the close ramp has finished (clamped). The old rule fired
      // on the first touch — pads grazed the cube TOP and the lift tore the
      // immature grip open (cube dropped at lift start).
      const bool deep = in.tcp_actual[2] < grasp_z + 0.012f;
      // v1.6.0: relative clamp time (the old hard 0.55 s broke whenever RL
      // sampled a shorter grasp_t — the phase could NEVER exit and every
      // training episode failed identically, destroying the gradient)
      const bool clamped = g <= kGripClosed + 0.004f
                           && phase_t_ >= 0.61f * phase_dur_;
      if (in.grasped && deep && clamped) {
        wiggle_tries_ = 0; wiggle_dx_ = wiggle_dy_ = 0.f;
        in_flight_ = true; flight_slot_ = cur_slot_;
        start_phase(PH_LIFT, in, out);
      } else if (phase_t_ >= 3.f) {
        if (wiggle_tries_ < 2) {
          // grasp-wiggle: re-read the (possibly pushed) cube, shift approach
          wiggle_tries_++;
          const float a = 3.1f * (float)wiggle_tries_;
          wiggle_dx_ = wig * cosf(a);
          wiggle_dy_ = wig * sinf(a);
          slot_tries_[cur_slot_ >= 0 ? cur_slot_ : 0]++;
          if (cur_slot_ >= 0 && cur_slot_ < in.n_cubes) {
            grab_xy_[0] = in.cubes[cur_slot_].x;
            grab_xy_[1] = in.cubes[cur_slot_].y;
            grab_yaw_ = in.cubes[cur_slot_].yaw;
          }
          start_phase(PH_DESCEND, in, out);
        } else {
          if (cur_slot_ >= 0 && cur_slot_ < 8) slot_dead_[cur_slot_] = true;
          wiggle_tries_ = 0; wiggle_dx_ = wiggle_dy_ = 0.f;
          start_phase(PH_HOME, in, out);
        }
      }
      break;
    }

    case PH_LIFT: {
      out.s = phase_t_ / phase_dur_;
      out.tcp_target[0] = grab_xy_[0];
      out.tcp_target[1] = grab_xy_[1];
      out.tcp_target[2] = kTransZ;
      out.tcp_yaw = grab_yaw_;
      out.yaw_valid = true;
      out.grip_target = kGripClosed;
      // v1.5.0: if the cube slipped away early in the lift, do NOT fly the
      // empty gripper to the zone — go home and retry the slot instead.
      if (!in.grasped && phase_t_ > 0.8f && phase_t_ < 4.6f) {
        in_flight_ = false; flight_slot_ = -1;
        start_phase(PH_HOME, in, out);
        break;
      }
      if ((out.s >= 1.f && in.tcp_actual[2] > kTransZ - 0.03f)
          || phase_t_ >= 6.f) {
        next_color_ = in.n_cubes > 0 && flight_slot_ >= 0 && flight_slot_ < in.n_cubes
                      ? in.cubes[flight_slot_].color : next_color_;
        start_phase(PH_TRANSPORT, in, out);
      }
      break;
    }

    case PH_TRANSPORT: {
      out.s = phase_t_ / phase_dur_;
      const int col = clampf((float)next_color_, 0.f, (float)(kNumColors - 1));
      out.tcp_target[0] = zones[col][0];
      out.tcp_target[1] = zones[col][1];
      out.tcp_target[2] = kTransZ;
      out.tcp_yaw = grab_yaw_;   // hold the grasp yaw while carrying
      out.yaw_valid = true;
      out.grip_target = kGripClosed;
      const float txy[3] = {out.tcp_target[0], out.tcp_target[1], in.tcp_actual[2]};
      if ((out.s >= 1.f && dist3(in.tcp_actual, txy) < 0.025f) || phase_t_ >= 6.f)
        start_phase(PH_PLACE, in, out);
      break;
    }

    case PH_PLACE: {
      out.s = phase_t_ / phase_dur_;
      const int col = clampf((float)next_color_, 0.f, (float)(kNumColors - 1));
      const float place_z0 = grasp_z_of(P);   // cube bottom touches table/stack
      out.tcp_target[0] = zones[col][0];
      out.tcp_target[1] = zones[col][1];
      out.tcp_target[2] = place_z0 + zone_stack[col] * kStackDz;
      out.tcp_yaw = grab_yaw_;
      out.yaw_valid = true;
      out.grip_target = out.s > (P ? P->release_s : 0.60f) ? kGripOpen
                                                           : kGripClosed;
      if ((out.s >= 1.f && dist3(in.tcp_actual, out.tcp_target) < 0.02f) || phase_t_ >= 5.f) {
        if (in_flight_) {
          zone_stack[col]++;
          if (flight_slot_ >= 0 && flight_slot_ < 8) slot_dead_[flight_slot_] = true;
          in_flight_ = false; flight_slot_ = -1;
        }
        start_phase(PH_HOME, in, out);
      }
      break;
    }

    default:
      start_phase(PH_HOME, in, out);
      break;
  }

  out.phase = phase_;
  out.s = clampf(phase_t_ / phase_dur_, 0.f, 1.f);
  out.phase_t = phase_t_;
}

}  // namespace pcs
