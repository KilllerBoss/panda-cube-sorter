// pcs/task_layer.cpp — sorting/stacking state machine
#include "pcs/task_layer.h"
#include <cstdlib>
#include <cstdio>
#include <algorithm>

namespace pcs {

// tcp sits at the fingertip plane of the REAL Panda hand (0.126 below the
// hand origin; pads at 0.103): grasp = pads beside cube mid-height =>
// tcp = 0.275 - 0.023 = 0.252; carried cubes ride ~0.023 above tcp.
static constexpr float kGraspZ   = 0.252f;   // pads beside cube mid-height
static constexpr float kPlaceZ0  = 0.252f;   // cube bottom touches table/stack
static constexpr float kTransZ   = 0.42f;
static constexpr float kStackDz  = 0.052f;   // cube height + clearance
static constexpr float kSlotScoreMin = -10.f;  // relative selection only

void TaskLayer::reset() {
  next_color_ = 0; in_flight_ = false; flight_slot_ = -1; phase_t_ = 0.f;
  wiggle_dx_ = wiggle_dy_ = 0.f; wiggle_tries_ = 0; done_ = false; cur_slot_ = -1;
  for (int i = 0; i < 4; ++i) zone_stack[i] = 0;
  for (int i = 0; i < 8; ++i) { slot_tries_[i] = 0; slot_dead_[i] = false; }
  phase_ = PH_RESET;
}

static float phase_duration(int p) {
  switch (p) {
    case PH_RESET: return 0.3f;
    case PH_HOME: return 1.1f;
    case PH_HOVER: return 0.7f;
    case PH_DESCEND: return 1.0f;
    case PH_GRASP: return 0.55f;
    case PH_LIFT: return 0.9f;
    case PH_TRANSPORT: return 1.1f;
    case PH_PLACE: return 1.0f;
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
  phase_dur_ = phase_duration(p);
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
}  // namespace

void TaskLayer::update(const TaskInput& in, TaskOutput& out) {
  out.phase = phase_;
  out.ik_needed = false;
  out.episode_done = done_;
  out.in_flight = in_flight_ ? flight_slot_ : -1;
  out.active_color = next_color_;
  out.grip_target = kGripOpen;
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
      }
      out.tcp_target[2] = kTransZ;
      out.grip_target = kGripOpen;
      if (out.s >= 1.f) start_phase(PH_DESCEND, in, out);
      break;
    }

    case PH_DESCEND: {
      // live tracking of the decoded cube + capture-gap funnel: the pads
      // squeeze rotated corners into alignment (~1.5 N < table friction*mu)
      out.s = phase_t_ / phase_dur_;
      if (cur_slot_ >= 0 && cur_slot_ < in.n_cubes) {
        const CubeSlot& c = in.cubes[cur_slot_];
        out.tcp_target[0] = c.x + wiggle_dx_;
        out.tcp_target[1] = c.y + wiggle_dy_;
      }
      out.tcp_target[2] = kGraspZ;
      // funnel: fully open (91 mm, over-diagonal) all the way down; the
      // pre-close to diagonal-contact (stationary) happens at the bottom and
      // the final close to kGripClosed in PH_GRASP
      float g;
      if (out.s < 0.5f) g = kGripOpen;
      else g = kGripOpen + (kGripPre - kGripOpen) * std::min(1.f, (out.s - 0.5f) / 0.2f);
      out.grip_target = g;
      if ((out.s >= 1.f && dist3(in.tcp_actual, out.tcp_target) < 0.02f) || phase_t_ >= 4.f)
        start_phase(PH_GRASP, in, out);
      break;
    }

    case PH_GRASP: {
      out.s = phase_t_ / phase_dur_;
      out.tcp_target[2] = kGraspZ;
      if (cur_slot_ >= 0 && cur_slot_ < in.n_cubes) {
        const CubeSlot& c = in.cubes[cur_slot_];
        out.tcp_target[0] = c.x + wiggle_dx_;
        out.tcp_target[1] = c.y + wiggle_dy_;
      }
      out.grip_target = kGripClosed;
      if (in.grasped || phase_t_ >= 2.f) {
        if (in.grasped) {
          wiggle_tries_ = 0; wiggle_dx_ = wiggle_dy_ = 0.f;
          in_flight_ = true; flight_slot_ = cur_slot_;
          start_phase(PH_LIFT, in, out);
        } else if (wiggle_tries_ < 2) {
          // grasp-wiggle: shift approach target and retry
          wiggle_tries_++;
          const float a = 3.1f * (float)wiggle_tries_;
          wiggle_dx_ = 0.010f * cosf(a);
          wiggle_dy_ = 0.010f * sinf(a);
          slot_tries_[cur_slot_ >= 0 ? cur_slot_ : 0]++;
          start_phase(PH_HOVER, in, out);
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
      if (cur_slot_ >= 0 && cur_slot_ < in.n_cubes) {
        out.tcp_target[0] = in.cubes[cur_slot_].x;
        out.tcp_target[1] = in.cubes[cur_slot_].y;
      }
      out.tcp_target[2] = kTransZ;
      out.grip_target = kGripClosed;
      if ((out.s >= 1.f && in.tcp_actual[2] > kTransZ - 0.025f) || phase_t_ >= 3.f) {
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
      out.grip_target = kGripClosed;
      const float txy[3] = {out.tcp_target[0], out.tcp_target[1], in.tcp_actual[2]};
      if ((out.s >= 1.f && dist3(in.tcp_actual, txy) < 0.025f) || phase_t_ >= 4.f)
        start_phase(PH_PLACE, in, out);
      break;
    }

    case PH_PLACE: {
      out.s = phase_t_ / phase_dur_;
      const int col = clampf((float)next_color_, 0.f, (float)(kNumColors - 1));
      out.tcp_target[0] = zones[col][0];
      out.tcp_target[1] = zones[col][1];
      out.tcp_target[2] = kPlaceZ0 + zone_stack[col] * kStackDz;
      out.grip_target = out.s > 0.6f ? kGripOpen : kGripClosed;
      if ((out.s >= 1.f && dist3(in.tcp_actual, out.tcp_target) < 0.02f) || phase_t_ >= 3.f) {
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
