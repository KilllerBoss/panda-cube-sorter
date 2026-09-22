// pcs/controller.cpp — 100 Hz pipeline orchestration (Phase 2/3/5 execution)
#include "pcs/controller.h"
#include <chrono>

namespace pcs {

bool Controller::load_weights(const uint8_t* blob, size_t size) {
  if (!w.load_from_memory(blob, size)) return false;
  reset();
  return true;
}

bool Controller::load_weights_file(const char* path) {
  if (!w.load_from_file(path)) return false;
  reset();
  return true;
}

void Controller::reset() {
  snn.reset();
  coder.reset();
  lora_st.reset();
  task.reset();
  for (int i = 0; i < kLoraRank; ++i) {
    for (int h = 0; h < kMlpHidden; ++h)
      lora_st.A[(size_t)h * kLoraRank + i] = w.lora_a[(size_t)h * kLoraRank + i];
    for (int o = 0; o < kMlpOut; ++o)
      lora_st.B[(size_t)i * kMlpOut + o] = w.lora_b[(size_t)i * kMlpOut + o];
  }
}

void Controller::cycle(const ControllerInput& in, ControllerOutput& out) {
  using clk = std::chrono::steady_clock;
  auto t0 = clk::now();

  // ---------- Phase 2: perception ----------
  cam.process(in.frame_rgb, in.frame_w, in.frame_h, in.refresh_pulse, out.events);
  if (in.refresh_pulse) {
    memcpy(pulse_bins_, out.events.bins, sizeof(pulse_bins_));  // latch snapshot
  }
  auto t1 = clk::now();

  snn.step(out.events.bins, w, out.snn);
  auto t2 = clk::now();

  coder.step(out.events.bins, out.snn, w, out.emb, out.free_energy);
  auto t3 = clk::now();

  TaskInput tin;   // filled by the detection head below + task inputs here
  // ---------- task input ----------
  tin.dt = in.dt;
  for (int j = 0; j < kDof; ++j) tin.q[j] = in.q[j];
  tin.grip = in.grip;
  tin.grasped = in.grasped;
  for (int k = 0; k < 3; ++k) tin.tcp_actual[k] = in.tcp_actual[k];
  for (int j = 0; j < kDof; ++j) tin.q_goal_ik[j] = in.q_goal_ik[j];
  tin.contact_l = in.contact_l;
  tin.contact_r = in.contact_r;
  tin.n_cubes = kNumCubes;
  {
    // ---- analytic detection head on the latched pulse frame ----
    // 12x8 cell grid; hue channels at cell*6 + {2..5} = r,g,b,y.
    // Camera affine (cam_event at (0.42,0,1.05), fovy 78, depth ~0.75 m):
    //   world_x = 0.42 + (u-0.5)*1.62 ; world_y = (0.5-v)*1.215
    float act[4][kBinCols * kBinRows];
    for (int k = 0; k < 4; ++k)
      for (int cell = 0; cell < kBinCols * kBinRows; ++cell)
        act[k][cell] = pulse_bins_[(size_t)cell * kBinsPerCell + 2 + k];
    // mask the sorting zones (same colors as the cubes!) — fixed positions
    static const float kZoneXY[4][2] = {{0.36f,-0.26f},{0.36f,0.26f},{0.50f,-0.12f},{0.50f,0.12f}};
    bool masked[kBinCols * kBinRows];
    for (int cell = 0; cell < kBinCols * kBinRows; ++cell) {
      const int r = cell / kBinCols, c = cell % kBinCols;
      const float wx = 0.42f + (((float)c + 0.5f) / kBinCols - 0.5f) * 1.62f;
      const float wy = (0.5f - ((float)r + 0.5f) / kBinRows) * 1.215f;
      masked[cell] = false;
      for (int z = 0; z < 4; ++z) {
        const float dx = wx - kZoneXY[z][0], dy = wy - kZoneXY[z][1];
        if (dx * dx + dy * dy < 0.145f * 0.145f) { masked[cell] = true; break; }
      }
    }
    float sm[4][kBinCols * kBinRows];
    for (int k = 0; k < 4; ++k)
      for (int r = 0; r < kBinRows; ++r)
        for (int c = 0; c < kBinCols; ++c) {
          float acc = 0; int wn = 0;
          for (int dr = -1; dr <= 1; ++dr)
            for (int dc = -1; dc <= 1; ++dc) {
              const int rr = r + dr, cc = c + dc;
              if (rr < 0 || rr >= kBinRows || cc < 0 || cc >= kBinCols) continue;
              acc += act[k][rr * kBinCols + cc]; wn++;
            }
          sm[k][r * kBinCols + c] = masked[r * kBinCols + c] ? 0.f : acc / (float)wn;
        }
    int slot = 0;
    for (int k = 0; k < kNumColors && slot < kNumCubes; ++k) {
      for (int peak = 0; peak < 2 && slot < kNumCubes; ++peak) {
        int best = -1; float bv = 0.03f;
        for (int cell = 0; cell < kBinCols * kBinRows; ++cell) {
          if (sm[k][cell] <= bv) continue;
          const int r = cell / kBinCols, c = cell % kBinCols;
          bool is_max = true;
          for (int dr = -1; dr <= 1 && is_max; ++dr)
            for (int dc = -1; dc <= 1 && is_max; ++dc) {
              const int rr = r + dr, cc = c + dc;
              if (rr < 0 || rr >= kBinRows || cc < 0 || cc >= kBinCols) continue;
              if (sm[k][rr * kBinCols + cc] > sm[k][cell] + 1e-6f) is_max = false;
            }
          if (is_max) { bv = sm[k][cell]; best = cell; }
        }
        if (best < 0) break;
        sm[k][best] = 0.f;
        const int r0 = best / kBinCols, c0 = best % kBinCols;
        float wsum = 0, uacc = 0, vacc = 0;
        for (int dr = -1; dr <= 1; ++dr)
          for (int dc = -1; dc <= 1; ++dc) {
            const int rr = r0 + dr, cc = c0 + dc;
            if (rr < 0 || rr >= kBinRows || cc < 0 || cc >= kBinCols) continue;
            const float w = sm[k][best] + act[k][rr * kBinCols + cc] * 0.5f;
            wsum += w;
            uacc += w * ((float)cc + 0.5f) / (float)kBinCols;
            vacc += w * ((float)rr + 0.5f) / (float)kBinRows;
          }
        if (wsum <= 0.f) continue;
        const float u = uacc / wsum, v = vacc / wsum;
        CubeSlot& sl = tin.cubes[slot];
        sl.x = 0.42f + (u - 0.5f) * 1.62f;
        sl.y = (0.5f - v) * 1.215f;
        sl.color = k;
        for (int kk = 0; kk < kNumColors; ++kk) sl.color_logits[kk] = (kk == k) ? 1.5f : -1.f;
        sl.score = bv;
        out.dbg_slots[slot] = sl;
        slot++;
      }
    }
    for (; slot < kNumCubes; ++slot) {
      CubeSlot& sl = tin.cubes[slot];
      sl.x = 0.38f; sl.y = 0.f; sl.color = 0; sl.score = -10.f;
      for (int kk = 0; kk < kNumColors; ++kk) sl.color_logits[kk] = -1.f;
      out.dbg_slots[slot] = sl;
    }
  }
  auto t4 = clk::now();

  // ---------- Phase 3: motor synthesis ----------
  task.update(tin, out.task);
  auto t5 = clk::now();

  float bias[kMlpOut];
  softmoe_bias(w, out.emb, bias);
  auto t6 = clk::now();

  float x[kMlpIn];
  for (int j = 0; j < kDof; ++j) {
    x[j] = out.task.q_goal[j] - out.task.q_start[j];        // dq_goal
    x[kDof + j] = out.task.q_start[j];                      // q_start
  }
  x[14] = out.task.s;
  x[15] = out.task.grip_target;
  for (int p = 0; p < kNumPhases; ++p) x[16 + p] = (out.task.phase == p) ? 1.f : 0.f;

  mlp.forward(x, w, bias, out.mlp);                          // logits+LoRA via caller
  // LoRA path: forward computed logits without adapter; add B*A*h1 then re-softmax
  float delta[kMlpOut];
  lora.forward(out.mlp.h1, lora_st);
  float mx = -1e30f, sum = 0.f;
  for (int o = 0; o < kMlpOut; ++o) {
    delta[o] = lora_st.delta[o];
    out.mlp.logits[o] += delta[o];
    if (out.mlp.logits[o] > mx) mx = out.mlp.logits[o];
  }
  for (int o = 0; o < kMlpOut; ++o) { out.mlp.w[o] = expf(out.mlp.logits[o] - mx); sum += out.mlp.w[o]; }
  for (int o = 0; o < kMlpOut; ++o) out.mlp.w[o] /= sum;
  auto t7 = clk::now();

  // ---------- trajectory texture ----------
  // NOTE: the geometric baseline (q_start -> q_goal * minjerk) is added by the
  // glue AFTER the IK update, so q_des tracks the freshest IK solution.
  proto.sample(out.mlp.w, out.task.s, w, out.q_add);
  for (int j = 0; j < kDof; ++j) out.q_des[j] = out.task.q_start[j];
  out.grip_target = out.task.grip_target;

  // ---------- Phase 5: Lyapunov LoRA adaptation from tracking error ----------
  float e[kDof];
  for (int j = 0; j < kDof; ++j) e[j] = out.q_des[j] - in.q[j];
  lora.update(e, out.mlp.h1, lora_st, w);
  // keep the sample for the FINETUNE button burst
  for (int j = 0; j < kDof; ++j) last_e_[j] = e[j];
  for (int h = 0; h < kMlpHidden; ++h) last_h1_[h] = out.mlp.h1[h];
  has_sample_ = true;
  auto t8 = clk::now();

  // ---------- stats ----------
  auto us = [](std::chrono::steady_clock::duration d) {
    return (uint32_t)std::chrono::duration_cast<std::chrono::microseconds>(d).count(); };
  out.stats.t_event = us(t1 - t0);
  out.stats.t_snn   = us(t2 - t1) + out.stats.t_event;   // cumulative so far
  out.stats.t_moe   = us(t4 - t3);
  out.stats.t_mlp   = us(t7 - t6);
  out.stats.t_lora  = us(t8 - t7);
  out.stats.t_task  = us(t5 - t4);
  out.stats.t_total = us(t8 - t0);
  out.stats.t_phys  = 0;   // measured by the glue around mj_step
  out.stats.n_events = out.events.n_events;
  out.stats.n_spikes = out.snn.n_spikes;
  out.stats.cycles++;
}

// FINETUNE button: replay the Lyapunov update on the last stored tracking
// error a few times — a cheap on-device burst that tightens the adapter
// around the currently observed error without any extra data.
void Controller::finetune(int iters) {
  if (!has_sample_) return;
  const int n = iters > 0 ? iters : 16;
  for (int i = 0; i < n; ++i) lora.update(last_e_, last_h1_, lora_st, w);
}

}  // namespace pcs
