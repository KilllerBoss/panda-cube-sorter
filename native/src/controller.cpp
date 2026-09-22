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
  auto t1 = clk::now();

  snn.step(out.events.bins, w, out.snn);
  auto t2 = clk::now();

  coder.step(out.events.bins, out.snn, w, out.emb, out.free_energy);
  auto t3 = clk::now();

  // ---------- decode cube map ----------
  TaskInput tin;
  tin.dt = in.dt;
  for (int j = 0; j < kDof; ++j) tin.q[j] = in.q[j];
  tin.grip = in.grip;
  tin.grasped = in.grasped;
  for (int k = 0; k < 3; ++k) tin.tcp_actual[k] = in.tcp_actual[k];
  for (int j = 0; j < kDof; ++j) tin.q_goal_ik[j] = in.q_goal_ik[j];
  tin.contact_l = in.contact_l;
  tin.contact_r = in.contact_r;
  tin.n_cubes = kNumCubes;
  float dec[kDecOut];
  for (int o = 0; o < kDecOut; ++o) {
    float acc = w.dec_b[o];
    for (int d = 0; d < kEmbDim; ++d) acc += out.emb[d] * w.dec_w[(size_t)d * kDecOut + o];
    dec[o] = acc;
  }
  for (int c = 0; c < kNumCubes; ++c) {
    CubeSlot& s = tin.cubes[c];
    s.x = 0.38f + dec[c];                       // x offset around scene center
    s.y = dec[kNumCubes + c];
    int best = 0;
    float bv = dec[2 * kNumCubes + c];
    for (int k = 1; k < kNumColors; ++k) {
      const float v = dec[2 * kNumCubes + k * kNumCubes + c];
      if (v > bv) { bv = v; best = k; }
    }
    s.color = best;
    s.score = bv;                               // max color logit = confidence
    for (int k = 0; k < kNumColors; ++k)
      s.color_logits[k] = dec[2 * kNumCubes + k * kNumCubes + c];
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

}  // namespace pcs
