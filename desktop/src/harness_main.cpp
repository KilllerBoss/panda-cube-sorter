// desktop/src/harness_main.cpp — desktop validation harness
// Modes:
//   eval  : N episodes, full pipeline (event camera via CPU projector),
//           reports success rate + per-stage timings
//   bench : warmup (1000 cycles) + timing benchmark (mj_step budget check)
//   stress: 50-cube scene physics benchmark
//
// The projector renders a tiny 96x72 RGB view from the fixed overhead event
// camera on the CPU (colored rectangles) — same downstream pipeline as the
// Android GLES FBO path.
#include "glue/sim_glue.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

using namespace pcs;

// ---------------- CPU projector (event camera substitute) ----------------
class CpuProjector {
 public:
  void init(SimGlue& g) {
    float fovy;
    g.cam_event_pose(cam_pos_, fwd_, up_, right_, &fovy);
    f_ = 0.5f * 72.f / tanf(fovy * 0.5f * (float)M_PI / 180.f);
    n_cubes_ = g.total_cubes();
    for (int i = 0; i < n_cubes_; ++i) {
      colors_[i][0] = 0.9f; colors_[i][1] = 0.1f; colors_[i][2] = 0.1f;  // set per color below
    }
    static const float col_tab[4][3] = {
      {0.92f, 0.10f, 0.10f}, {0.10f, 0.82f, 0.18f},
      {0.15f, 0.35f, 0.95f}, {0.95f, 0.85f, 0.12f}};
    for (int i = 0; i < n_cubes_; ++i) memcpy(colors_[i], col_tab[g.cube_color(i)], 3);
  }

  // jitter: lateral camera offset for refresh pulses
  void render(SimGlue& g, bool pulse, uint8_t* rgb) {
    const float W = kEvW, H = kEvH;
    // background = table
    for (int i = 0; i < kEvW * kEvH; ++i) {
      rgb[3 * i + 0] = (uint8_t)(0.45f * 255); rgb[3 * i + 1] = (uint8_t)(0.38f * 255);
      rgb[3 * i + 2] = (uint8_t)(0.30f * 255);
    }
    const float jx = pulse ? 0.008f : 0.f, jy = pulse ? 0.004f : 0.f;
    // zones (world rects, depth below cubes)
    static const float zone_col[4][3] = {
      {0.90f, 0.12f, 0.12f}, {0.12f, 0.80f, 0.20f},
      {0.15f, 0.35f, 0.95f}, {0.95f, 0.85f, 0.15f}};
    for (int c = 0; c < 4; ++c) {
      const float* zc = g.zone_pos(c);
      draw_rect(g, rgb, zc[0] - 0.07f + jx, zc[1] - 0.07f + jy, zc[0] + 0.07f + jx,
                zc[1] + 0.07f + jy, 0.3545f, zone_col[c]);
    }
    // cubes: AABB of projected corners, painter-sorted by distance to camera
    int order[64];
    for (int i = 0; i < n_cubes_; ++i) order[i] = i;
    std::sort(order, order + n_cubes_, [&](int a, int b) {
      return g.cube_pos(a)[2] < g.cube_pos(b)[2]; });  // lower = farther from top cam
    for (int oi = 0; oi < n_cubes_; ++oi) {
      const int i = order[oi];
      const float* p = g.cube_pos(i);
      float umin = 1e9f, vmin = 1e9f, umax = -1e9f, vmax = -1e9f;
      bool vis = false;
      for (int cz = -1; cz <= 1; cz += 2)
        for (int cy = -1; cy <= 1; cy += 2)
          for (int cx = -1; cx <= 1; cx += 2) {
            float pt[3] = {p[0] + cx * 0.025f + jx, p[1] + cy * 0.025f + jy, p[2] + cz * 0.025f};
            float u, v; float zc;
            if (project(pt, u, v, zc)) {
              vis = true;
              umin = std::min(umin, u); umax = std::max(umax, u);
              vmin = std::min(vmin, v); vmax = std::max(vmax, v);
            }
          }
      if (vis) fill_rect(rgb, umin, vmin, umax, vmax, colors_[i]);
    }
  }

 private:
  bool project(const float p[3], float& u, float& v, float& zc) {
    const float d[3] = {p[0] - cam_pos_[0], p[1] - cam_pos_[1], p[2] - cam_pos_[2]};
    zc = d[0] * fwd_[0] + d[1] * fwd_[1] + d[2] * fwd_[2];
    if (zc < 0.02f) return false;
    const float xc = d[0] * right_[0] + d[1] * right_[1] + d[2] * right_[2];
    const float yc = d[0] * up_[0] + d[1] * up_[1] + d[2] * up_[2];
    u = kEvW / 2.f + f_ * xc / zc;
    v = kEvH / 2.f - f_ * yc / zc;
    return u >= -8 && u < kEvW + 8 && v >= -8 && v < kEvH + 8;
  }
  void draw_rect(SimGlue& g, uint8_t* rgb, float x0, float y0, float x1, float y1,
                 float z, const float col[3]) {
    float u0, v0, u1, v1, u2, v2, u3, v3, zz;
    const float p0[3] = {x0, y0, z}, p1[3] = {x1, y0, z};
    const float p2[3] = {x1, y1, z}, p3[3] = {x0, y1, z};
    if (!project(p0, u0, v0, zz)) return;
    if (!project(p1, u1, v1, zz)) return;
    if (!project(p2, u2, v2, zz)) return;
    if (!project(p3, u3, v3, zz)) return;
    const float umin = std::min({u0, u1, u2, u3}), umax = std::max({u0, u1, u2, u3});
    const float vmin = std::min({v0, v1, v2, v3}), vmax = std::max({v0, v1, v2, v3});
    fill_rect(rgb, umin, vmin, umax, vmax, col);
  }
  void fill_rect(uint8_t* rgb, float u0, float v0, float u1, float v1,
                 const float col[3]) {
    const int x0 = std::max(0, (int)u0), y0 = std::max(0, (int)v0);
    const int x1 = std::min(kEvW - 1, (int)u1), y1 = std::min(kEvH - 1, (int)v1);
    for (int y = y0; y <= y1; ++y)
      for (int x = x0; x <= x1; ++x) {
        uint8_t* px = rgb + 3 * (y * kEvW + x);
        px[0] = (uint8_t)(col[0] * 255); px[1] = (uint8_t)(col[1] * 255);
        px[2] = (uint8_t)(col[2] * 255);
      }
  }
  float cam_pos_[3], fwd_[3], up_[3], right_[3], f_;
  int n_cubes_;
  float colors_[64][3];
};

// ---------------- stats helpers ----------------
struct Stat { std::vector<uint32_t> v;
  void add(uint32_t x) { v.push_back(x); }
  uint32_t pct(double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::min((size_t)(p * (v.size() - 1)), v.size() - 1)];
  } };

int main(int argc, char** argv) {
  std::string mode = "eval", scene = "scene/scene_8.mjb", weights = "weights/weights.bin";
  int episodes = 10;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&](const char* def) { return (i + 1 < argc) ? argv[++i] : def; };
    if (a == "--mode") mode = next("eval");
    else if (a == "--scene") scene = next(scene.c_str());
    else if (a == "--weights") weights = next(weights.c_str());
    else if (a == "--episodes") episodes = atoi(next("10"));
  }

  SimGlue glue;
  if (!glue.load_mjb_file(scene.c_str())) {
    fprintf(stderr, "scene load failed: %s\n", glue.last_error());
    return 1;
  }
  Controller ctrl;
  if (!ctrl.load_weights_file(weights.c_str())) {
    fprintf(stderr, "weights load failed: %s\n", ctrl.w.last_error.c_str());
    return 1;
  }
  CpuProjector proj;
  proj.init(glue);
  bool grasp_logged_ = false;

  ControllerOutput out;
  static uint8_t frame[kEvW * kEvH * 3];

  if (mode == "eval") {
    int total_sorted = 0, total_stacked = 0, total_cubes = 0;
    Stat phys, pipe, ev, snn, mlpt, lora;
    for (int ep = 0; ep < episodes; ++ep) {
      glue.reset_episode(1000 + ep * 7919);
      ctrl.reset();
      bool done = false;
      int cyc = 0;
      const int max_cycles = 30000;  // 300 s sim time cap
      while (!done && cyc < max_cycles) {
        bool pulse = (cyc % 50) == 0;             // refresh every 0.5 s
        proj.render(glue, pulse, frame);
        glue.step_cycle(ctrl, out, frame, kEvW, kEvH, pulse);
        phys.add(out.stats.t_phys); pipe.add(out.stats.t_total);
        ev.add(out.stats.n_events); snn.add(out.stats.n_spikes);
        mlpt.add(out.stats.t_mlp); lora.add(out.stats.t_lora);
        if (cyc == 100 && ep == 0 && getenv("PCS_DUMP")) {
          FILE* f = fopen("/home/z/my-project/logs/pulse_frame.ppm", "wb");
          fprintf(f, "P6\n%d %d\n255\n", kEvW, kEvH);
          fwrite(frame, 1, kEvW * kEvH * 3, f);
          fclose(f);
          f = fopen("/home/z/my-project/logs/pulse_bins.txt", "w");
          for (int b = 0; b < kNumBins; ++b)
            if (out.events.bins[b] > 0.02f)
              fprintf(f, "bin %d cell %d ch %d val %.3f\n", b, b / 6, b % 6, out.events.bins[b]);
          fclose(f);
          printf("  dumped pulse frame @c=100\n");
        }
        if (cyc % 2000 == 0 && ep == 0) {
          printf("  c=%5d %s dec0=(%.2f,%.2f) truth0=(%.2f,%.2f) dec4=(%.2f,%.2f) truth4=(%.2f,%.2f)\n",
                 cyc, kPhaseName[out.task.phase],
                 out.dbg_slots[0].x, out.dbg_slots[0].y,
                 glue.cube_pos(0)[0], glue.cube_pos(0)[1],
                 out.dbg_slots[4].x, out.dbg_slots[4].y,
                 glue.cube_pos(4)[0], glue.cube_pos(4)[1]);
        }
        if (getenv("PCS_TRACE") && out.task.phase == pcs::PH_GRASP && out.task.s > 0.95f
            && !grasp_logged_) {
          grasp_logged_ = true;
          printf("  [grasp-end] slot=%d grab=(%.3f,%.3f) ctrl_grip=%.4f ncon=%d\n",
                 out.task.in_flight, out.task.tcp_target[0], out.task.tcp_target[1],
                 glue.data()->ctrl[7], glue.data()->ncon);
        }
        if (out.task.phase != pcs::PH_GRASP) grasp_logged_ = false;
        done = out.task.episode_done;
        ++cyc;
      }
      const int srt = glue.sorted_count();
      const int stk = glue.stacked_count();
      total_sorted += srt; total_stacked += stk; total_cubes += glue.total_cubes();
      printf("episode %2d: cycles=%5d sorted=%d/%d stacked=%d  [phases end: %s]\n",
             ep, cyc, srt, glue.total_cubes(), stk, kPhaseName[out.task.phase]);
    }
    printf("\n=== RESULT: sorted %d/%d = %.1f%% (stacked %d) ===\n",
           total_sorted, total_cubes, 100.0 * total_sorted / (total_cubes ? total_cubes : 1),
           total_stacked);
    printf("timing us: phys p50=%u p95=%u | pipeline p50=%u p95=%u | mlp p50=%u | lora p50=%u\n",
           phys.pct(0.5), phys.pct(0.95), pipe.pct(0.5), pipe.pct(0.95),
           mlpt.pct(0.5), lora.pct(0.5));
    printf("events/frame p50=%u spikes/cycle p50=%u\n", ev.pct(0.5), snn.pct(0.5));
  } else if (mode == "bench") {
    // warmup: 1000 cycles (cache fill, DVFS stabilization)
    glue.reset_episode(42);
    for (int i = 0; i < 1000; ++i) {
      bool pulse = (i % 50) == 0;
      proj.render(glue, pulse, frame);
      glue.step_cycle(ctrl, out, frame, kEvW, kEvH, pulse);
    }
    Stat phys, pipe, ev;
    for (int i = 0; i < 2000; ++i) {
      bool pulse = (i % 50) == 0;
      proj.render(glue, pulse, frame);
      glue.step_cycle(ctrl, out, frame, kEvW, kEvH, pulse);
      phys.add(out.stats.t_phys); pipe.add(out.stats.t_total); ev.add(out.stats.n_events);
    }
    printf("bench: mj_step(5 substeps) p50=%u us p95=%u us | full pipeline p50=%u us\n",
           phys.pct(0.5), phys.pct(0.95), pipe.pct(0.5));
    printf("budget: %s (target phys<800us, total<10000us)\n",
           (phys.pct(0.95) < 800 && pipe.pct(0.95) < 10000) ? "PASS" : "CHECK");
  } else if (mode == "stress") {
    glue.randomize_cubes(777);
    for (int i = 0; i < 1000; ++i) {
      bool pulse = (i % 50) == 0;
      proj.render(glue, pulse, frame);
      glue.step_cycle(ctrl, out, frame, kEvW, kEvH, pulse);
    }
    Stat phys, pipe;
    for (int i = 0; i < 2000; ++i) {
      proj.render(glue, false, frame);
      glue.step_cycle(ctrl, out, frame, kEvW, kEvH, false);
      phys.add(out.stats.t_phys); pipe.add(out.stats.t_total);
    }
    printf("stress(50 cubes): phys p50=%u us p95=%u us | pipe p50=%u us\n",
           phys.pct(0.5), phys.pct(0.95), pipe.pct(0.5));
  } else {
    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 1;
  }
  return 0;
}
