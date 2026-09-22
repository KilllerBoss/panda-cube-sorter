// gles_renderer.h — GLES 3.0 rendering of the MuJoCo scene + event-camera FBO
#pragma once
#include <cstdint>
#include "glue/sim_glue.h"
#include "pcs/controller.h"

using namespace pcs;

// one-time GL setup (program, buffers); called with the EGL context current
void gles_init(SimGlue& glue, int win_w, int win_h);

// interactive view (cam_track), rendered at display rate
void gles_render_view(SimGlue& glue, Controller& ctrl);

// event-camera pass: renders cam_event into a 96x72 FBO and reads it back
// into `rgb` (the same pipeline as the desktop CPU projector)
void gles_render_event_frame(SimGlue& glue, bool jitter, uint8_t* rgb, int w, int h);

// minimal HUD (latency budget, events, spikes) drawn over the view
void gles_render_hud(const pcs::CycleStats& st, const pcs::TaskOutput& task);

// HUD data bridge (control thread -> render thread)
void out_stats_set(const pcs::CycleStats& st);
void out_task_set(const pcs::TaskOutput& t);

// window size tracking (safe before gles_init; also updates the view camera)
void gles_set_window_size(int w, int h);
int gles_win_w();
int gles_win_h();

// pose snapshot: the 100 Hz loop thread publishes after each cycle; the view
// thread renders from this stable copy (no mjData race)
void gles_publish_poses(const mjModel* m, const mjData* d);

// status / error screen: code 0 = loading (amber), >=2 = error (red + code)
// renders "L0" or "E<n>" with a 3x5 pixel font, works without gles_init.
// `sub` (optional) draws a smaller hex line below (e.g. "0X3009") — glyphs
// available: 0-9, A-Z, and a few punctuation marks.
void gles_render_status(int code, int win_w, int win_h, const char* sub = nullptr);

// ---------------- v1.1.0 UI ----------------

// on-screen buttons (bottom bar), drawn by the HUD, hit-tested by the
// input thread via gles_hit_button(); state flows through lock-free flags.
enum PcsButton {
  BTN_START = 0,   // toggle run/pause of the 100 Hz pipeline
  BTN_STOP,        // freeze the arm where it is, open the gripper
  BTN_FINE,        // LoRA finetune burst on the current tracking error
  BTN_NEW,         // new episode: re-randomize 4-8 cubes
  BTN_COUNT
};

// window-coords touch point -> button id, or -1 if none hit
int gles_hit_button(float x, float y);

// UI flags written by the input thread, consumed (and cleared) by the loop
struct PcsUiState {
  bool toggle_run = false;  // START/PAUSE was pressed
  bool stop = false;        // STOP pressed
  bool fine = false;        // FINETUNE pressed
  bool new_episode = false; // NEW pressed
};
PcsUiState gles_take_ui();
// input thread -> UI flags (OR-accumulated until the loop consumes them)
void gles_push_ui(const PcsUiState& s);

// loop thread -> renderer: latest perception frame for the picture-in-picture
// robot-camera view (copied under a light mutex, uploaded by the view thread)
void gles_push_pip(const uint8_t* rgb, int w, int h);

// loop thread -> HUD diagnostics line (visible WITHOUT adb)
void gles_set_diag(int bind, int gl_errs, long cycles, int sorted, int total,
                   int stacked, bool paused, bool halted, bool finetuning);

// number of view frames that ended with a GL error (shown as G<n> in HUD)
int gles_gl_errs();
