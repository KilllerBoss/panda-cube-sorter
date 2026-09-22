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

// status / error screen: code 0 = loading (amber), >=2 = error (red + code)
// renders "L0" or "E<n>" with a 3x5 pixel font, works without gles_init.
// `sub` (optional) draws a smaller hex line below (e.g. "0X3009") — glyphs
// available: 0-9, A-F, E, L, X.
void gles_render_status(int code, int win_w, int win_h, const char* sub = nullptr);
