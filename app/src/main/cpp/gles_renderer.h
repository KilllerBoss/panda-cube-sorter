// gles_renderer.h — GLES 3.0 rendering of the MuJoCo scene + event-camera FBO
#pragma once
#include <android/asset_manager.h>
#include <cstdint>
#include "glue/sim_glue.h"
#include "pcs/controller.h"

using namespace pcs;

// one-time GL setup (program, buffers, Roboto glyph atlas); called with the
// EGL context current
void gles_init(SimGlue& glue, int win_w, int win_h, AAssetManager* am);

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
  BTN_NN,          // v1.3.0: toggle the neural-net info window
  BTN_MOT,         // v1.3.0: toggle the motion-manager window
  BTN_RL,          // v1.6.0: toggle the RL-training window
  BTN_COUNT
};

// ---------------- v1.3.0 windows (NN info + motion manager) ----------------

enum PcsWindow { WIN_NN = 0, WIN_MOT, WIN_RL, WIN_COUNT };

// toggle a window on/off (called directly from the input thread)
void gles_toggle_window(int which);
bool gles_window_open(int which);

// hit-test INSIDE the open windows (call AFTER gles_hit_button).
// returns 0 = no window hit, else an action id:
//   1 NN close   2 NN camera-reset
//   3 MOT close  4 MOT record (AUFZ)  5 MOT convert (UMW)
//   6 MOT train  7 MOT clear (LOESCH)
//  20 RL close  21 RL train start/stop  22 RL best  23 RL reset
int gles_hit_window_button(float x, float y);

// ---------------- v1.3.0 touch camera (orbit / zoom / pan) ----------------
// one finger drag  -> orbit around the table target
// two finger pinch -> zoom (dolly)
// two finger drag  -> pan the look-at target
void gles_cam_orbit(float d_az, float d_el);
void gles_cam_zoom(float dist_factor);
void gles_cam_pan(float dx_screen, float dy_screen, int win_w, int win_h);
void gles_cam_reset();

// window-coords touch point -> button id, or -1 if none hit
int gles_hit_button(float x, float y);

// UI flags written by the input thread, consumed (and cleared) by the loop
struct PcsUiState {
  bool toggle_run = false;  // START/PAUSE was pressed
  bool stop = false;        // STOP pressed
  bool fine = false;        // FINETUNE pressed
  bool new_episode = false; // NEW pressed
  // v1.3.0 motion-manager actions (MOT window buttons)
  bool mot_record = false;  // AUFZ: snapshot the last ~2.5 s of joint motion
  bool mot_convert = false; // UMW: motion clips -> 32-d features + dataset file
  bool mot_train = false;   // TRAIN: LoRA-Lyapunov updates on the dataset
  bool mot_clear = false;   // LOESCH: drop all clips + dataset
  // v1.6.0 RL-window actions
  bool rl_train_toggle = false;  // TRAINIEREN / STOPP
  bool rl_best = false;          // BESTE WERTE: apply best-ever parameters
  bool rl_reset = false;         // policy back to v1.6.0 defaults
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

// ---------------- v1.3.0 live NN + motion state (loop -> window) ----------------

// published by the 100 Hz loop each cycle; shown live in the NN window
struct PcsNnState {
  uint32_t events = 0, spikes = 0;
  float emb_norm = 0.f;        // ||32-d FEP embedding||
  float free_energy = 0.f;     // predictive-coding error energy
  float lora_eta = 0.f;        // adaptive Lyapunov gain
  float lora_v = 0.f;          // V = e^T e (must decrease)
  float moe_max = 0.f;         // max mixture weight after softmax
  int   phase = 0;             // task phase id
  uint32_t t_phys = 0, t_event = 0, t_snn = 0, t_mlp = 0;  // us per stage
};
void gles_set_nn(const PcsNnState& s);

// published by the loop after motion-manager actions
struct PcsMotState {
  int   clips = 0;             // recorded motion clips
  int   samples = 0;           // converted 32-d feature samples in the dataset
  float last_feat_norm = 0.f;  // ||feature|| of the last conversion
  int   updates = 0;           // LoRA updates done by the last TRAIN press
  int   rec_left = 0;          // >0: countdown "recording ..." (cycles)
  int   train_left = 0;        // >0: countdown "training ..." (updates left)
  char  msg[32] = {0};         // short status text for the MOT window
};
void gles_set_motion(const PcsMotState& s);

// ---------------- v1.6.0 RL training state (loop -> RL window) -------------

struct PcsRlState {
  bool  train_active = false;  // training episodes running
  int   episodes = 0;          // completed training episodes (all time)
  float rate_all = 0.f;        // success rate over all episodes (0..1)
  float rate_recent = 0.f;     // success rate, last 20 episodes
  float reward_last = 0.f;
  float reward_best = 0.f;
  int   ep_sorted = 0;         // current episode progress
  int   ep_total = 0;
  float theta[14] = {0};       // mu, normalized 0..1 per parameter
  float sigma_mean = 0.f;      // mean exploration width (0..1)
  char  msg[32] = {0};
};
void gles_set_rl(const PcsRlState& s);

// ---------------- v1.4.0 UI redesign (style, feedback, toasts) ----------------

// toast accent colors (index into the renderer palette)
enum PcsToastColor {
  TOAST_GREEN = 0, TOAST_AMBER, TOAST_RED, TOAST_BLUE,
  TOAST_VIOLET, TOAST_TEAL, TOAST_ORANGE, TOAST_GRAY
};

// show a toast notification (centered above the action bar, auto-fades).
// Any thread; latest toast wins. msg: ASCII (A-Z 0-9 - : / . auml ouml uuml).
void gles_toast(const char* msg, int color);

// pressed-state feedback (input thread calls on ACTION_DOWN / MOVE / UP)
void gles_set_pressed(int btn);       // -1 clears the bottom-bar press
void gles_set_pressed_wb(int id);     // -1 clears the window-button press

// tap-to-enlarge robot camera: hit test (window coords) + reset
bool gles_hit_pip(float x, float y);
void gles_pip_set_big(bool big);
bool gles_pip_big();

// two-step confirm for destructive MOT actions (LOESCH): arm -> press again
// within 3 s -> action fires; auto-expires (checked by the draw loop)
void gles_mot_arm_confirm();
bool gles_mot_confirm_armed();  // auto-expires on read
void gles_mot_disarm();
