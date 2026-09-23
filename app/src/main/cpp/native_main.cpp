// native_main.cpp — NativeActivity entry point (pure C++, no Java overhead)
// Panda Cube Sorter — autark Android app for the S26 Ultra.
//
// v1.3.0 camera + windows + motion manager:
//   * full touch camera: ONE finger orbits the scene, TWO fingers pinch-zoom
//     and pan the look-at target (reset via the NN window "KAM" button)
//   * flicker root fix (field diagnosis: "Kamera-Bild wird kurz Vollbild"):
//     the PiP camera image is now drawn as a fullscreen quad THROUGH a
//     viewport+scissor locked to its rectangle — it can physically never
//     extend beyond its region again, whatever races the driver throws at it
//   * NN info window (NN button): architecture summary + live values
//     (events, spikes, embedding norm, free energy, LoRA eta/V, MoE max,
//     phase, per-stage micro timings)
//   * motion manager (MOT button): AUFZ records the last 2.5 s of real joint
//     motion, UMW converts clips to 32-d features + persists motions.bin,
//     TRAIN runs analytic LoRA-Lyapunov updates on the dataset, LOESCH clears
//
// v1.0.1 black-screen fixes:
//   * two EGL contexts from ONE share group: the execution-loop thread owns
//     the event-camera FBO (a context-local object), the worker thread renders
//     the interactive view. An EGL context can only be current on ONE thread —
//     the old single-context design silently dropped GL calls.
//   * status/error screens: any init failure now shows a color-coded screen
//     with a numeric code instead of an eternal black screen.
//     Codes: L0 loading | E2 scene.mjb missing | E3 weights.bin missing
//            E4 temp file failed | E5 MJB load failed | E6 weights invalid
//            E7 EGL failed | E8 GL init failed
//   * window destroy/recreate fully tears down and restarts the pipeline
//     (old code kept rendering to a dead surface -> black after resume).
//   * pullable error report at <internalDataPath>/pcs_error.txt.
//
// v1.0.2 E7 fix (loop-thread EGL binding):
//   * field report: status screen showed E7 (= ERR_EGL). For E7 to be VISIBLE
//     the worker must already render fine — the failure was the LOOP thread's
//     eglMakeCurrent on the SHARED window surface. Some drivers refuse a
//     window surface current in two threads at once.
//   * the loop only renders into an FBO (no swap) — it never needs the window.
//     New fallback chain: surfaceless context -> 1x1 pbuffer -> dedicated
//     second window surface -> legacy shared surface. Every step logged.
//   * error screen now shows the raw EGL error as a hex sub-line (e.g. E7 /
//     0X3009); pcs_error.txt carries EGL/GL driver strings + the decision log.
#include <android/native_activity.h>
#include <android/native_window.h>
#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <android/asset_manager.h>
#include <android/log.h>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <mujoco/mujoco.h>

#include "glue/sim_glue.h"
#include "pcs/controller.h"
#include "gles_renderer.h"
#include "android_asset.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "panda-sorter", __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, "panda-sorter", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "panda-sorter", __VA_ARGS__)

using namespace pcs;

// stage codes for the status screen
enum {
  ST_LOADING = 0,   // assets loading / GL warm-up
  ST_RUNNING = 1,   // pipeline live
  ERR_SCENE = 2,    // scene.mjb missing
  ERR_WEIGHTS = 3,  // weights.bin missing
  ERR_TMPFILE = 4,  // internal storage write failed
  ERR_MJB = 5,      // MuJoCo model load failed
  ERR_WFORMAT = 6,  // weights.bin invalid
  ERR_EGL = 7       // EGL init failed
};

static std::atomic<bool> g_running{false};
static std::atomic<bool> g_new_episode{false};
static std::atomic<bool> g_has_surface{false};
static std::atomic<int> g_err{ST_LOADING};     // 0 = loading, 1 = running, >=2 = error
static std::atomic<bool> g_gl_ready{false};    // gles_init done (loop thread)
static std::atomic<bool> g_paused{false};      // START/PAUSE button
static std::atomic<int> g_fine_until{0};       // show "FINE" in the HUD until cycle N

static ANativeWindow* g_window = nullptr;
static std::mutex g_window_mutex;

static SimGlue g_glue;
static Controller g_ctrl;
static std::vector<uint8_t> g_frame(kEvW * kEvH * 3);

static AInputQueue* g_queue = nullptr;
static int looper_callback(int fd, int events, void* data);

// ---------------- v1.4.0 touch: press feedback, fire-on-UP, pip tap --------
// DOWN  -> visual press only (button shrinks/brightens); PiP becomes a tap
// MOVE  -> press follows the finger (leaving the button un-presses)
// UP    -> the action fires ONLY if the finger is still on the element
//          (standard mobile UX: no accidental triggers, visible feedback)
// Encoded ids: 0..5 bottom-bar buttons, 100+wb window buttons, pip tap flag.

// ---------------- v1.3.0 touch gestures (orbit / zoom / pan) ----------------
// One finger drag on the scene  -> orbit the camera around the target.
// Two finger pinch              -> zoom (dolly). Two finger drag -> pan.
// Touches on buttons/windows/pip are consumed by the UI handlers and never
// start a gesture. A gesture ENDS when a finger lifts — the surviving
// finger does not jump the camera.
struct Gesture {
  bool cam = false;      // gesture active (owns the camera)
  bool two = false;      // two-finger mode
  float lx = 0, ly = 0;  // last single-finger position
  float p0x = 0, p0y = 0, p1x = 0, p1y = 0;  // last two-finger positions
  float last_dist = 0.f; // last pinch distance
  float lmx = 0, lmy = 0;  // last midpoint
};
static Gesture g_gest;

// pending press state (input thread only — single-threaded by design)
static int g_press_id = -1;      // pressed element, -1 none
static bool g_pip_tap = false;   // PiP tap candidate (fires on UP)
static float g_tap_x0 = 0.f, g_tap_y0 = 0.f;

static void ui_clear_press() {
  g_press_id = -1;
  g_pip_tap = false;
  gles_set_pressed(-1);
  gles_set_pressed_wb(-1);
}

static void ui_fire_btn(int btn) {
  PcsUiState s;
  switch (btn) {
    case BTN_START: s.toggle_run = true; break;
    case BTN_STOP:  s.stop = true; break;
    case BTN_FINE:  s.fine = true; break;
    case BTN_NEW:   s.new_episode = true; break;
    case BTN_NN:    gles_toggle_window(WIN_NN); return;
    case BTN_MOT:   gles_toggle_window(WIN_MOT); return;
    default: return;
  }
  gles_push_ui(s);
}

static void ui_fire_wb(int wb) {
  PcsUiState s;
  switch (wb) {
    case 1: gles_toggle_window(WIN_NN); break;      // NN window close
    case 2: gles_cam_reset();                       // NN window: KAM
      gles_toast("KAMERA ZURÜCKGESETZT", TOAST_BLUE);
      break;
    case 3: gles_toggle_window(WIN_MOT); break;     // MOT window close
    case 4: s.mot_record = true; break;             // AUFZ
    case 5: s.mot_convert = true; break;            // UMW
    case 6: s.mot_train = true; break;              // TRAIN
    case 7:                                          // LOESCH (2-step)
      if (!gles_mot_confirm_armed()) {
        gles_mot_arm_confirm();
        gles_toast("LÖSCHEN: ERNEUT DRÜCKEN", TOAST_RED);
        return;
      }
      gles_mot_disarm();
      s.mot_clear = true;
      break;
    default: return;                                 // 8: window body
  }
  if (wb >= 4 && wb <= 7) gles_push_ui(s);
}

// ACTION_DOWN: begin a press (visual only). true = touch consumed by UI.
static bool handle_ui_down(float x, float y) {
  ui_clear_press();
  if (gles_hit_pip(x, y)) {           // robot camera: tap toggles large view
    g_pip_tap = true;
    g_tap_x0 = x; g_tap_y0 = y;
    return true;
  }
  const int btn = gles_hit_button(x, y);
  if (btn >= 0) {
    g_press_id = btn;
    gles_set_pressed(btn);
    return true;
  }
  const int wb = gles_hit_window_button(x, y);
  if (wb > 0) {
    if (wb != 8) {                    // window body consumes silently
      g_press_id = 100 + wb;
      gles_set_pressed_wb(wb);
    }
    return true;
  }
  return false;                       // scene -> camera gesture
}

// ACTION_MOVE: keep the press alive only while the finger stays on it.
// A PiP tap that turns into a drag becomes a camera orbit.
static void handle_ui_move(float x, float y) {
  if (g_pip_tap) {
    const float dx = x - g_tap_x0, dy = y - g_tap_y0;
    if (dx * dx + dy * dy > 24.f * 24.f) {
      g_pip_tap = false;
      g_gest.cam = true;
      g_gest.two = false;
      g_gest.lx = x; g_gest.ly = y;
    }
    return;
  }
  if (g_press_id < 0) return;
  if (g_press_id < 100) {
    gles_set_pressed(gles_hit_button(x, y) == g_press_id ? g_press_id : -1);
  } else {
    const int wb = g_press_id - 100;
    gles_set_pressed_wb(gles_hit_window_button(x, y) == wb ? wb : -1);
  }
}

// ACTION_UP: fire the action if the finger is still on the pressed element
static void handle_ui_up(float x, float y) {
  if (g_pip_tap) {
    ui_clear_press();
    gles_pip_set_big(!gles_pip_big());
    return;
  }
  const int id = g_press_id;
  ui_clear_press();
  if (id < 0) return;
  if (id < 100) {
    if (gles_hit_button(x, y) == id) ui_fire_btn(id);
  } else {
    const int wb = id - 100;
    if (gles_hit_window_button(x, y) == wb) ui_fire_wb(wb);
  }
}

static void handle_motion_event(AInputEvent* ev) {
  const int32_t action = AMotionEvent_getAction(ev);
  const int32_t plain = action & AMOTION_EVENT_ACTION_MASK;
  const int n = (int)AMotionEvent_getPointerCount(ev);

  switch (plain) {
    case AMOTION_EVENT_ACTION_DOWN: {
      const float x = AMotionEvent_getX(ev, 0);
      const float y = AMotionEvent_getY(ev, 0);
      if (handle_ui_down(x, y)) { g_gest = Gesture{}; return; }
      g_gest.cam = true;
      g_gest.two = false;
      g_gest.lx = x; g_gest.ly = y;
      return;
    }
    case AMOTION_EVENT_ACTION_POINTER_DOWN: {
      // a second finger cancels any pending press / tap
      ui_clear_press();
      if (g_gest.cam && n >= 2) {
        // switch to two-finger mode: seed pinch + pan reference
        g_gest.two = true;
        g_gest.p0x = AMotionEvent_getX(ev, 0);
        g_gest.p0y = AMotionEvent_getY(ev, 0);
        g_gest.p1x = AMotionEvent_getX(ev, 1);
        g_gest.p1y = AMotionEvent_getY(ev, 1);
        const float dx = g_gest.p1x - g_gest.p0x;
        const float dy = g_gest.p1y - g_gest.p0y;
        g_gest.last_dist = sqrtf(dx * dx + dy * dy);
        g_gest.lmx = 0.5f * (g_gest.p0x + g_gest.p1x);
        g_gest.lmy = 0.5f * (g_gest.p0y + g_gest.p1y);
      }
      return;
    }
    case AMOTION_EVENT_ACTION_MOVE: {
      const float x = AMotionEvent_getX(ev, 0);
      const float y = AMotionEvent_getY(ev, 0);
      if (g_pip_tap || g_press_id >= 0) { handle_ui_move(x, y); return; }
      if (!g_gest.cam) return;
      if (!g_gest.two) {
        const float dx = x - g_gest.lx, dy = y - g_gest.ly;
        g_gest.lx = x; g_gest.ly = y;
        // 0.0038 rad/px: a full-screen drag turns the scene ~1.3x
        gles_cam_orbit(dx * 0.0038f, dy * 0.0038f);
      } else {
        if (n < 2) return;
        const float x0 = AMotionEvent_getX(ev, 0);
        const float y0 = AMotionEvent_getY(ev, 0);
        const float x1 = AMotionEvent_getX(ev, 1);
        const float y1 = AMotionEvent_getY(ev, 1);
        const float dx = x1 - x0, dy = y1 - y0;
        const float d = sqrtf(dx * dx + dy * dy);
        const float mx = 0.5f * (x0 + x1), my = 0.5f * (y0 + y1);
        if (g_gest.last_dist > 1.f && d > 1.f) {
          // fingers apart (d grows) -> factor < 1 -> dolly IN
          gles_cam_zoom(g_gest.last_dist / d);
          gles_cam_pan(mx - g_gest.lmx, my - g_gest.lmy, gles_win_w(),
                       gles_win_h());
        }
        g_gest.p0x = x0; g_gest.p0y = y0;
        g_gest.p1x = x1; g_gest.p1y = y1;
        g_gest.last_dist = d;
        g_gest.lmx = mx; g_gest.lmy = my;
      }
      return;
    }
    case AMOTION_EVENT_ACTION_POINTER_UP:
      // a two-finger gesture loses a finger -> end it completely
      ui_clear_press();
      g_gest = Gesture{};
      return;
    case AMOTION_EVENT_ACTION_UP: {
      const float x = AMotionEvent_getX(ev, 0);
      const float y = AMotionEvent_getY(ev, 0);
      handle_ui_up(x, y);
      g_gest = Gesture{};
      return;
    }
    case AMOTION_EVENT_ACTION_CANCEL:
      ui_clear_press();
      g_gest = Gesture{};
      return;
    default:
      return;
  }
}

// ------------------------ diagnostics ------------------------
static ANativeActivity* g_activity = nullptr;
static std::mutex g_diag_mutex;
static std::string g_diag;                 // rolling EGL/GL decision + failure log
static std::atomic<unsigned> g_sub{0};     // last EGL error code (HUD hex sub-line)
static std::atomic<bool> g_drv_logged{false};
static EGLDisplay g_display = EGL_NO_DISPLAY;  // single EGL display, both threads

static void diag_add(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  char line[256];
  vsnprintf(line, sizeof(line), fmt, ap);
  va_end(ap);
  std::lock_guard<std::mutex> lk(g_diag_mutex);
  if (g_diag.size() < 8000) {
    g_diag += line;
    g_diag += "\n";
  }
}

// driver identity — call once with a context current (GL strings)
static void diag_driver_info() {
  bool expect = false;
  if (!g_drv_logged.compare_exchange_strong(expect, true)) return;
  const char* ev = eglQueryString(g_display, EGL_VENDOR);
  const char* evv = eglQueryString(g_display, EGL_VERSION);
  const char* ext = eglQueryString(g_display, EGL_EXTENSIONS);
  diag_add("EGL_VENDOR=%s", ev ? ev : "?");
  diag_add("EGL_VERSION=%s", evv ? evv : "?");
  diag_add("EGL_KHR_surfaceless_context=%s",
           (ext && strstr(ext, "EGL_KHR_surfaceless_context")) ? "yes" : "no");
  const GLubyte* r = glGetString(GL_RENDERER);
  const GLubyte* v = glGetString(GL_VERSION);
  diag_add("GL_RENDERER=%s", r ? (const char*)r : "?");
  diag_add("GL_VERSION=%s", v ? (const char*)v : "?");
}

// ------------------------ EGL / window helpers ------------------------
static EGLSurface g_surface = EGL_NO_SURFACE;   // worker (view) window surface
static EGLContext g_ctx_loop = EGL_NO_CONTEXT;  // event-FBO owner
static EGLContext g_ctx_view = EGL_NO_CONTEXT;  // interactive view (share group)

// How the 100 Hz loop thread binds EGL. The loop only renders into an FBO and
// glReadPixels — it never presents to the window.
// v1.2.0 FLICKER FIX: the loop thread is NEVER given a window surface again.
// The old fallback chain (window-own / window-shared) let the loop thread
// make a WINDOW surface current while the view thread was rendering to it —
// drivers requeue the buffer chain on such a call, so the view intermittently
// presented a stale buffer: the scene "jumped and came back".
// New chain: surfaceless -> 1x1 pbuffer -> NO GL (loop keeps running, the
// event pass is skipped, the last perception frame stays active).
enum LoopBind {
  LB_SURFACELESS = 0,   // EGL_KHR_surfaceless_context (loop needs no surface)
  LB_PBUFFER = 1,       // dedicated 1x1 pbuffer
  LB_NONE = 2           // no EGL binding: loop runs without the event pass
};
static const char* kBindName[3] = {"surfaceless", "pbuffer", "none"};
static LoopBind g_loop_bind = LB_SURFACELESS;
static EGLSurface g_loop_surface = EGL_NO_SURFACE;  // pbuffer

static bool choose_config(EGLConfig* cfg, bool allow_pbuffer) {
  const EGLint attrs[] = {
      EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
      EGL_SURFACE_TYPE,
      (EGLint)(EGL_WINDOW_BIT | (allow_pbuffer ? EGL_PBUFFER_BIT : 0)),
      EGL_BLUE_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_RED_SIZE, 8,
      EGL_DEPTH_SIZE, 16, EGL_NONE};
  EGLint n = 0;
  return eglChooseConfig(g_display, attrs, cfg, 1, &n) && n >= 1;
}

static bool init_egl(ANativeWindow* win) {
  if (g_display == EGL_NO_DISPLAY) {
    g_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (g_display == EGL_NO_DISPLAY) {
      diag_add("eglGetDisplay failed");
      return false;
    }
    if (!eglInitialize(g_display, nullptr, nullptr)) {
      diag_add("eglInitialize failed 0x%x", eglGetError());
      g_display = EGL_NO_DISPLAY;
      return false;
    }
    const char* ext = eglQueryString(g_display, EGL_EXTENSIONS);
    if (!(ext && strstr(ext, "EGL_KHR_surfaceless_context")))
      g_loop_bind = LB_PBUFFER;
  }

  EGLConfig cfg = 0;
  bool cfg_has_pb = true;
  if (!choose_config(&cfg, true)) {
    if (!choose_config(&cfg, false)) {
      diag_add("eglChooseConfig(8888) failed 0x%x", eglGetError());
      return false;
    }
    cfg_has_pb = false;
  }

  if (g_surface == EGL_NO_SURFACE) {
    g_surface = eglCreateWindowSurface(g_display, cfg, win, nullptr);
    if (g_surface == EGL_NO_SURFACE) {
      diag_add("eglCreateWindowSurface(8888) failed 0x%x", eglGetError());
      // legacy-driver fallback: RGB565 window config
      const EGLint attrs565[] = {
          EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_SURFACE_TYPE,
          EGL_WINDOW_BIT, EGL_BLUE_SIZE, 6, EGL_GREEN_SIZE, 6, EGL_RED_SIZE, 5,
          EGL_DEPTH_SIZE, 16, EGL_NONE};
      EGLConfig cfg565 = 0;
      EGLint n565 = 0;
      if (eglChooseConfig(g_display, attrs565, &cfg565, 1, &n565) && n565 >= 1)
        g_surface = eglCreateWindowSurface(g_display, cfg565, win, nullptr);
      if (g_surface == EGL_NO_SURFACE) {
        diag_add("eglCreateWindowSurface(565) failed 0x%x", eglGetError());
        return false;
      }
      cfg = cfg565;
      cfg_has_pb = false;
    }
  }

  const EGLint ctx_attrs[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
  if (g_ctx_loop == EGL_NO_CONTEXT) {
    g_ctx_loop = eglCreateContext(g_display, cfg, EGL_NO_CONTEXT, ctx_attrs);
    if (g_ctx_loop == EGL_NO_CONTEXT) {
      diag_add("eglCreateContext(loop) failed 0x%x", eglGetError());
      return false;
    }
  }
  if (g_ctx_view == EGL_NO_CONTEXT) {
    g_ctx_view = eglCreateContext(g_display, cfg, g_ctx_loop, ctx_attrs);
    if (g_ctx_view == EGL_NO_CONTEXT) {
      diag_add("eglCreateContext(view) failed 0x%x", eglGetError());
      return false;
    }
  }

  // v1.2.0 — loop-thread surface: surfaceless or pbuffer ONLY (never a
  // window surface, see LoopBind comment)
  if (g_loop_bind == LB_SURFACELESS) {
    // nothing to create; the loop binds with EGL_NO_SURFACE
  } else if (g_loop_surface == EGL_NO_SURFACE && cfg_has_pb) {
    const EGLint pb[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
    g_loop_surface = eglCreatePbufferSurface(g_display, cfg, pb);
    if (g_loop_surface == EGL_NO_SURFACE) {
      diag_add("eglCreatePbufferSurface failed 0x%x", eglGetError());
      g_loop_bind = LB_NONE;
    }
  }
  if (g_loop_bind == LB_PBUFFER && g_loop_surface == EGL_NO_SURFACE)
    g_loop_bind = LB_NONE;
  diag_add("loop bind=%s", kBindName[g_loop_bind]);
  return true;
}

static void destroy_egl() {
  if (g_display != EGL_NO_DISPLAY) {
    eglMakeCurrent(g_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    if (g_ctx_view != EGL_NO_CONTEXT) eglDestroyContext(g_display, g_ctx_view);
    if (g_ctx_loop != EGL_NO_CONTEXT) eglDestroyContext(g_display, g_ctx_loop);
    if (g_surface != EGL_NO_SURFACE) eglDestroySurface(g_display, g_surface);
    if (g_loop_surface != EGL_NO_SURFACE)
      eglDestroySurface(g_display, g_loop_surface);
    eglTerminate(g_display);
  }
  g_display = EGL_NO_DISPLAY;
  g_ctx_loop = g_ctx_view = EGL_NO_CONTEXT;
  g_surface = EGL_NO_SURFACE;
  g_loop_surface = EGL_NO_SURFACE;
}

static void destroy_surface_only() {
  if (g_display != EGL_NO_DISPLAY) {
    if (g_surface != EGL_NO_SURFACE) eglDestroySurface(g_display, g_surface);
    if (g_loop_surface != EGL_NO_SURFACE)
      eglDestroySurface(g_display, g_loop_surface);
  }
  g_surface = EGL_NO_SURFACE;
  g_loop_surface = EGL_NO_SURFACE;  // init_egl recreates it with the window
}

static void write_error_report(int err) {
  if (!g_activity || !g_activity->internalDataPath) return;
  std::string p = std::string(g_activity->internalDataPath) + "/pcs_error.txt";
  FILE* f = fopen(p.c_str(), "w");
  if (!f) return;
  fprintf(f,
          "PandaCubeSorter v1.3.0 | init error %d | sub 0x%X\n"
          "2=scene.mjb missing 3=weights.bin missing 4=storage write failed\n"
          "5=MJB load failed 6=weights invalid 7=EGL failed\n"
          "loop bind=%d (0=surfaceless 1=pbuffer 2=none)",
          err, g_sub.load(), (int)g_loop_bind);
  {
    std::lock_guard<std::mutex> lk(g_diag_mutex);
    if (!g_diag.empty()) fprintf(f, "--- diag ---\n%s", g_diag.c_str());
  }
  fclose(f);
}

// ------------------------ v1.3.0 motion manager ------------------------
// The user-facing "what you did with the pipeline" panel:
//   AUFZ  — keep the last 2.5 s of real MuJoCo joint motion as a clip
//   UMW   — convert clips into 32-d feature samples (the same 32-d scale the
//           FEP embedding lives on) and persist them to motions.bin
//   TRAIN — Lyapunov LoRA updates on the converted dataset (analytic,
//           divergence-free — same rule as the FINETUNE button)
//   LOESCH— drop clips + dataset
// All state lives on the loop thread; the renderer only gets copies.
struct MotSample {           // 11 floats, POD — persisted as-is
  float q[7];
  float grip;
  float tcp[3];
};
struct MotClip {
  std::vector<MotSample> s;
  bool converted = false;
};
static constexpr int kRingCap = 250;   // 2.5 s at 100 Hz
static std::vector<MotSample> g_ring;
static size_t g_ring_head = 0;
static std::vector<MotClip> g_clips;
static std::vector<std::array<float, 32>> g_dataset;
static int g_mot_updates = 0;
static float g_mot_last_norm = 0.f;
static char g_mot_msg[32] = "BEREIT";
static float g_last_h1[kMlpHidden] = {0};   // last MLP hidden state (loop)

static void mot_save();
static void mot_load();

static void mot_ring_push(const MotSample& s) {
  if ((int)g_ring.size() < kRingCap) {
    g_ring.push_back(s);
  } else {
    g_ring[g_ring_head] = s;
  }
  g_ring_head = (g_ring_head + 1) % kRingCap;
}

static std::vector<MotSample> mot_ring_take() {
  std::vector<MotSample> out;
  const size_t n = g_ring.size();
  if (n == 0) return out;
  out.reserve(n);
  if (n < (size_t)kRingCap) {
    out = g_ring;
  } else {
    for (size_t i = 0; i < n; ++i) out.push_back(g_ring[(g_ring_head + i) % n]);
  }
  return out;
}

// clips -> 32-d features (joint moments + gripper + tcp geometry), all
// normalized to the FEP embedding scale (~unit)
static void mot_convert_all() {
  int fresh = 0;
  for (auto& c : g_clips) {
    if (c.converted || c.s.size() < 8) continue;
    std::array<float, 32> f{};
    const float n = (float)c.s.size();
    for (int j = 0; j < 7; ++j) {
      float mean = 0, amean = 0, mn = 1e9f, mx = -1e9f;
      for (const auto& s : c.s) {
        const float v = s.q[j];
        mean += v; amean += fabsf(v);
        mn = std::min(mn, v); mx = std::max(mx, v);
      }
      mean /= n; amean /= n;
      float var = 0;
      for (const auto& s : c.s) { const float d = s.q[j] - mean; var += d * d; }
      var /= n;
      f[(size_t)j * 4 + 0] = mean / 3.f;       // Panda joint range ~ +-3 rad
      f[(size_t)j * 4 + 1] = amean / 3.f;
      f[(size_t)j * 4 + 2] = sqrtf(var);
      f[(size_t)j * 4 + 3] = (mx - mn) / 3.f;
    }
    {   // gripper: mean + range (slide range 0..0.04)
      float mean = 0, mn = 1e9f, mx = -1e9f;
      for (const auto& s : c.s) {
        mean += s.grip;
        mn = std::min(mn, s.grip); mx = std::max(mx, s.grip);
      }
      mean /= n;
      f[28] = mean / 0.04f;
      f[29] = (mx - mn) / 0.04f;
    }
    {   // tcp: path length + net displacement (typical reach ~0.5 m)
      float path = 0;
      float net[3] = {0, 0, 0};
      for (size_t i = 1; i < c.s.size(); ++i) {
        for (int k = 0; k < 3; ++k) {
          const float d = c.s[i].tcp[k] - c.s[i - 1].tcp[k];
          path += fabsf(d);
          net[k] += d;
        }
      }
      f[30] = path / 0.5f;
      f[31] = sqrtf(net[0]*net[0] + net[1]*net[1] + net[2]*net[2]) / 0.5f;
    }
    g_dataset.push_back(f);
    float nrm = 0;
    for (float v : f) nrm += v * v;
    g_mot_last_norm = sqrtf(nrm);
    c.converted = true;
    ++fresh;
  }
  if (fresh > 0) {
    snprintf(g_mot_msg, sizeof g_mot_msg, "UMW: %d NEU (%u)", fresh,
             (unsigned)g_dataset.size());
    mot_save();
  } else {
    snprintf(g_mot_msg, sizeof g_mot_msg, "NICHTS NEU ZUM UMWANDELN");
  }
}

// LoRA-Lyapunov training burst over the converted dataset
static void mot_train(Controller& c) {
  if (g_dataset.empty()) {
    snprintf(g_mot_msg, sizeof g_mot_msg, "ERST UMWANDELN (UMW)");
    return;
  }
  int updates = 0;
  for (const auto& f : g_dataset) {
    // the clip's mean posture, scaled into the tracking-error space the
    // Lyapunov rule expects (eta adapts, V = e^T e provably decreases)
    float e[kDof];
    for (int j = 0; j < kDof; ++j)
      e[j] = f[(size_t)j * 4 + 0] * 0.3f;
    c.lora.update(e, g_last_h1, c.lora_st, c.w);
    ++updates;
    if (updates >= 64) break;   // bounded burst per press
  }
  g_mot_updates += updates;
  snprintf(g_mot_msg, sizeof g_mot_msg, "TRAIN: %d UPDATES", updates);
}

static void mot_save() {
  if (!g_activity || !g_activity->internalDataPath) return;
  const std::string p =
      std::string(g_activity->internalDataPath) + "/motions.bin";
  FILE* f = fopen(p.c_str(), "wb");
  if (!f) return;
  const char magic[8] = "PCSMOT1";
  fwrite(magic, 1, 8, f);
  const uint32_t nc = (uint32_t)g_clips.size();
  const uint32_t ns = (uint32_t)g_dataset.size();
  fwrite(&nc, 4, 1, f);
  fwrite(&ns, 4, 1, f);
  for (const auto& c : g_clips) {
    const uint32_t n = (uint32_t)c.s.size();
    fwrite(&n, 4, 1, f);
    if (n) fwrite(c.s.data(), sizeof(MotSample), n, f);
  }
  for (const auto& s : g_dataset) fwrite(s.data(), sizeof(float), 32, f);
  fclose(f);
}

static void mot_load() {
  if (!g_activity || !g_activity->internalDataPath) return;
  const std::string p =
      std::string(g_activity->internalDataPath) + "/motions.bin";
  FILE* f = fopen(p.c_str(), "rb");
  if (!f) return;
  char magic[8] = {0};
  uint32_t nc = 0, ns = 0;
  if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "PCSMOT1", 8) != 0 ||
      fread(&nc, 4, 1, f) != 1 || fread(&ns, 4, 1, f) != 1 || nc > 64 ||
      ns > 4096) {
    fclose(f);
    return;
  }
  g_clips.clear();
  g_dataset.clear();
  bool ok = true;
  for (uint32_t i = 0; i < nc && ok; ++i) {
    uint32_t n = 0;
    MotClip c;
    if (fread(&n, 4, 1, f) != 1 || n > 4096) { ok = false; break; }
    c.s.resize(n);
    if (n && fread(c.s.data(), sizeof(MotSample), n, f) != n) { ok = false; break; }
    c.converted = true;
    g_clips.push_back(std::move(c));
  }
  for (uint32_t i = 0; i < ns && ok; ++i) {
    std::array<float, 32> s{};
    if (fread(s.data(), sizeof(float), 32, f) != 32) { ok = false; break; }
    g_dataset.push_back(s);
  }
  fclose(f);
  if (!ok) { g_clips.clear(); g_dataset.clear(); return; }
  snprintf(g_mot_msg, sizeof g_mot_msg, "GELADEN: %u CLIPS", nc);
}

static void mot_clear_all() {
  g_clips.clear();
  g_dataset.clear();
  g_mot_updates = 0;
  g_mot_last_norm = 0.f;
  mot_save();
  snprintf(g_mot_msg, sizeof g_mot_msg, "GELOSCHT");
}

static void mot_publish() {
  PcsMotState m;
  m.clips = (int)g_clips.size();
  m.samples = (int)g_dataset.size();
  m.last_feat_norm = g_mot_last_norm;
  m.updates = g_mot_updates;
  m.rec_left = 0;
  m.train_left = 0;
  snprintf(m.msg, sizeof m.msg, "%s", g_mot_msg);
  gles_set_motion(m);
}

// ------------------------ the 100 Hz execution loop ------------------------
static void execution_loop() {
  LOGI("execution loop started (bind=%s)", kBindName[g_loop_bind]);
  g_err = ST_LOADING;  // fresh attempt: a stale E7 from an earlier window
                       // must not stick to the new one

  bool bound = false;
  for (int attempt = 0; attempt < 4 && !bound; ++attempt) {
    if (!g_running || !g_has_surface) break;
    if (g_loop_bind == LB_NONE) break;
    EGLSurface draw = EGL_NO_SURFACE;
    if (g_loop_bind == LB_PBUFFER) draw = g_loop_surface;
    if (g_display == EGL_NO_DISPLAY ||
        (g_loop_bind != LB_SURFACELESS && draw == EGL_NO_SURFACE)) {
      diag_add("loop bind %s unusable (no display/surface)",
               kBindName[g_loop_bind]);
      g_loop_bind = LB_NONE;
      continue;
    }
    if (eglMakeCurrent(g_display, draw, draw, g_ctx_loop)) {
      bound = true;
      break;
    }
    const unsigned e = (unsigned)eglGetError();
    g_sub = e;
    diag_add("loop eglMakeCurrent(%s) failed 0x%x (attempt %d)",
             kBindName[g_loop_bind], e, attempt + 1);
    LOGE("loop eglMakeCurrent(%s) failed 0x%x", kBindName[g_loop_bind], e);
    // degrade: surfaceless -> pbuffer -> none (NEVER a window surface)
    if (g_loop_bind == LB_SURFACELESS) {
      g_loop_bind = LB_PBUFFER;
      if (g_loop_surface == EGL_NO_SURFACE) {
        EGLConfig cfg = 0;
        if (choose_config(&cfg, true) || choose_config(&cfg, false)) {
          const EGLint pb[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
          g_loop_surface = eglCreatePbufferSurface(g_display, cfg, pb);
        }
      }
      if (g_loop_surface == EGL_NO_SURFACE) g_loop_bind = LB_NONE;
    } else {
      g_loop_bind = LB_NONE;
    }
    if (g_loop_bind == LB_NONE) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
  }
  if (g_loop_bind == LB_NONE) {
    LOGW("loop thread: no EGL binding — event pass disabled, pipeline continues");
  }
  diag_add("loop bound via %s", kBindName[g_loop_bind]);

  // gles_init now runs on the VIEW thread (share group) before this loop
  // starts; only bind-less setups never ran it — guard anyway.
  if (!g_gl_ready && g_loop_bind != LB_NONE) {
    {
      std::lock_guard<std::mutex> lk(g_window_mutex);
      if (g_window) {
        gles_init(g_glue, ANativeWindow_getWidth(g_window),
                  ANativeWindow_getHeight(g_window), g_activity->assetManager);
        g_gl_ready = true;
        LOGI("gles_init done (loop context, fallback)");
      }
    }
  }

  ControllerOutput out;
  uint64_t episode = 1;
  g_glue.reset_episode(1000 + episode * 7919);
  gles_publish_poses(g_glue.model(), g_glue.data());  // view shows the scene immediately
  g_ctrl.reset();
  g_err = ST_RUNNING;
  mot_load();  // v1.3.0: restore the motion dataset from motions.bin

  auto next = std::chrono::steady_clock::now();
  const auto period = std::chrono::microseconds(10000);  // 100 Hz
  bool refresh = false;
  uint64_t cyc = 0;

  while (g_running && g_has_surface) {
    next += period;

    // ---- UI buttons (START/PAUSE, STOP, FINE, NEW, MOT-*) ----
    const PcsUiState ui = gles_take_ui();
    if (ui.toggle_run) {
      const bool now = !g_paused.load();
      g_paused = now;
      if (!now) g_glue.set_halt(false);  // START also releases a STOP
      LOGI("button: %s", now ? "PAUSE" : "START");
      gles_toast(now ? "PAUSIERT" : "GESTARTET",
                 now ? TOAST_AMBER : TOAST_GREEN);
    }
    if (ui.stop) {
      g_glue.set_halt(!g_glue.halted());  // STOP toggles freeze
      LOGI("button: STOP -> %s", g_glue.halted() ? "halt" : "run");
      gles_toast(g_glue.halted() ? "ROBOTER GESTOPPT" : "L\u00c4UFT WIEDER",
                 g_glue.halted() ? TOAST_RED : TOAST_GREEN);
    }
    if (ui.fine) {
      g_ctrl.finetune(32);
      g_fine_until = (int)cyc + 150;  // show "FINE" for ~1.5 s
      LOGI("button: FINETUNE burst");
      gles_toast("FINETUNE: 32 UPDATES", TOAST_ORANGE);
    }
    if (ui.new_episode) g_new_episode = true;
    if (ui.mot_record) {
      // v1.3.0: snapshot the last 2.5 s of real joint motion
      MotClip c;
      c.s = mot_ring_take();
      if (!c.s.empty()) {
        g_clips.push_back(std::move(c));
        snprintf(g_mot_msg, sizeof g_mot_msg, "AUFZ: %u SAMPLES",
                 (unsigned)g_clips.back().s.size());
        LOGI("motion: recorded %u samples (%u clips)",
             (unsigned)g_clips.back().s.size(), (unsigned)g_clips.size());
        gles_toast(g_mot_msg, TOAST_TEAL);
      } else {
        snprintf(g_mot_msg, sizeof g_mot_msg, "NOCH KEINE BEWEGUNG");
        gles_toast(g_mot_msg, TOAST_GRAY);
      }
    }
    if (ui.mot_convert) {
      mot_convert_all();
      gles_toast(g_mot_msg, TOAST_BLUE);
    }
    if (ui.mot_train) {
      mot_train(g_ctrl);
      gles_toast(g_mot_msg, TOAST_ORANGE);
    }
    if (ui.mot_clear) {
      mot_clear_all();
      gles_toast("MOTION-DATEN GEL\u00d6SCHT", TOAST_RED);
    }

    if (g_new_episode.exchange(false)) {
      ++episode;
      g_glue.reset_episode(1000 + episode * 7919);
      gles_publish_poses(g_glue.model(), g_glue.data());
      g_ctrl.reset();
      g_glue.set_halt(false);
      LOGI("new episode %llu", (unsigned long long)episode);
      gles_toast("NEUE EPISODE", TOAST_BLUE);
    }

    if (!g_paused.load()) {
      // ---- physics + perception + control (the whole pipeline) ----
      // GLES event-camera pass on THIS thread — the loop is bound to a
      // NON-window surface only (surfaceless/pbuffer), so it can never
      // interfere with the view thread's buffer chain (v1.2.0 flicker fix).
      if (g_loop_bind != LB_NONE && g_gl_ready.load()) {
        gles_render_event_frame(g_glue, refresh, g_frame.data(), kEvW, kEvH);
        // PiP shows the UNJITTERED camera: the refresh pulse (every 0.5 s)
        // micro-shifts the sensor for the event pipeline — feeding that
        // frame to the PiP made the robot view visibly jump twice a second.
        if (!refresh) gles_push_pip(g_frame.data(), kEvW, kEvH);
      }
      g_glue.step_cycle(g_ctrl, out, g_frame.data(), kEvW, kEvH, refresh);
      gles_publish_poses(g_glue.model(), g_glue.data());  // stable snapshot for the view thread
      refresh = (cyc % 50) == 49;  // micro-jitter pulse every 0.5 s
      out_stats_set(out.stats);
      out_task_set(out.task);

      // ---- v1.3.0: motion ring buffer + NN live state ----
      MotSample ms;
      g_glue.arm_state(ms.q, &ms.grip, ms.tcp);
      mot_ring_push(ms);
      memcpy(g_last_h1, out.mlp.h1, sizeof(g_last_h1));
      {
        PcsNnState nn;
        nn.events = out.events.n_events;
        nn.spikes = out.snn.n_spikes;
        float en = 0.f;
        for (int i = 0; i < kEmbDim; ++i) en += out.emb[i] * out.emb[i];
        nn.emb_norm = sqrtf(en);
        nn.free_energy = out.free_energy;
        nn.lora_eta = g_ctrl.lora_st.eta;
        nn.lora_v = g_ctrl.lora_st.v_prev;
        float mx = 0.f;
        for (int i = 0; i < kMlpOut; ++i) mx = std::max(mx, out.mlp.w[i]);
        nn.moe_max = mx;
        nn.phase = out.task.phase;
        nn.t_phys = out.stats.t_phys;
        nn.t_event = out.stats.t_event;
        nn.t_snn = out.stats.t_snn;
        nn.t_mlp = out.stats.t_mlp;
        gles_set_nn(nn);
      }
    }
    mot_publish();
    gles_set_diag((int)g_loop_bind, gles_gl_errs(), (long)cyc,
                  g_glue.sorted_count(), g_glue.total_cubes(),
                  g_glue.stacked_count(), g_paused.load(), g_glue.halted(),
                  (int)cyc < g_fine_until.load());

    // ---- pacing ----
    auto now = std::chrono::steady_clock::now();
    if (now < next) {
      std::this_thread::sleep_until(next);
    } else {
      if (out.stats.t_total > 8000 && (cyc % 500) == 0) {
        LOGW("cycle overrun: %u us", out.stats.t_total);
      }
      next = now + period;  // do not accumulate debt
    }
    ++cyc;
  }
  eglMakeCurrent(g_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
  LOGI("execution loop stopped");
}

// ------------------------ lifecycle callbacks ------------------------
static void onStart(ANativeActivity*) { LOGI("onStart"); }
static void onResume(ANativeActivity*) { LOGI("onResume"); }
static void onPause(ANativeActivity*) { LOGI("onPause"); }
static void onStop(ANativeActivity*) { LOGI("onStop"); }

static void onDestroy(ANativeActivity*) {
  LOGI("onDestroy");
  g_running = false;
  g_has_surface = false;
  std::lock_guard<std::mutex> lk(g_window_mutex);
  if (g_window) {
    ANativeWindow_release(g_window);
    g_window = nullptr;
  }
}

static void onNativeWindowCreated(ANativeActivity*, ANativeWindow* win) {
  LOGI("window created %dx%d", ANativeWindow_getWidth(win), ANativeWindow_getHeight(win));
  std::lock_guard<std::mutex> lk(g_window_mutex);
  if (g_window) ANativeWindow_release(g_window);
  g_window = win;
  ANativeWindow_acquire(win);
  gles_set_window_size(ANativeWindow_getWidth(win), ANativeWindow_getHeight(win));
  g_has_surface = true;
}

static void onNativeWindowResized(ANativeActivity*, ANativeWindow* win) {
  gles_set_window_size(ANativeWindow_getWidth(win), ANativeWindow_getHeight(win));
}

static void onNativeWindowDestroyed(ANativeActivity*, ANativeWindow*) {
  LOGI("window destroyed");
  g_has_surface = false;  // worker tears down surface + loop thread
  std::lock_guard<std::mutex> lk(g_window_mutex);
  if (g_window) {
    ANativeWindow_release(g_window);
    g_window = nullptr;
  }
}

static void onInputQueueCreated(ANativeActivity*, AInputQueue* queue) {
  ALooper* looper = ALooper_forThread();
  if (!looper) looper = ALooper_prepare(ALOOPER_PREPARE_ALLOW_NON_CALLBACKS);
  AInputQueue_attachLooper(queue, looper, 1, looper_callback, queue);
  g_queue = queue;
}

static int looper_callback(int fd, int events, void* data) {
  AInputQueue* queue = (AInputQueue*)data;
  AInputEvent* ev = nullptr;
  while (AInputQueue_getEvent(queue, &ev) >= 0) {
    if (AInputEvent_getType(ev) == AINPUT_EVENT_TYPE_MOTION)
      handle_motion_event(ev);
    AInputQueue_finishEvent(queue, ev, 1);
  }
  return 1;
}

static void onConfigurationChanged(ANativeActivity*) {}
static void onLowMemory(ANativeActivity*) {}

// ------------------------ main worker thread ------------------------
static void app_worker(ANativeActivity* activity) {
  // 1) load assets (scene + weights) — errors go to the status screen, never
  //    abort: the user sees E2..E6 instead of a black screen.
  AAssetManager* am = activity->assetManager;
  bool pipeline_ok = true;
  std::vector<uint8_t> scene_data, weights_data;

  if (!read_asset(am, "scene.mjb", scene_data)) {
    LOGE("scene.mjb not found in assets");
    g_err = ERR_SCENE;
    pipeline_ok = false;
  } else if (!read_asset(am, "weights.bin", weights_data)) {
    LOGE("weights.bin not found in assets");
    g_err = ERR_WEIGHTS;
    pipeline_ok = false;
  } else {
    LOGI("assets loaded: scene=%zu KB weights=%zu KB", scene_data.size() / 1024,
         weights_data.size() / 1024);
    std::string tmp = std::string(activity->internalDataPath) + "/scene.mjb";
    FILE* f = fopen(tmp.c_str(), "wb");
    if (!f) {
      LOGE("cannot open temp scene file");
      g_err = ERR_TMPFILE;
      pipeline_ok = false;
    } else {
      fwrite(scene_data.data(), 1, scene_data.size(), f);
      fclose(f);
      if (!g_glue.load_mjb_file(tmp.c_str())) {
        LOGE("scene load failed: %s", g_glue.last_error());
        g_err = ERR_MJB;
        pipeline_ok = false;
      } else if (!g_ctrl.load_weights(weights_data.data(), weights_data.size())) {
        LOGE("weights load failed: %s", g_ctrl.w.last_error.c_str());
        g_err = ERR_WFORMAT;
        pipeline_ok = false;
      } else {
        LOGI("model loaded: nq=%ld nu=%ld", (long)g_glue.model()->nq,
         (long)g_glue.model()->nu);
      }
    }
  }
  if (!pipeline_ok) {
    write_error_report(g_err.load());
    LOGE("init error %d — status screen active", g_err.load());
  }

  // 2) main loop: (re)create EGL on every window, run view or status screen
  bool loop_started = false;
  bool worker_gl_current = false;
  std::thread loop;

  while (g_running) {
    if (!g_has_surface) {
      // tear down surface-bound state until a new window arrives
      if (loop_started) {
        if (loop.joinable()) loop.join();
        loop_started = false;
      }
      if (worker_gl_current) {
        eglMakeCurrent(g_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        worker_gl_current = false;
      }
      if (g_surface != EGL_NO_SURFACE) destroy_surface_only();
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      continue;
    }

    ANativeWindow* win;
    { std::lock_guard<std::mutex> lk(g_window_mutex); win = g_window; }
    if (!win) { std::this_thread::sleep_for(std::chrono::milliseconds(20)); continue; }

    if (g_surface == EGL_NO_SURFACE) {
      if (!init_egl(win)) {
        const unsigned e = (unsigned)eglGetError();
        g_sub = e;
        diag_add("worker init_egl failed 0x%x", e);
        LOGE("EGL init failed 0x%x", e);
        if (g_err.load() < ST_RUNNING) g_err = ERR_EGL;
        write_error_report(g_err.load());
        destroy_egl();
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        continue;
      }
      if (!eglMakeCurrent(g_display, g_surface, g_surface, g_ctx_view)) {
        const unsigned e = (unsigned)eglGetError();
        g_sub = e;
        diag_add("worker eglMakeCurrent(view) failed 0x%x", e);
        LOGE("worker eglMakeCurrent failed 0x%x", e);
        if (g_err.load() < ST_RUNNING) g_err = ERR_EGL;
        write_error_report(g_err.load());
        destroy_surface_only();
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        continue;
      }
      worker_gl_current = true;
      eglSwapInterval(g_display, 1);
      diag_driver_info();
      // v1.2.0: GL objects are built ONCE on the view context (share group).
      // The loop thread only uses them via its own context — and only with a
      // non-window binding. This also guarantees a visible scene even when
      // the loop ends up without any EGL binding (LB_NONE).
      if (!g_gl_ready.load()) {
        gles_init(g_glue, ANativeWindow_getWidth(win),
                  ANativeWindow_getHeight(win), g_activity->assetManager);
        g_gl_ready = true;
        LOGI("gles_init done (view context)");
      }
      LOGI("EGL ready (share group: loop+view contexts)");
    }

    if (pipeline_ok && !loop_started) {
      loop = std::thread(execution_loop);
      loop_started = true;
    }

    if (worker_gl_current) {
      const int err_now = g_err.load();
      if (err_now >= 2 || !g_gl_ready || !pipeline_ok) {
        // loading / error screen (own minimal GL setup on ctx_view)
        int w = gles_win_w(), h = gles_win_h();
        if (w < 1 || h < 1) {
          w = ANativeWindow_getWidth(win);
          h = ANativeWindow_getHeight(win);
        }
        const int code = (err_now >= 2) ? err_now : ST_LOADING;
        char sub[16] = {0};
        if (err_now >= 2 && g_sub.load())
          snprintf(sub, sizeof sub, "0X%04X", g_sub.load());
        gles_render_status(code, w, h, sub[0] ? sub : nullptr);
      } else {
        gles_render_view(g_glue, g_ctrl);
      }
      if (!eglSwapBuffers(g_display, g_surface)) {
        LOGW("eglSwapBuffers failed 0x%x", eglGetError());
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    }
  }

  if (loop.joinable()) loop.join();
  destroy_egl();
  LOGI("worker done");
}

// ------------------------ entry point ------------------------
extern "C" JNIEXPORT void ANativeActivity_onCreate(ANativeActivity* activity,
                                                   void* savedState,
                                                   size_t savedStateSize) {
  activity->callbacks->onStart = onStart;
  activity->callbacks->onResume = onResume;
  activity->callbacks->onPause = onPause;
  activity->callbacks->onStop = onStop;
  activity->callbacks->onDestroy = onDestroy;
  activity->callbacks->onNativeWindowCreated = onNativeWindowCreated;
  activity->callbacks->onNativeWindowResized = onNativeWindowResized;
  activity->callbacks->onNativeWindowDestroyed = onNativeWindowDestroyed;
  activity->callbacks->onConfigurationChanged = onConfigurationChanged;
  activity->callbacks->onLowMemory = onLowMemory;
  activity->callbacks->onInputQueueCreated = onInputQueueCreated;

  g_running = true;
  g_activity = activity;
  std::thread(app_worker, activity).detach();
  LOGI("NativeActivity created");
}
