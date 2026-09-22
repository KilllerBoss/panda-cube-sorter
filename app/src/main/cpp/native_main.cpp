// native_main.cpp — NativeActivity entry point (pure C++, no Java overhead)
// Panda Cube Sorter — autark Android app for the S26 Ultra.
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
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
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

static ANativeWindow* g_window = nullptr;
static std::mutex g_window_mutex;

static SimGlue g_glue;
static Controller g_ctrl;
static std::vector<uint8_t> g_frame(kEvW * kEvH * 3);

static AInputQueue* g_queue = nullptr;
static int looper_callback(int fd, int events, void* data);

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
// glReadPixels — it never presents to the window. v1.0.1 made the loop share
// the worker's WINDOW surface; drivers that refuse a window surface current
// in two threads returned EGL errors there -> visible E7. Fallback chain,
// first fit wins (chosen in init_egl, degraded live in execution_loop):
enum LoopBind {
  LB_SURFACELESS = 0,   // EGL_KHR_surfaceless_context (loop needs no surface)
  LB_PBUFFER = 1,       // dedicated 1x1 pbuffer
  LB_WINDOW_OWN = 2,    // dedicated second window surface on the same window
  LB_WINDOW_SHARED = 3  // legacy v1.0.1 behaviour (last resort)
};
static const char* kBindName[4] = {"surfaceless", "pbuffer", "window-own",
                                   "window-shared"};
static LoopBind g_loop_bind = LB_WINDOW_SHARED;
static EGLSurface g_loop_surface = EGL_NO_SURFACE;  // pbuffer / own window surf.

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
    if (ext && strstr(ext, "EGL_KHR_surfaceless_context"))
      g_loop_bind = LB_SURFACELESS;
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

  // v1.0.2 E7 fix — loop-thread surface, see LoopBind comment above
  if (g_loop_bind == LB_SURFACELESS) {
    // nothing to create; the loop binds with EGL_NO_SURFACE
  } else if (g_loop_surface == EGL_NO_SURFACE) {
    if (cfg_has_pb) {
      const EGLint pb[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
      g_loop_surface = eglCreatePbufferSurface(g_display, cfg, pb);
      if (g_loop_surface != EGL_NO_SURFACE) g_loop_bind = LB_PBUFFER;
      else diag_add("eglCreatePbufferSurface failed 0x%x", eglGetError());
    }
    if (g_loop_surface == EGL_NO_SURFACE) {
      g_loop_surface = eglCreateWindowSurface(g_display, cfg, win, nullptr);
      if (g_loop_surface != EGL_NO_SURFACE) {
        g_loop_bind = LB_WINDOW_OWN;
      } else {
        g_loop_bind = LB_WINDOW_SHARED;
        diag_add("loop falls back to window-shared");
      }
    }
  }
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
          "PandaCubeSorter v1.0.2 | init error %d | sub 0x%X\n"
          "2=scene.mjb missing 3=weights.bin missing 4=storage write failed\n"
          "5=MJB load failed 6=weights invalid 7=EGL failed\n"
          "loop bind=%d (0=surfaceless 1=pbuffer 2=window-own 3=window-shared)\n",
          err, g_sub.load(), (int)g_loop_bind);
  {
    std::lock_guard<std::mutex> lk(g_diag_mutex);
    if (!g_diag.empty()) fprintf(f, "--- diag ---\n%s", g_diag.c_str());
  }
  fclose(f);
}

// ------------------------ the 100 Hz execution loop ------------------------
static void execution_loop() {
  LOGI("execution loop started (bind=%s)", kBindName[g_loop_bind]);
  g_err = ST_LOADING;  // fresh attempt: a stale E7 from an earlier window
                       // must not stick to the new one

  bool bound = false;
  for (int attempt = 0; attempt < 4 && !bound; ++attempt) {
    if (!g_running || !g_has_surface) break;
    EGLSurface draw = EGL_NO_SURFACE;
    if (g_loop_bind == LB_PBUFFER || g_loop_bind == LB_WINDOW_OWN)
      draw = g_loop_surface;
    else if (g_loop_bind == LB_WINDOW_SHARED)
      draw = g_surface;
    if (g_display == EGL_NO_DISPLAY ||
        (g_loop_bind != LB_SURFACELESS && draw == EGL_NO_SURFACE)) {
      diag_add("loop bind %s unusable (no display/surface)",
               kBindName[g_loop_bind]);
      g_loop_bind = LB_WINDOW_SHARED;
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
    // degrade one step down the chain
    if (g_loop_bind == LB_SURFACELESS) g_loop_bind = LB_PBUFFER;
    else if (g_loop_bind == LB_PBUFFER) g_loop_bind = LB_WINDOW_OWN;
    else g_loop_bind = LB_WINDOW_SHARED;
    // a pbuffer/own-window surface may not exist yet — create it here if the
    // window is still alive, otherwise skip to the legacy binding
    if ((g_loop_bind == LB_PBUFFER || g_loop_bind == LB_WINDOW_OWN) &&
        g_loop_surface == EGL_NO_SURFACE) {
      EGLConfig cfg = 0;
      if (choose_config(&cfg, true) || choose_config(&cfg, false)) {
        if (g_loop_bind == LB_PBUFFER) {
          const EGLint pb[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
          g_loop_surface = eglCreatePbufferSurface(g_display, cfg, pb);
        } else {
          ANativeWindow* w = nullptr;
          {
            std::lock_guard<std::mutex> lk(g_window_mutex);
            w = g_window;
            if (w) ANativeWindow_acquire(w);
          }
          if (w) {
            g_loop_surface = eglCreateWindowSurface(g_display, cfg, w, nullptr);
            ANativeWindow_release(w);
          }
        }
        if (g_loop_surface == EGL_NO_SURFACE)
          diag_add("loop surface create failed 0x%x", eglGetError());
      }
      if ((g_loop_bind == LB_PBUFFER || g_loop_bind == LB_WINDOW_OWN) &&
          g_loop_surface == EGL_NO_SURFACE)
        g_loop_bind = LB_WINDOW_SHARED;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
  }
  if (!bound) {
    LOGE("loop thread: no EGL binding possible");
    if (g_err.load() < ST_RUNNING) g_err = ERR_EGL;
    write_error_report(g_err.load());
    return;
  }
  diag_add("loop bound via %s", kBindName[g_loop_bind]);

  if (!g_gl_ready) {
    {
      std::lock_guard<std::mutex> lk(g_window_mutex);
      if (!g_window) {
        eglMakeCurrent(g_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        return;
      }
      gles_init(g_glue, ANativeWindow_getWidth(g_window),
                ANativeWindow_getHeight(g_window));
    }
    g_gl_ready = true;
    LOGI("gles_init done (loop context)");
  }

  ControllerOutput out;
  uint64_t episode = 1;
  g_glue.reset_episode(1000 + episode * 7919);
  g_ctrl.reset();
  g_err = ST_RUNNING;

  auto next = std::chrono::steady_clock::now();
  const auto period = std::chrono::microseconds(10000);  // 100 Hz
  bool refresh = false;
  uint64_t cyc = 0;

  while (g_running && g_has_surface) {
    next += period;

    if (g_new_episode.exchange(false)) {
      ++episode;
      g_glue.reset_episode(1000 + episode * 7919);
      g_ctrl.reset();
      LOGI("new episode %llu", (unsigned long long)episode);
    }

    // ---- physics + perception + control (the whole pipeline) ----
    // GLES event-camera pass on THIS thread (context owner, single GL thread)
    gles_render_event_frame(g_glue, refresh, g_frame.data(), kEvW, kEvH);
    g_glue.step_cycle(g_ctrl, out, g_frame.data(), kEvW, kEvH, refresh);
    refresh = (cyc % 50) == 49;  // micro-jitter pulse every 0.5 s
    out_stats_set(out.stats);
    out_task_set(out.task);

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
    if (AInputEvent_getType(ev) == AINPUT_EVENT_TYPE_MOTION &&
        AMotionEvent_getAction(ev) == AMOTION_EVENT_ACTION_DOWN) {
      g_new_episode = true;
    }
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
