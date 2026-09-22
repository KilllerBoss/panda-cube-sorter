// native_main.cpp — NativeActivity entry point (pure C++, no Java overhead)
// Panda Cube Sorter — autark Android app for the S26 Ultra.
//
// Responsibilities:
//   * EGL/GLES 3.0 context + window lifecycle (NativeActivity callbacks)
//   * load scene.mjb + weights.bin from the APK assets into memory
//   * 100 Hz execution loop on a dedicated thread (10 ms budget):
//       mj_step x5 -> event camera (FBO) -> ALIF-LSNN -> predictive coding
//       -> 32-d embedding -> Soft-MoE -> compiled KAN-MLP (+LoRA) -> torques
//   * minimal touch UI (tap = new episode / new random cube layout)
#include <android/native_activity.h>
#include <android/native_window.h>
#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <android/asset_manager.h>
#include <android/log.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
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

static std::atomic<bool> g_running{false};
static std::atomic<bool> g_new_episode{false};
static ANativeWindow* g_window = nullptr;
static std::mutex g_window_mutex;

static SimGlue g_glue;
static Controller g_ctrl;
static std::vector<uint8_t> g_frame(kEvW * kEvH * 3);

// called from the UI thread on taps (minimal UI)
static AInputQueue* g_queue = nullptr;
static int looper_callback(int fd, int events, void* data);

// ------------------------ the 100 Hz execution loop ------------------------
static void execution_loop() {
  LOGI("execution loop started");

  ControllerOutput out;
  uint64_t episode = 1;
  g_glue.reset_episode(1000 + episode * 7919);
  g_ctrl.reset();

  auto next = std::chrono::steady_clock::now();
  const auto period = std::chrono::microseconds(10000);  // 100 Hz
  bool refresh = false;
  uint64_t cyc = 0;

  while (g_running) {
    next += period;

    if (g_new_episode.exchange(false)) {
      ++episode;
      g_glue.reset_episode(1000 + episode * 7919);
      g_ctrl.reset();
      LOGI("new episode %llu", (unsigned long long)episode);
    }

    // ---- physics + perception + control (the whole pipeline) ----
    {
      // GLES event-camera pass happens on this thread right before control:
      // the renderer is owned by this thread (single-threaded GL).
      gles_render_event_frame(g_glue, refresh, g_frame.data(), kEvW, kEvH);

      g_glue.step_cycle(g_ctrl, out, g_frame.data(), kEvW, kEvH, refresh);
      refresh = (cyc % 50) == 49;  // micro-jitter pulse every 0.5 s
      out_stats_set(out.stats);
      out_task_set(out.task);
    }

    // ---- pacing ----
    auto now = std::chrono::steady_clock::now();
    if (now < next) {
      std::this_thread::sleep_until(next);
    } else {
      // budget overrun: log occasionally (thermal guard should keep this rare)
      if (out.stats.t_total > 8000 && (cyc % 500) == 0) {
        LOGW("cycle overrun: %u us", out.stats.t_total);
      }
      next = now + period;  // do not accumulate debt
    }
    ++cyc;
  }
  LOGI("execution loop stopped");
}

// ------------------------ EGL / window helpers ------------------------
static EGLDisplay g_display = EGL_NO_DISPLAY;
static EGLSurface g_surface = EGL_NO_SURFACE;
static EGLContext g_context = EGL_NO_CONTEXT;

static bool init_egl(ANativeWindow* win) {
  g_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  if (g_display == EGL_NO_DISPLAY) return false;
  if (!eglInitialize(g_display, nullptr, nullptr)) return false;
  const EGLint cfg_attrs[] = {
      EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
      EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
      EGL_BLUE_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_RED_SIZE, 8,
      EGL_DEPTH_SIZE, 16, EGL_NONE};
  const EGLint ctx_attrs[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
  EGLConfig cfg;
  EGLint n = 0;
  if (!eglChooseConfig(g_display, cfg_attrs, &cfg, 1, &n) || n < 1) return false;
  g_surface = eglCreateWindowSurface(g_display, cfg, win, nullptr);
  if (g_surface == EGL_NO_SURFACE) return false;
  g_context = eglCreateContext(g_display, cfg, nullptr, ctx_attrs);
  if (g_context == EGL_NO_CONTEXT) return false;
  if (!eglMakeCurrent(g_display, g_surface, g_context, g_context)) return false;
  return true;
}

static void destroy_egl() {
  if (g_display != EGL_NO_DISPLAY) {
    eglMakeCurrent(g_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    if (g_context != EGL_NO_CONTEXT) eglDestroyContext(g_display, g_context);
    if (g_surface != EGL_NO_SURFACE) eglDestroySurface(g_display, g_surface);
    eglTerminate(g_display);
  }
  g_display = EGL_NO_DISPLAY;
  g_context = EGL_NO_CONTEXT;
  g_surface = EGL_NO_SURFACE;
}

// ------------------------ lifecycle callbacks ------------------------
static void onStart(ANativeActivity*) { LOGI("onStart"); }
static void onResume(ANativeActivity*) { LOGI("onResume"); }
static void onPause(ANativeActivity*) { LOGI("onPause"); }
static void onStop(ANativeActivity*) { LOGI("onStop"); }

static void onDestroy(ANativeActivity* activity) {
  LOGI("onDestroy");
  g_running = false;
  std::lock_guard<std::mutex> lk(g_window_mutex);
  if (g_window) {
    ANativeWindow_release(g_window);
    g_window = nullptr;
  }
}

static void onNativeWindowCreated(ANativeActivity* activity, ANativeWindow* win) {
  LOGI("window created %dx%d", ANativeWindow_getWidth(win), ANativeWindow_getHeight(win));
  std::lock_guard<std::mutex> lk(g_window_mutex);
  g_window = win;
  ANativeWindow_acquire(win);
}

static void onNativeWindowDestroyed(ANativeActivity*, ANativeWindow*) {
  LOGI("window destroyed");
  std::lock_guard<std::mutex> lk(g_window_mutex);
  if (g_window) {
    ANativeWindow_release(g_window);
    g_window = nullptr;
  }
}

static int looper_callback(int fd, int events, void* data);

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
  // 1) load assets (scene + weights)
  AAssetManager* am = activity->assetManager;
  std::vector<uint8_t> scene_data, weights_data;
  if (!read_asset(am, "scene.mjb", scene_data)) {
    LOGE("scene.mjb not found in assets");
    return;
  }
  if (!read_asset(am, "weights.bin", weights_data)) {
    LOGE("weights.bin not found in assets");
    return;
  }
  LOGI("assets loaded: scene=%zu KB weights=%zu KB", scene_data.size() / 1024,
       weights_data.size() / 1024);

  // 2) write the scene to a temp file (mj_loadModel needs a path) and load
  std::string tmp = activity->internalDataPath;
  tmp += "/scene.mjb";
  {
    FILE* f = fopen(tmp.c_str(), "wb");
    if (!f) { LOGE("cannot open temp scene file"); return; }
    fwrite(scene_data.data(), 1, scene_data.size(), f);
    fclose(f);
  }
  if (!g_glue.load_mjb_file(tmp.c_str())) {
    LOGE("scene load failed: %s", g_glue.last_error());
    return;
  }
  if (!g_ctrl.load_weights(weights_data.data(), weights_data.size())) {
    LOGE("weights load failed: %s", g_ctrl.w.last_error.c_str());
    return;
  }
  LOGI("model loaded: nq=%d nu=%d", g_glue.model()->nq, g_glue.model()->nu);

  // 3) wait for a window, then run the render+loop pipeline
  bool started = false;
  std::thread loop;
  while (g_running) {
    {
      std::lock_guard<std::mutex> lk(g_window_mutex);
      if (g_window && !started) {
        if (!init_egl(g_window)) {
          LOGE("EGL init failed");
          destroy_egl();
        } else {
          gles_init(g_glue, ANativeWindow_getWidth(g_window),
                    ANativeWindow_getHeight(g_window));
          g_running = true;
          loop = std::thread(execution_loop);
          started = true;
          LOGI("render+loop pipeline started");
        }
      }
    }
    if (started) {
      // render the interactive view at vsync rate (separate from control)
      gles_render_view(g_glue, g_ctrl);
      eglSwapBuffers(g_display, g_surface);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(8));
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
  activity->callbacks->onNativeWindowDestroyed = onNativeWindowDestroyed;
  activity->callbacks->onConfigurationChanged = onConfigurationChanged;
  activity->callbacks->onLowMemory = onLowMemory;
  activity->callbacks->onInputQueueCreated = onInputQueueCreated;

  g_running = true;
  std::thread(app_worker, activity).detach();
  LOGI("NativeActivity created");
}
