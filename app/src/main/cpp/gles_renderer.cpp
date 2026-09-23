// gles_renderer.cpp — GLES 3.0 renderer: scene view + event-camera FBO + HUD
// Draws the scene as colored primitives (capsules/boxes) directly from
// mjData kinematics — no meshes, no textures, minimal GPU load.
#include "gles_renderer.h"
#include <android/log.h>
#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <android/log.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, "panda-sorter", __VA_ARGS__)

using namespace pcs;

// forward decl (defined below, used by gles_render_view)
static void gles_draw_pip();
// pixel-font helpers (defined in the status-screen section below)
static void text_quads(std::vector<float>& v, const char* s, float x0,
                       float y0, float scale, int win_w, int win_h);
static float text_width(const char* s, float scale);
// shared bottom-bar button layout (defined in the touch-buttons section)
void ui_button_rect(int i, float r[4]);
// v1.3.0/1.4.0 overlay windows (NN info + motion manager)
static void ui_window_rect(int which, float r[4]);
static bool ui_window_button_rect(int win, int btn, float r[4]);

// UI/diag state — written by input + loop threads, read by the view thread.
// Guarded by one small mutex; contention is negligible (few writes/cycle).
static std::mutex g_ui_mtx;
static PcsUiState g_ui;
struct DiagState {
  int bind = 0;
  int gl_errs = 0;
  long cycles = 0;
  int sorted = 0, total = 0, stacked = 0;
  bool paused = false, halted = false, finetuning = false;
};
static DiagState g_diag;

// ---------------- v1.3.0 touch camera (orbit / zoom / pan) ----------------
struct ViewState {
  float az = 2.46f;     // orbit azimuth (rad) — default = the v1.2 viewpoint
  float el = 0.417f;    // orbit elevation (rad)
  float dist = 1.62f;   // orbit radius (m)
  float tx = 0.42f, ty = 0.f, tz = 0.30f;  // look-at target
};
static ViewState g_cam;
static std::mutex g_cam_mtx;

static const float kCamAz0 = 2.46f, kCamEl0 = 0.417f, kCamDist0 = 1.62f;
static const float kCamT0[3] = {0.42f, 0.f, 0.30f};

void gles_cam_orbit(float d_az, float d_el) {
  std::lock_guard<std::mutex> lk(g_cam_mtx);
  g_cam.az += d_az;
  g_cam.el = clampf(g_cam.el + d_el, 0.10f, 1.40f);
}
void gles_cam_zoom(float dist_factor) {
  std::lock_guard<std::mutex> lk(g_cam_mtx);
  g_cam.dist = clampf(g_cam.dist * dist_factor, 0.55f, 4.0f);
}
void gles_cam_pan(float dx_screen, float dy_screen, int win_w, int win_h) {
  if (win_w < 1 || win_h < 1) return;
  std::lock_guard<std::mutex> lk(g_cam_mtx);
  // camera basis (up = world z)
  const float ce = cosf(g_cam.el), se = sinf(g_cam.el);
  const float ca = cosf(g_cam.az), sa = sinf(g_cam.az);
  float fwd[3] = {-ce * sa, -ce * ca, -se};   // eye -> target
  float fl = sqrtf(fwd[0]*fwd[0] + fwd[1]*fwd[1] + fwd[2]*fwd[2]) + 1e-9f;
  for (float& v : fwd) v /= fl;
  // right = normalize(fwd x up_world)
  float right[3] = {fwd[1] * 1.f - fwd[2] * 0.f, fwd[2] * 0.f - fwd[0] * 1.f, 0.f};
  // (fwd x (0,0,1)) = (fwd.y, -fwd.x, 0)
  right[0] = fwd[1]; right[1] = -fwd[0]; right[2] = 0.f;
  float rl = sqrtf(right[0]*right[0] + right[1]*right[1]) + 1e-9f;
  right[0] /= rl; right[1] /= rl;
  float camup[3] = {right[1]*fwd[2] - right[2]*fwd[1],
                    right[2]*fwd[0] - right[0]*fwd[2],
                    right[0]*fwd[1] - right[1]*fwd[0]};
  // pan speed scales with orbit radius (constant screen-space speed)
  const float k = 0.0022f * g_cam.dist;
  // screen x grows right, screen y grows DOWN (Android): dragging the scene
  // with two fingers moves the look-at opposite to the finger motion
  g_cam.tx += (-right[0] * dx_screen + camup[0] * dy_screen) * k;
  g_cam.ty += (-right[1] * dx_screen + camup[1] * dy_screen) * k;
  g_cam.tz += (                              camup[2] * dy_screen) * k;
  g_cam.tx = clampf(g_cam.tx, -0.4f, 1.3f);
  g_cam.ty = clampf(g_cam.ty, -0.8f, 0.8f);
  g_cam.tz = clampf(g_cam.tz, 0.05f, 0.9f);
}
void gles_cam_reset() {
  std::lock_guard<std::mutex> lk(g_cam_mtx);
  g_cam.az = kCamAz0; g_cam.el = kCamEl0; g_cam.dist = kCamDist0;
  g_cam.tx = kCamT0[0]; g_cam.ty = kCamT0[1]; g_cam.tz = kCamT0[2];
}

// ---------------- v1.3.0 windows + live NN/motion state ----------------
static std::atomic<bool> g_win_open[WIN_COUNT]{false, false};

void gles_toggle_window(int which) {
  if (which < 0 || which >= WIN_COUNT) return;
  const bool make_open = !g_win_open[which].load();
  // only one window open at a time — they share the same screen region
  for (int i = 0; i < WIN_COUNT; ++i) g_win_open[i] = false;
  g_win_open[which] = make_open;
}
bool gles_window_open(int which) {
  return (which >= 0 && which < WIN_COUNT) ? g_win_open[which].load() : false;
}

// ---------------- v1.4.0 UI-redesign state (pressed / toast / pip / confirm) ---
static std::atomic<int> g_pressed_btn{-1};   // bottom-bar button pressed (-1 none)
static std::atomic<int> g_pressed_wb{-1};    // window-button id pressed (-1 none)
static std::atomic<bool> g_pip_big{false};   // robot camera tapped large

void gles_set_pressed(int btn) { g_pressed_btn.store(btn); }
void gles_set_pressed_wb(int id) { g_pressed_wb.store(id); }

static std::mutex g_toast_mtx;
static char g_toast_msg[48] = {0};
static double g_toast_t0 = -1.0;             // start time (s, CLOCK_MONOTONIC)
static double g_toast_dur = 2.2;
static int g_toast_col = TOAST_GRAY;

static double now_sec() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

void gles_toast(const char* msg, int color) {
  std::lock_guard<std::mutex> lk(g_toast_mtx);
  snprintf(g_toast_msg, sizeof g_toast_msg, "%s", msg ? msg : "");
  g_toast_t0 = now_sec();
  g_toast_dur = 2.2;
  g_toast_col = color;
}

// two-step confirm for LOESCH (armed for 3 s, auto-expires)
static std::atomic<long> g_confirm_t0{0};    // 0 = disarmed, else ms epoch
void gles_mot_arm_confirm() { g_confirm_t0.store((long)(now_sec() * 1000.0)); }
void gles_mot_disarm() { g_confirm_t0.store(0); }
bool gles_mot_confirm_armed() {
  const long t0 = g_confirm_t0.load();
  if (t0 <= 0) return false;
  if ((long)(now_sec() * 1000.0) - t0 > 3000) {  // expired
    g_confirm_t0.store(0);
    return false;
  }
  return true;
}

// ---------------- v1.4.0 design system: palette + icons ----------------
// One accent palette for buttons, chips, bars and toasts — everything that
// carries meaning uses the same color, so the screen reads at a glance.
static const float kAcc[8][3] = {
    {0.24f, 0.80f, 0.44f},  // 0 green  (run / ok)
    {0.98f, 0.72f, 0.18f},  // 1 amber  (pause / tight budget)
    {0.94f, 0.32f, 0.28f},  // 2 red    (stop / error / delete)
    {0.32f, 0.60f, 0.97f},  // 3 blue   (new episode / info)
    {0.66f, 0.46f, 0.95f},  // 4 violet (neural net)
    {0.14f, 0.72f, 0.65f},  // 5 teal   (motion manager)
    {0.95f, 0.56f, 0.18f},  // 6 orange (finetune)
    {0.64f, 0.68f, 0.74f}}; // 7 gray   (neutral diag)

// 7x7 pixel icons, one byte per row, bit 6 = leftmost column
enum PcsIcon {
  IC_PLAY = 0, IC_PAUSE, IC_STOP, IC_BOLT, IC_PLUS, IC_CHIP, IC_BARS, IC_COUNT
};
static const uint8_t kIcons[IC_COUNT][7] = {
    {0x40, 0x60, 0x70, 0x78, 0x70, 0x60, 0x40},  // play   (triangle right)
    {0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66},  // pause  (two bars)
    {0x00, 0x1E, 0x1E, 0x1E, 0x1E, 0x1E, 0x00},  // stop   (square)
    {0x06, 0x0C, 0x18, 0x3E, 0x0C, 0x18, 0x30},  // bolt   (finetune)
    {0x08, 0x08, 0x08, 0x3E, 0x08, 0x08, 0x08},  // plus   (new episode)
    {0x08, 0x3E, 0x22, 0x2A, 0x22, 0x3E, 0x08},  // chip   (NN)
    {0x7C, 0x00, 0x3C, 0x00, 0x1C, 0x00, 0x00}}; // bars   (motion clips)

static PcsNnState g_nn;
static PcsMotState g_mot;
// g_nn/g_mot are PODs written by the loop, read by the view — guarded by
// g_ui_mtx together with the diag state (same contention profile)
void gles_set_nn(const PcsNnState& s) {
  std::lock_guard<std::mutex> lk(g_ui_mtx);
  g_nn = s;
}
void gles_set_motion(const PcsMotState& s) {
  std::lock_guard<std::mutex> lk(g_ui_mtx);
  g_mot = s;
}

// view frames that ended with a GL error (HUD "G<n>")
static std::atomic<int> g_gl_errs{0};

// ---------------- tiny math (column-major mat4) ----------------
struct Mat4 {
  float m[16];
  // column-major storage (m[4*col+row]) to match glUniformMatrix4fv(GL_FALSE)
  // and the uMVP * vec4 shader order. v1.0.2 had a row-major formula here,
  // which silently transposed every product -> garbage projections.
  Mat4 operator*(const Mat4& o) const {
    Mat4 r{};
    for (int c = 0; c < 4; ++c)
      for (int row = 0; row < 4; ++row) {
        float acc = 0;
        for (int k = 0; k < 4; ++k) acc += m[4*k+row] * o.m[4*c+k];
        r.m[4*c+row] = acc;
      }
    return r;
  }
  static Mat4 identity() {
    Mat4 r{};
    r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.f;
    return r;
  }
  static Mat4 perspective(float fovy_rad, float aspect, float zn, float zf) {
    Mat4 r{};
    const float f = 1.f / tanf(fovy_rad * 0.5f);
    r.m[0] = f / aspect; r.m[5] = f;
    r.m[10] = (zf + zn) / (zn - zf); r.m[11] = -1.f;
    r.m[14] = 2.f * zf * zn / (zn - zf);
    return r;
  }
  static Mat4 lookAt(const float eye[3], const float ctr[3], const float up[3]) {
    float z[3] = {eye[0]-ctr[0], eye[1]-ctr[1], eye[2]-ctr[2]};
    float zl = sqrtf(z[0]*z[0]+z[1]*z[1]+z[2]*z[2]);
    for (float& v : z) v /= zl;
    float x[3] = {up[1]*z[2]-up[2]*z[1], up[2]*z[0]-up[0]*z[2], up[0]*z[1]-up[1]*z[0]};
    float xl = sqrtf(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
    for (float& v : x) v /= xl;
    float y[3] = {z[1]*x[2]-z[2]*x[1], z[2]*x[0]-z[0]*x[2], z[0]*x[1]-z[1]*x[0]};
    Mat4 r{};
    r.m[0]=x[0]; r.m[4]=x[1]; r.m[8]=x[2];
    r.m[1]=y[0]; r.m[5]=y[1]; r.m[9]=y[2];
    r.m[2]=z[0]; r.m[6]=z[1]; r.m[10]=z[2];
    r.m[12]=-(x[0]*eye[0]+x[1]*eye[1]+x[2]*eye[2]);
    r.m[13]=-(y[0]*eye[0]+y[1]*eye[1]+y[2]*eye[2]);
    r.m[14]=-(z[0]*eye[0]+z[1]*eye[1]+z[2]*eye[2]);
    r.m[15]=1.f;
    return r;
  }
};

static void quat_to_mat(const float q[4], float R[9]) {
  // MuJoCo quat = (w,x,y,z); R is row-major
  const float w=q[0],x=q[1],y=q[2],z=q[3];
  R[0]=1-2*(y*y+z*z); R[1]=2*(x*y-w*z); R[2]=2*(x*z+w*y);
  R[3]=2*(x*y+w*z);   R[4]=1-2*(x*x+z*z); R[5]=2*(y*z-w*x);
  R[6]=2*(x*z-w*y);   R[7]=2*(y*z+w*x);   R[8]=1-2*(x*x+y*y);
}

// ---------------- pose snapshot (physics thread -> render threads) ----------------
// The 100 Hz loop mutates mjData; the view thread must not read it mid-step.
// The loop publishes one stable float snapshot per cycle (gles_publish_poses)
// and both render paths draw from a copied snapshot — no data race.
static std::mutex g_pose_mtx;
static std::vector<float> g_pose_pos;   // 3 floats per body
static std::vector<float> g_pose_quat;  // 4 floats per body (w,x,y,z)

void gles_publish_poses(const mjModel* m, const mjData* d) {
  std::lock_guard<std::mutex> lk(g_pose_mtx);
  g_pose_pos.resize(3 * m->nbody);
  g_pose_quat.resize(4 * m->nbody);
  for (int b = 0; b < m->nbody; ++b) {
    for (int i = 0; i < 3; ++i) g_pose_pos[3*b+i] = (float)d->xpos[3*b+i];
    for (int i = 0; i < 4; ++i) g_pose_quat[4*b+i] = (float)d->xquat[4*b+i];
  }
}

static void copy_poses(std::vector<float>& pos, std::vector<float>& quat) {
  std::lock_guard<std::mutex> lk(g_pose_mtx);
  pos = g_pose_pos;
  quat = g_pose_quat;
}

static void body_transform(int body, const std::vector<float>& pos,
                           const std::vector<float>& quat, float M[16]) {
  const float* p = &pos[3 * body];
  float R[9];
  quat_to_mat(&quat[4 * body], R);
  for (int c = 0; c < 3; ++c) {          // column-major: m[4*col+row]
    for (int r = 0; r < 3; ++r) M[4*c+r] = R[3*r+c];
    M[12+c] = p[c];
  }
  M[3]=M[7]=M[11]=0.f; M[15]=1.f;
}

// HUD data (defined at the bottom, set by the control thread)
CycleStats g_last_stats;
TaskOutput g_last_task;

// ---------------- shaders ----------------
static const char* kVS =
    "uniform mat4 uMVP;\n"
    "uniform mat4 uModel;\n"
    "attribute vec3 aPos;\n"
    "attribute vec3 aNrm;\n"
    "varying vec3 vNrm;\n"
    "varying vec4 vPos;\n"
    "void main(){ vNrm = normalize((uModel * vec4(aNrm,0.0)).xyz);\n"
    "  vPos = uModel * vec4(aPos,1.0);\n"
    "  gl_Position = uMVP * uModel * vec4(aPos,1.0); }";
static const char* kFS =
#ifdef PCS_DESKTOP
    // desktop GLSL has no ES precision qualifiers
    "uniform vec4 uColor;\n"
#else
    "precision mediump float;\n"
    "uniform vec4 uColor;\n"
#endif
    "varying vec3 vNrm;\n"
    "varying vec4 vPos;\n"
    "void main(){ vec3 N = normalize(vNrm);\n"
    "  float d = 0.35;\n"
    "  d += 0.55 * max(0.0, dot(N, normalize(vec3(0.4,0.5,0.8))));\n"
    "  d += 0.25 * max(0.0, dot(N, normalize(vec3(-0.6,0.3,-0.5))));\n"
    "  gl_FragColor = vec4(uColor.rgb * min(d, 1.25), uColor.a); }";

static const char* kHudVS =
    "attribute vec2 aPos;\n"
    "void main(){ gl_Position = vec4(aPos,0.0,1.0); }";
static const char* kHudFS =
#ifdef PCS_DESKTOP
    "uniform vec4 uColor;\n"
#else
    "precision mediump float;\n"
    "uniform vec4 uColor;\n"
#endif
    "void main(){ gl_FragColor = uColor; }";

static GLuint g_prog = 0, g_hud_prog = 0;
static GLint g_uMVP = -1, g_uModel = -1, g_uColor = -1;
static GLint g_aPos = -1, g_aNrm = -1;
static GLint g_huColor = -1, g_huPos = -1;
static int g_win_w = 1, g_win_h = 1;

// ---- v1.4.0 UI program: rounded rects via SDF, per-vertex color ----
// Each quad carries: NDC pos, local coords (-1..1), rect (cx,cy,hw,hh in
// pixels), RGBA color, and (radius_px, border_px). border>0 draws only a
// ring of that width; radius 0 = sharp rect. This gives the whole UI real
// rounded corners + per-element colors in ONE draw call.
static const char* kUiVS =
    "attribute vec2 aPos;\n"
    "attribute vec2 aLocal;\n"
    "attribute vec4 aRect;\n"
    "attribute vec4 aCol;\n"
    "attribute vec2 aMod;\n"
    "varying vec2 vLocal;\n"
    "varying vec4 vRect;\n"
    "varying vec4 vCol;\n"
    "varying vec2 vMod;\n"
    "void main(){ vLocal = aLocal * aRect.zw; vRect = aRect; vCol = aCol;\n"
    "  vMod = aMod; gl_Position = vec4(aPos, 0.0, 1.0); }";
static const char* kUiFS =
#ifdef PCS_DESKTOP
    "varying vec2 vLocal;\n"
    "varying vec4 vRect;\n"
    "varying vec4 vCol;\n"
    "varying vec2 vMod;\n"
#else
    "precision mediump float;\n"
    "varying vec2 vLocal;\n"
    "varying vec4 vRect;\n"
    "varying vec4 vCol;\n"
    "varying vec2 vMod;\n"
#endif
    "void main(){\n"
    "  vec2 h = max(vRect.zw - vec2(vMod.x), vec2(0.0));\n"
    "  vec2 d = abs(vLocal) - h;\n"
    "  float dist = length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0)\n"
    "             - vMod.x;\n"
    "  float a;\n"
    "  if (vMod.y > 0.0) a = 1.0 - smoothstep(vMod.y - 1.0, vMod.y + 1.0,\n"
    "                                         abs(dist));\n"
    "  else a = 1.0 - smoothstep(-1.0, 1.0, dist);\n"
    "  if (a <= 0.004) discard;\n"
    "  gl_FragColor = vec4(vCol.rgb, vCol.a * a); }";

static GLuint g_ui_prog = 0;
static GLint g_uiPos = -1, g_uiLocal = -1, g_uiRect = -1, g_uiCol = -1,
             g_uiMod = -1;
static GLuint g_ui_vbo = 0;

// one rounded rect = 6 vertices x 14 floats (pos2 local2 rect4 col4 mod2)
static void ui_rect(std::vector<float>& v, const float r[4], float cr, float cg,
                    float cb, float ca, float radius, float border, int W,
                    int H) {
  const float hw = 0.5f * (r[2] - r[0]), hh = 0.5f * (r[3] - r[1]);
  if (hw <= 0.5f || hh <= 0.5f) return;
  const float cx = 0.5f * (r[0] + r[2]), cy = 0.5f * (r[1] + r[3]);
  const float xa = 2.f * r[0] / W - 1.f, xb = 2.f * r[2] / W - 1.f;
  const float ya = 1.f - 2.f * r[1] / H, yb = 1.f - 2.f * r[3] / H;
  static const float L[6][2] = {{-1, -1}, {1, -1}, {1, 1},
                                {-1, -1}, {1, 1},  {-1, 1}};
  const float Q[6][2] = {{xa, ya}, {xb, ya}, {xb, yb},
                         {xa, ya}, {xb, yb}, {xa, yb}};
  for (int i = 0; i < 6; ++i) {
    v.push_back(Q[i][0]); v.push_back(Q[i][1]);
    v.push_back(L[i][0]); v.push_back(L[i][1]);
    v.push_back(cx); v.push_back(cy); v.push_back(hw); v.push_back(hh);
    v.push_back(cr); v.push_back(cg); v.push_back(cb); v.push_back(ca);
    v.push_back(radius); v.push_back(border);
  }
}

// draw a batch of UI quads (stream buffer, one call)
static void ui_draw(GLuint vbo, const std::vector<float>& v) {
  if (v.empty() || g_ui_prog == 0) return;
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, v.size() * sizeof(float), v.data(),
               GL_STREAM_DRAW);
  const GLint locs[5] = {g_uiPos, g_uiLocal, g_uiRect, g_uiCol, g_uiMod};
  const GLint sizes[5] = {2, 2, 4, 4, 2};
  uintptr_t off = 0;
  for (int i = 0; i < 5; ++i) {
    glEnableVertexAttribArray((GLuint)locs[i]);
    glVertexAttribPointer((GLuint)locs[i], sizes[i], GL_FLOAT, GL_FALSE, 56,
                          (void*)off);
    off += (uintptr_t)sizes[i] * 4;
  }
  glDrawArrays(GL_TRIANGLES, 0, (GLint)(v.size() / 14));
  for (int i = 0; i < 5; ++i) glDisableVertexAttribArray((GLuint)locs[i]);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// ---------------- v1.4.0 helpers (layout + icons + bars) ----------------
// ONE source of truth per element: draw and hit-test both call the same
// ui_*_rect function, so what you see is exactly what you touch.

static float g_fps = 0.f;            // view-thread FPS (EMA), for the chip
static double g_last_frame = -1.0;

void gles_pip_set_big(bool big) { g_pip_big.store(big); }
bool gles_pip_big() { return g_pip_big.load(); }

// 7x7 icon as colored quads (flat HUD program, accent color batch)
static void icon_quads(std::vector<float>& v, int icon, float x0, float y0,
                       float cs, int W, int H) {
  if (icon < 0 || icon >= IC_COUNT) return;
  for (int r = 0; r < 7; ++r) {
    for (int c = 0; c < 7; ++c) {
      if (!((kIcons[icon][r] >> (6 - c)) & 1)) continue;
      const float px = x0 + c * cs, py = y0 + r * cs;
      const float xa = 2.f * px / W - 1.f, xb = 2.f * (px + cs) / W - 1.f;
      const float ya = 1.f - 2.f * py / H, yb = 1.f - 2.f * (py + cs) / H;
      v.insert(v.end(), {xa, ya, xb, ya, xb, yb});
      v.insert(v.end(), {xa, ya, xb, yb, xa, yb});
    }
  }
}

// ---- layout rects ----
static void ui_status_rect(float r[4]) {
  const int W = g_win_w, H = g_win_h;
  r[0] = 0.014f * W; r[1] = 0.016f * H;
  r[2] = r[0] + 0.302f * W;
  r[3] = r[1] + 0.138f * H;
}
static void ui_chip_rect(int i, float r[4]) {   // 0 FPS, 1 MS, 2 DIAG
  const int W = g_win_w, H = g_win_h;
  static const float kW[3] = {0.068f, 0.078f, 0.058f};
  float x = 0.328f * W;
  for (int k = 0; k < i; ++k) x += kW[k] * W + 0.008f * W;
  r[0] = x; r[1] = 0.016f * H;
  r[2] = x + kW[i] * W; r[3] = r[1] + 0.050f * H;
}
static void ui_pip_rect(float r[4]) {
  const int W = g_win_w, H = g_win_h;
  const float ph = (g_pip_big.load() ? 0.56f : 0.295f) * H;
  const float pw = ph * 4.f / 3.f;              // source is 96x72
  const float x1 = W - 0.014f * W;
  const float y0 = 0.055f * H;
  r[2] = x1; r[0] = x1 - pw;
  r[1] = y0; r[3] = y0 + ph;
}
bool gles_hit_pip(float x, float y) {
  float r[4];
  ui_pip_rect(r);
  return x >= r[0] && x <= r[2] && y >= r[1] && y <= r[3];
}

// horizontal bar: dark track + accent fill (rounded, reads instantly)
static void ui_bar(std::vector<float>& v, float x0, float y0, float w, float h,
                   float frac, int col, int W, int H) {
  float bg[4] = {x0, y0, x0 + w, y0 + h};
  ui_rect(v, bg, 0.f, 0.f, 0.f, 0.38f, h * 0.5f, 0.f, W, H);
  if (frac > 0.004f) {
    float fg[4] = {x0, y0, x0 + w * std::min(1.f, frac), y0 + h};
    const float* c = kAcc[col & 7];
    ui_rect(v, fg, c[0], c[1], c[2], 0.95f, h * 0.5f, 0.f, W, H);
  }
}

// window content: rounded rects (close button, action buttons, live bars)
// rows are laid out identically in ui_render_window_text below
static void ui_render_window_bars(std::vector<float>& ui, int win,
                                  const float wr[4], int W, int H,
                                  double tnow) {
  const float s2 = std::max(3.f, H / 150.f);
  const float lh = 6.5f * s2;
  const float pad = 0.012f * W;
  const int pressed_wb = g_pressed_wb.load();

  // close button (top-right, red square, brightens when pressed)
  {
    float r[4];
    if (ui_window_button_rect(win, 0, r)) {
      float br = (pressed_wb == 0) ? 1.35f : 1.f;
      ui_rect(ui, r, std::min(1.f, 0.94f * br), std::min(1.f, 0.32f * br),
              std::min(1.f, 0.28f * br), 0.95f, 0.008f * W, 0.f, W, H);
    }
  }

  if (win == WIN_NN) {
    // KAM reset button (bottom row, slate + violet border)
    float r[4];
    if (ui_window_button_rect(win, 1, r)) {
      const float br = (pressed_wb == 1) ? 1.35f : 1.f;
      ui_rect(ui, r, std::min(1.f, 0.24f * br), std::min(1.f, 0.26f * br),
              std::min(1.f, 0.34f * br), 0.95f, 0.010f * W, 0.f, W, H);
    }
    // live bars rows 5..8 (labels EMB / FEP / MOE / ETA)
    const float bx0 = wr[0] + 0.082f * W, bx1 = wr[2] - 0.115f * W;
    const float bh = 2.6f * s2;
    for (int i = 0; i < 4; ++i) {
      const float cy = wr[1] + 0.042f * H + (5 + i) * lh + 3.3f * s2;
      float frac = 0.f;
      int col = TOAST_VIOLET;
      if (i == 0) frac = g_nn.emb_norm / 2.0f;
      if (i == 1) { frac = g_nn.free_energy * 8.f; col = TOAST_AMBER; }
      if (i == 2) frac = g_nn.moe_max;
      if (i == 3) { frac = g_nn.lora_eta / 0.5f; col = TOAST_ORANGE; }
      ui_bar(ui, bx0, cy - bh * 0.5f, bx1 - bx0, bh, frac, col, W, H);
    }
  } else {
    // MOT: status blink dot (row 0)
    bool rec = g_mot.rec_left > 0, train = g_mot.train_left > 0;
    if (rec || train) {
      const bool on = sinf((float)(tnow * 6.0)) > -0.2f;
      if (on) {
        const float dr = 1.6f * s2;
        float dot[4] = {wr[0] + pad, wr[1] + 0.042f * H + 1.0f * s2,
                        wr[0] + pad + 2 * dr, wr[1] + 0.042f * H + 1.0f * s2 + 2 * dr};
        const float* c = kAcc[rec ? TOAST_RED : TOAST_ORANGE];
        ui_rect(ui, dot, c[0], c[1], c[2], 0.95f, dr, 0.f, W, H);
      }
    }
    // countdown bar (row 5)
    {
      const float bx0 = wr[0] + pad, bx1 = wr[2] - pad;
      const float cy = wr[1] + 0.042f * H + 5 * lh + 3.3f * s2;
      const float bh = 2.6f * s2;
      float frac = 0.f;
      int col = TOAST_TEAL;
      if (rec) { frac = (float)g_mot.rec_left / 250.f; col = TOAST_RED; }
      else if (train) { frac = (float)g_mot.train_left / 64.f; col = TOAST_ORANGE; }
      ui_bar(ui, bx0, cy - bh * 0.5f, bx1 - bx0, bh, frac, col, W, H);
    }
    // 4 action buttons (bottom row): AUFZ / UMW / TRAIN / LOESCH
    static const int kMotCol[4] = {TOAST_RED, TOAST_BLUE, TOAST_ORANGE,
                                   TOAST_GRAY};
    const bool armed = gles_mot_confirm_armed();
    for (int b = 1; b <= 4; ++b) {
      float r[4];
      if (!ui_window_button_rect(win, b, r)) continue;
      int col = kMotCol[b - 1];
      if (b == 4 && armed) col = TOAST_RED;  // armed confirm = red
      const float br = (pressed_wb == b) ? 1.35f : 1.f;
      const float* c = kAcc[col];
      ui_rect(ui, r, std::min(1.f, c[0] * br), std::min(1.f, c[1] * br),
              std::min(1.f, c[2] * br), 0.95f, 0.010f * W, 0.f, W, H);
    }
  }
}

// window content text (rows match ui_render_window_bars)
static void ui_render_window_text(int win, const float wr[4], int W, int H,
                                  float s, float s2, std::vector<float>& t_white,
                                  std::vector<float>& t_gray,
                                  std::vector<float> t_acc[8], double tnow) {
  (void)tnow;
  const float lh = 6.5f * s2;
  const float pad = 0.014f * W;
  const float tx0 = wr[0] + pad + 0.004f * W;  // clear of the accent strip
  // title
  text_quads(t_white, win == WIN_NN ? "NEURONALES NETZ - 100 HZ"
                                    : "MOTION-MANAGER",
             tx0, wr[1] + 0.010f * H, s2, W, H);
  // close X
  {
    float r[4];
    if (ui_window_button_rect(win, 0, r)) {
      const float tw = text_width("X", s2);
      text_quads(t_white, "X", (r[0] + r[2]) * 0.5f - tw * 0.5f,
                 (r[1] + r[3]) * 0.5f - 2.5f * s2, s2, W, H);
    }
  }
  auto row_y = [&](int i) { return wr[1] + 0.042f * H + i * lh + 0.8f * s2; };

  if (win == WIN_NN) {
    static const char* kArch[3] = {
        "EVENT 96X72 - 576 BINS - LSNN 128",
        "FEP 32D - SOFT-MOE 8 - MLP 24-8-8",
        "LORA R4 + LYAPUNOV-REGELUNG"};
    for (int i = 0; i < 3; ++i) text_quads(t_gray, kArch[i], tx0, row_y(i), s2, W, H);
    char l[48];
    snprintf(l, sizeof l, "EVENTS %u  SPIKES %u", g_nn.events, g_nn.spikes);
    text_quads(t_white, l, tx0, row_y(4), s2, W, H);
    snprintf(l, sizeof l, "V %.4f", g_nn.lora_v);
    const float vw = text_width(l, s2);
    text_quads(t_white, l, wr[2] - pad - vw, row_y(4), s2, W, H);
    // bar rows: label left, value right (bars drawn in _bars)
    char b0[24], b1[24], b2[24], b3[24];
    snprintf(b0, sizeof b0, "%.3f", g_nn.emb_norm);
    snprintf(b1, sizeof b1, "%.4f", g_nn.free_energy);
    snprintf(b2, sizeof b2, "%.2f", g_nn.moe_max);
    snprintf(b3, sizeof b3, "%.3f", g_nn.lora_eta);
    const char* lbls[4] = {"EMB", "FEP", "MOE", "ETA"};
    const char* vals[4] = {b0, b1, b2, b3};
    for (int i = 0; i < 4; ++i) {
      const float ry = row_y(5 + i);
      text_quads(t_gray, lbls[i], tx0, ry, s2, W, H);
      const float vw2 = text_width(vals[i], s2);
      text_quads(t_white, vals[i], wr[2] - pad - vw2, ry, s2, W, H);
    }
    snprintf(l, sizeof l, "PHYS %u  EV %u  SNN %u  MLP %u US", g_nn.t_phys,
             g_nn.t_event, g_nn.t_snn, g_nn.t_mlp);
    text_quads(t_gray, l, tx0, row_y(10), s2, W, H);
  } else {
    // status row (dot drawn in _bars)
    const char* st = "BEREIT";
    int stc = TOAST_TEAL;
    if (g_mot.rec_left > 0) { st = "AUFNAHME..."; stc = TOAST_RED; }
    else if (g_mot.train_left > 0) { st = "TRAINING..."; stc = TOAST_ORANGE; }
    text_quads(t_acc[stc], st, tx0 + 3.4f * s2, row_y(0), s2, W, H);
    char l[48];
    snprintf(l, sizeof l, "CLIPS %d - SAMPLES %d", g_mot.clips, g_mot.samples);
    text_quads(t_white, l, tx0, row_y(1), s2, W, H);
    snprintf(l, sizeof l, "UPDATES %d - NORM %.2f", g_mot.updates,
             g_mot.last_feat_norm);
    text_quads(t_white, l, tx0, row_y(2), s2, W, H);
    snprintf(l, sizeof l, "MSG: %s", g_mot.msg[0] ? g_mot.msg : "-");
    text_quads(t_gray, l, tx0, row_y(4), s2, W, H);
    // button labels
    static const char* kMotLbl[4] = {"AUFZ", "UMW", "TRAIN", "LOESCH"};
    const bool armed = gles_mot_confirm_armed();
    for (int b = 1; b <= 4; ++b) {
      float r[4];
      if (!ui_window_button_rect(win, b, r)) continue;
      const char* lbl = (b == 4 && armed) ? "SICHER?" : kMotLbl[b - 1];
      const float tw = text_width(lbl, s2);
      text_quads(t_white, lbl, (r[0] + r[2]) * 0.5f - tw * 0.5f,
                 (r[1] + r[3]) * 0.5f - 2.5f * s2, s2, W, H);
    }
  }
  // KAM button label (NN only)
  if (win == WIN_NN) {
    float r[4];
    if (ui_window_button_rect(win, 1, r)) {
      const float tw = text_width("KAMERA ZURUECKSETZEN", s2);
      text_quads(t_white, "KAMERA ZURUECKSETZEN",
                 (r[0] + r[2]) * 0.5f - tw * 0.5f,
                 (r[1] + r[3]) * 0.5f - 2.5f * s2, s2, W, H);
    }
  }
}

// cube geometry (unit cube, 24 verts with normals)
static GLuint g_cube_vbo = 0;
// capsule/cylinder geometry (unit cylinder along +x, r=1, len=1 centered)
static GLuint g_cyl_vbo = 0;
static int g_cyl_count = 0;
// unit sphere
static GLuint g_sph_vbo = 0;
static int g_sph_count = 0;

// PiP (robot-camera picture-in-picture) program + texture
static GLuint g_pip_prog = 0;
static GLint g_pip_uTex = -1;
static GLint g_pip_aPos = -1, g_pip_aUV = -1;
static GLuint g_pip_tex = 0;
static int g_pip_w = 0, g_pip_h = 0;
static bool g_pip_has = false;
static std::mutex g_pip_mtx;
static std::vector<uint8_t> g_pip_buf;  // latest perception frame (RGB)

// persistent stream VBOs (one orphaning glBufferData per use instead of
// create/destroy per frame — driver-side churn was a flicker contributor)
static GLuint g_hud_vbo = 0, g_pip_vbo = 0, g_status_vbo = 0;

// draw a batch of vec2 triangles from an orphaned stream buffer
static void stream_draw_2f(GLuint vbo, GLint loc, const std::vector<float>& v) {
  if (v.empty()) return;
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, v.size() * sizeof(float), v.data(), GL_STREAM_DRAW);
  glEnableVertexAttribArray((GLuint)loc);
  glVertexAttribPointer((GLuint)loc, 2, GL_FLOAT, GL_FALSE, 8, (void*)0);
  glDrawArrays(GL_TRIANGLES, 0, (GLint)(v.size() / 2));
  glDisableVertexAttribArray((GLuint)loc);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

static GLuint make_program(const char* vs, const char* fs) {
  GLuint v = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(v, 1, &vs, nullptr); glCompileShader(v);
  GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(f, 1, &fs, nullptr); glCompileShader(f);
  // status checks: a silent compile/link failure turns every draw into a
  // no-op -> the screen shows ONLY the clear color (the v1.0.3 field bug).
  char log[512]; GLsizei llen = 0;
  GLint ok = GL_FALSE;
  glGetShaderiv(v, GL_COMPILE_STATUS, &ok);
  if (ok != GL_TRUE) {
    log[0] = 0; glGetShaderInfoLog(v, sizeof(log), &llen, log);
    LOGW("vertex shader compile failed: %s", log);
  }
  ok = GL_FALSE;
  glGetShaderiv(f, GL_COMPILE_STATUS, &ok);
  if (ok != GL_TRUE) {
    log[0] = 0; glGetShaderInfoLog(f, sizeof(log), &llen, log);
    LOGW("fragment shader compile failed: %s", log);
  }
  GLuint p = glCreateProgram();
  glAttachShader(p, v); glAttachShader(p, f); glLinkProgram(p);
  ok = GL_FALSE;
  glGetProgramiv(p, GL_LINK_STATUS, &ok);
  if (ok != GL_TRUE) {
    log[0] = 0; glGetProgramInfoLog(p, sizeof(log), &llen, log);
    LOGW("program link failed: %s", log);
  }
  glDeleteShader(v); glDeleteShader(f);
  return p;
}

static void push_cube(std::vector<float>& v) {
  // 6 faces, each: normal + 4 corners (strip order a,b,c,d). Expanded to
  // 2 triangles x 3 verts per face = 36 verts, interleaved pos+nrm (stride 24)
  // to match draw_geom's GL_TRIANGLES/36/24 expectations.
  static const float F[6][15] = {
      { 1,0,0,  1,-1,-1, 1,1,-1, 1,1,1, 1,-1,1},
      {-1,0,0, -1,-1,1, -1,1,1, -1,1,-1, -1,-1,-1},
      {0, 1,0,  1,1,-1, -1,1,-1, -1,1,1, 1,1,1},
      {0,-1,0, -1,-1,-1, 1,-1,-1, 1,-1,1, -1,-1,1},
      {0,0, 1,  1,-1,1, -1,-1,1, -1,1,1, 1,1,1},
      {0,0,-1, -1,-1,-1, 1,-1,-1, 1,1,-1, -1,1,-1}};
  for (const auto& f : F) {
    const float* n = f;      // 3 floats normal
    const float* c = f + 3;  // 4 corners
    static const int idx[6] = {0, 1, 2, 0, 2, 3};
    for (int k = 0; k < 6; ++k) {
      const float* p = c + 3 * idx[k];
      v.insert(v.end(), {p[0], p[1], p[2], n[0], n[1], n[2]});
    }
  }
}

static void push_cylinder(std::vector<float>& v) {
  // unit cylinder along +x, r=1, length 1 centered at origin; 12 segments
  // interleaved pos+nrm (stride 24) — v1.0.2 stored bare positions, so the
  // normals attribute read garbage.
  const int N = 12;
  for (int i = 0; i < N; ++i) {
    const float a0 = 2.f * (float)M_PI * i / N, a1 = 2.f * (float)M_PI * (i + 1) / N;
    const float c0 = cosf(a0), s0 = sinf(a0), c1 = cosf(a1), s1 = sinf(a1);
    // side (2 tris), radial normals
    v.insert(v.end(), {c0, s0, 0.f,  c0, s0, 0.f});
    v.insert(v.end(), {c1, s1, 0.f,  c1, s1, 0.f});
    v.insert(v.end(), {c1, s1, 1.f,  c1, s1, 0.f});
    v.insert(v.end(), {c0, s0, 0.f,  c0, s0, 0.f});
    v.insert(v.end(), {c1, s1, 1.f,  c1, s1, 0.f});
    v.insert(v.end(), {c0, s0, 1.f,  c0, s0, 0.f});
    // +x cap (normal +x)
    v.insert(v.end(), {1.f, 0.f, 0.f,  1.f, 0.f, 0.f});
    v.insert(v.end(), {0.5f, c0, s0,  1.f, 0.f, 0.f});
    v.insert(v.end(), {0.5f, c1, s1,  1.f, 0.f, 0.f});
    // -x cap (normal -x)
    v.insert(v.end(), {-1.f, 0.f, 0.f,  -1.f, 0.f, 0.f});
    v.insert(v.end(), {-0.5f, c1, s1,  -1.f, 0.f, 0.f});
    v.insert(v.end(), {-0.5f, c0, s0,  -1.f, 0.f, 0.f});
  }
}

// unit sphere r=1 centered, lat-long grid, interleaved pos+nrm
static void push_sphere(std::vector<float>& v) {
  const int LA = 8, LO = 12;  // latitude bands, longitude segments
  for (int la = 0; la < LA; ++la) {
    const float t0 = (float)M_PI * la / LA, t1 = (float)M_PI * (la + 1) / LA;
    const float c0 = cosf(t0), s0 = sinf(t0), c1 = cosf(t1), s1 = sinf(t1);
    for (int lo = 0; lo < LO; ++lo) {
      const float p0 = 2.f * (float)M_PI * lo / LO;
      const float p1 = 2.f * (float)M_PI * (lo + 1) / LO;
      const float dx0 = cosf(p0), dy0 = sinf(p0), dx1 = cosf(p1), dy1 = sinf(p1);
      const float P[6][6] = {
          {s0 * dx0, s0 * dy0, c0, s0 * dx0, s0 * dy0, c0},
          {s1 * dx0, s1 * dy0, c1, s1 * dx0, s1 * dy0, c1},
          {s1 * dx1, s1 * dy1, c1, s1 * dx1, s1 * dy1, c1},
          {s0 * dx0, s0 * dy0, c0, s0 * dx0, s0 * dy0, c0},
          {s1 * dx1, s1 * dy1, c1, s1 * dx1, s1 * dy1, c1},
          {s0 * dx1, s0 * dy1, c0, s0 * dx1, s0 * dy1, c0}};
      for (auto& q : P) v.insert(v.end(), q, q + 6);
    }
  }
}

// ---------------- real MuJoCo-Menagerie meshes (the Panda) ----------------
// One interleaved VBO built once from mjModel mesh data (visual mesh geoms,
// group 2); collision meshes (group 3) are skipped entirely.
struct MeshRange {
  int first;   // first vertex (6 floats each) inside g_mesh_vbo
  int count;   // vertex count (3 per face)
};
static std::vector<MeshRange> g_mesh_rng;
static GLuint g_mesh_vbo = 0;

static void build_mesh_vbos(const mjModel* m) {
  g_mesh_rng.assign(m->nmesh, MeshRange{0, 0});
  std::vector<float> all;
  all.reserve((size_t)m->nmeshface * 18);
  for (int mi = 0; mi < m->nmesh; ++mi) {
    const int vadr = m->mesh_vertadr[mi];
    const int nadr = m->mesh_normaladr ? m->mesh_normaladr[mi] : vadr;
    const int fadr = m->mesh_faceadr[mi];
    const int nface = m->mesh_facenum[mi];
    MeshRange rng;
    rng.first = (int)all.size() / 6;
    rng.count = nface * 3;
    for (int f = 0; f < nface; ++f) {
      const int* fv = m->mesh_face + 3 * (fadr + f);
      const int* fn = m->mesh_facenormal ? m->mesh_facenormal + 3 * (fadr + f)
                                         : nullptr;
      float tri[3][6];
      for (int k = 0; k < 3; ++k) {
        const float* p = m->mesh_vert + 3 * (vadr + fv[k]);
        tri[k][0] = p[0]; tri[k][1] = p[1]; tri[k][2] = p[2];
        const float* n = nullptr;
        if (fn && m->mesh_normalnum[mi] > 0)
          n = m->mesh_normal + 3 * (nadr + fn[k]);
        if (n) {
          tri[k][3] = n[0]; tri[k][4] = n[1]; tri[k][5] = n[2];
        } else {
          // fallback: face normal from cross product
          const int k1 = (k + 1) % 3, k2 = (k + 2) % 3;
          const float* p1 = m->mesh_vert + 3 * (vadr + fv[k1]);
          const float* p2 = m->mesh_vert + 3 * (vadr + fv[k2]);
          const float e1[3] = {p1[0]-p[0], p1[1]-p[1], p1[2]-p[2]};
          const float e2[3] = {p2[0]-p[0], p2[1]-p[1], p2[2]-p[2]};
          float nrm[3] = {e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2],
                          e1[0]*e2[1]-e1[1]*e2[0]};
          const float len = sqrtf(nrm[0]*nrm[0]+nrm[1]*nrm[1]+nrm[2]*nrm[2])
                            + 1e-12f;
          tri[k][3] = nrm[0]/len; tri[k][4] = nrm[1]/len; tri[k][5] = nrm[2]/len;
        }
      }
      for (auto& corner : tri) all.insert(all.end(), corner, corner + 6);
    }
    g_mesh_rng[mi] = rng;
  }
  if (!all.empty()) {
    glGenBuffers(1, &g_mesh_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, g_mesh_vbo);
    glBufferData(GL_ARRAY_BUFFER, all.size() * sizeof(float), all.data(),
                 GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }
  LOGW("mesh VBOs: nmesh=%ld faces=%ld bytes=%ld", (long)m->nmesh,
       (long)m->nmeshface, (long)all.size() * (long)sizeof(float));
}

// geom-local transform (pos + quat) as a 4x4 column-major matrix
static void geom_local_transform(const mjModel* m, int g, Mat4& L) {
  const mjtNum* gp = m->geom_pos + 3 * g;
  float gq[4] = {1.f, 0.f, 0.f, 0.f};
  if (m->geom_quat) {
    gq[0] = (float)m->geom_quat[4 * g + 0];
    gq[1] = (float)m->geom_quat[4 * g + 1];
    gq[2] = (float)m->geom_quat[4 * g + 2];
    gq[3] = (float)m->geom_quat[4 * g + 3];
  }
  float R[9];
  quat_to_mat(gq, R);
  for (int c = 0; c < 3; ++c) {
    for (int r = 0; r < 3; ++r) L.m[4 * c + r] = R[3 * r + c];
    L.m[12 + c] = (float)gp[c];
  }
  L.m[3] = L.m[7] = L.m[11] = 0.f;
  L.m[15] = 1.f;
}

void gles_init(SimGlue& glue, int win_w, int win_h) {
  g_win_w = win_w; g_win_h = win_h;
  g_prog = make_program(kVS, kFS);
  g_hud_prog = make_program(kHudVS, kHudFS);
  g_uMVP = glGetUniformLocation(g_prog, "uMVP");
  g_uModel = glGetUniformLocation(g_prog, "uModel");
  g_uColor = glGetUniformLocation(g_prog, "uColor");
  g_aPos = glGetAttribLocation(g_prog, "aPos");
  g_aNrm = glGetAttribLocation(g_prog, "aNrm");
  g_huColor = glGetUniformLocation(g_hud_prog, "uColor");
  g_huPos = glGetAttribLocation(g_hud_prog, "aPos");

  // v1.4.0: rounded-rect UI program (panels, buttons, bars, toasts)
  g_ui_prog = make_program(kUiVS, kUiFS);
  g_uiPos = glGetAttribLocation(g_ui_prog, "aPos");
  g_uiLocal = glGetAttribLocation(g_ui_prog, "aLocal");
  g_uiRect = glGetAttribLocation(g_ui_prog, "aRect");
  g_uiCol = glGetAttribLocation(g_ui_prog, "aCol");
  g_uiMod = glGetAttribLocation(g_ui_prog, "aMod");
  if (g_ui_vbo == 0) glGenBuffers(1, &g_ui_vbo);

  std::vector<float> v;
  push_cube(v);
  glGenBuffers(1, &g_cube_vbo);
  glBindBuffer(GL_ARRAY_BUFFER, g_cube_vbo);
  glBufferData(GL_ARRAY_BUFFER, v.size() * 4, v.data(), GL_STATIC_DRAW);
  v.clear();
  push_cylinder(v);
  g_cyl_count = (int)v.size() / 6;  // 6 floats (pos+nrm) per vertex
  glGenBuffers(1, &g_cyl_vbo);
  glBindBuffer(GL_ARRAY_BUFFER, g_cyl_vbo);
  glBufferData(GL_ARRAY_BUFFER, v.size() * 4, v.data(), GL_STATIC_DRAW);
  v.clear();
  push_sphere(v);
  g_sph_count = (int)v.size() / 6;
  glGenBuffers(1, &g_sph_vbo);
  glBindBuffer(GL_ARRAY_BUFFER, g_sph_vbo);
  glBufferData(GL_ARRAY_BUFFER, v.size() * 4, v.data(), GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glDisable(GL_DEPTH_TEST);

  // real Panda: interleaved mesh VBO from the compiled model
  build_mesh_vbos(glue.model());

  // PiP textured-quad program (shares the HUD vertex layout style)
  static const char* kPipVS =
      "attribute vec2 aPos;\n"
      "attribute vec2 aUV;\n"
      "varying vec2 vUV;\n"
      "void main(){ vUV = aUV; gl_Position = vec4(aPos, 0.0, 1.0); }";
  static const char* kPipFS =
#ifdef PCS_DESKTOP
      "uniform sampler2D uTex;\n"
#else
      "precision mediump float;\n"
      "uniform sampler2D uTex;\n"
#endif
      "varying vec2 vUV;\n"
      "void main(){ gl_FragColor = texture2D(uTex, vUV); }";
  g_pip_prog = make_program(kPipVS, kPipFS);
  g_pip_uTex = glGetUniformLocation(g_pip_prog, "uTex");
  g_pip_aPos = glGetAttribLocation(g_pip_prog, "aPos");
  g_pip_aUV = glGetAttribLocation(g_pip_prog, "aUV");
  glGenTextures(1, &g_pip_tex);
  glBindTexture(GL_TEXTURE_2D, g_pip_tex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, kEvW, kEvH, 0, GL_RGB,
               GL_UNSIGNED_BYTE, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glBindTexture(GL_TEXTURE_2D, 0);

  glGenBuffers(1, &g_hud_vbo);
  glGenBuffers(1, &g_pip_vbo);

  LOGW("gles_init done %dx%d", win_w, win_h);
}

// draw one geom (primitive or mesh) with color; poses come from the snapshot.
// Includes the geom-local pos/quat transform (the Menagerie Panda geoms are
// offset inside their bodies — v1.1.0 ignored this, primitives happened to
// sit at their body origins).
static void draw_geom(const mjModel* m, const std::vector<float>& pos,
                      const std::vector<float>& quat, int g, float alpha) {
  const int body = m->geom_bodyid[g];
  float M[16];
  body_transform(body, pos, quat, M);
  Mat4 Mb, L;
  std::memcpy(Mb.m, M, sizeof(Mb.m));
  geom_local_transform(m, g, L);
  const mjtNum* size = m->geom_size + 3 * g;
  const int type = m->geom_type[g];
  // color: material first (Menagerie uses materials), else geom_rgba
  float col[4] = {1.f, 1.f, 1.f, 1.f};
  const int matid = m->geom_matid[g];
  if (matid >= 0) {
    const float* mr = m->mat_rgba + 4 * matid;
    col[0] = mr[0]; col[1] = mr[1]; col[2] = mr[2]; col[3] = mr[3];
  } else {
    const float* rgba = m->geom_rgba + 4 * g;
    col[0] = rgba[0]; col[1] = rgba[1]; col[2] = rgba[2]; col[3] = rgba[3];
  }
  if (alpha >= 0.f) col[3] = alpha;

  GLuint vbo = 0;
  int count = 0;
  GLint voff = 0;
  Mat4 model = Mb * L;
  if (type == mjGEOM_MESH) {
    const int dataid = m->geom_dataid[g];
    if (dataid < 0 || g_mesh_vbo == 0 || dataid >= (int)g_mesh_rng.size()) return;
    const MeshRange& rng = g_mesh_rng[dataid];
    if (rng.count <= 0) return;
    vbo = g_mesh_vbo;
    count = rng.count;
    voff = rng.first;
  } else if (type == mjGEOM_CAPSULE || type == mjGEOM_CYLINDER) {
    // MuJoCo capsule: along local z, half-length size[1]; our VBO: along x
    Mat4 scale = Mat4::identity();
    scale.m[0] = (float)size[0]; scale.m[5] = (float)size[0]; scale.m[10] = (float)size[1];
    Mat4 rot = Mat4::identity();   // align z-axis with x-axis
    rot.m[0] = 0; rot.m[2] = -1; rot.m[8] = 1; rot.m[10] = 0;
    model = model * rot * scale;
    vbo = g_cyl_vbo; count = g_cyl_count;
  } else if (type == mjGEOM_BOX) {
    Mat4 scale = Mat4::identity();
    scale.m[0] = (float)size[0]; scale.m[5] = (float)size[1]; scale.m[10] = (float)size[2];
    model = model * scale;
    vbo = g_cube_vbo; count = 36;
  } else if (type == mjGEOM_SPHERE) {
    Mat4 scale = Mat4::identity();
    scale.m[0] = scale.m[5] = scale.m[10] = (float)size[0];
    model = model * scale;
    vbo = g_sph_vbo; count = g_sph_count;
  } else if (type == mjGEOM_PLANE) {
    Mat4 scale = Mat4::identity();
    scale.m[0] = (float)size[0] > 0.01f ? (float)size[0] : 2.5f;
    scale.m[5] = (float)size[1] > 0.01f ? (float)size[1] : 2.5f;
    scale.m[10] = 0.01f;  // thin slab — MuJoCo planes have no thickness
    model = model * scale;
    vbo = g_cube_vbo; count = 36;
  } else {
    return;
  }
  glUniformMatrix4fv(g_uModel, 1, GL_FALSE, model.m);
  glUniform4fv(g_uColor, 1, col);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glEnableVertexAttribArray((GLuint)g_aPos);
  glEnableVertexAttribArray((GLuint)g_aNrm);
  glVertexAttribPointer((GLuint)g_aPos, 3, GL_FLOAT, GL_FALSE, 24,
                        (void*)(intptr_t)(voff * 24));
  glVertexAttribPointer((GLuint)g_aNrm, 3, GL_FLOAT, GL_FALSE, 24,
                        (void*)(intptr_t)(voff * 24 + 12));
  glDrawArrays(GL_TRIANGLES, 0, count);
  glDisableVertexAttribArray((GLuint)g_aPos);
  glDisableVertexAttribArray((GLuint)g_aNrm);
}

void gles_render_view(SimGlue& glue, Controller& ctrl) {
  const mjModel* m = glue.model();
  // frame-stable snapshot of the window size (the activity thread may resize
  // g_win_w/h at any moment — mixing sizes within one frame jitters the image)
  const int W = g_win_w, H = g_win_h;
  // v1.4.0: FPS estimate for the perf chip (EMA of frame deltas)
  {
    const double t = now_sec();
    if (g_last_frame > 0.0) {
      const double dt = t - g_last_frame;
      if (dt > 0.0005 && dt < 0.5)
        g_fps = g_fps <= 0.5f ? (float)(1.0 / dt)
                              : g_fps * 0.9f + (float)(1.0 / dt) * 0.1f;
    }
    g_last_frame = t;
  }
  std::vector<float> pos, quat;
  copy_poses(pos, quat);
  const bool have_poses = pos.size() >= (size_t)3 * m->nbody;
  // NEVER early-return: an invisible failure mode (grey screen) is worse
  // than a degraded frame. Without poses we still clear + show the HUD.
  glViewport(0, 0, W, H);
  glClearColor(0.09f, 0.10f, 0.12f, 1.f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);

  if (have_poses) {
    glUseProgram(g_prog);
    // v1.3.0 orbit camera: az/el/dist/target steered by touch gestures
    // (1 finger = orbit, 2 fingers = pinch zoom + pan). Default pose equals
    // the fixed v1.2 viewpoint.
    ViewState cam;
    { std::lock_guard<std::mutex> lk(g_cam_mtx); cam = g_cam; }
    const float ce = cosf(cam.el), se = sinf(cam.el);
    const float ca = cosf(cam.az), sa = sinf(cam.az);
    float eye[3] = {cam.tx + cam.dist * ce * sa, cam.ty + cam.dist * ce * ca,
                    cam.tz + cam.dist * se};
    float ctr[3] = {cam.tx, cam.ty, cam.tz};
    float up[3] = {0, 0, 1};
    Mat4 view = Mat4::lookAt(eye, ctr, up);
    Mat4 proj = Mat4::perspective(50.f * (float)M_PI / 180.f,
                                  (float)W / H, 0.05f, 10.f);
    Mat4 vp = proj * view;
    glUniformMatrix4fv(g_uMVP, 1, GL_FALSE, vp.m);

    // pass 1: opaque geoms (visual meshes group 2 + primitives),
    // pass 2: translucent (zone plates) with blending, depth-write off.
    // group 3 (collision meshes) is never drawn — it would z-fight the
    // visual meshes of the Panda.
    for (int g = 0; g < m->ngeom; ++g) {
      if (m->geom_group[g] == 3) continue;
      float col_a = 1.f;
      const int matid = m->geom_matid[g];
      if (matid >= 0) col_a = (float)m->mat_rgba[4 * matid + 3];
      else col_a = (float)m->geom_rgba[4 * g + 3];
      if (col_a < 0.05f) continue;  // fully invisible
      if (col_a < 0.95f) continue;  // translucent -> pass 2
      draw_geom(m, pos, quat, g, 1.f);
    }
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);
    for (int g = 0; g < m->ngeom; ++g) {
      if (m->geom_group[g] == 3) continue;
      float col_a = 1.f;
      const int matid = m->geom_matid[g];
      if (matid >= 0) col_a = (float)m->mat_rgba[4 * matid + 3];
      else col_a = (float)m->geom_rgba[4 * g + 3];
      if (col_a < 0.05f || col_a >= 0.95f) continue;
      draw_geom(m, pos, quat, g, -1.f);
    }
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
  }
  glDisable(GL_DEPTH_TEST);
  gles_draw_pip();
  gles_render_hud(g_last_stats, g_last_task);
  if (glGetError() != GL_NO_ERROR) ++g_gl_errs;  // HUD "G<n>" diagnostic
}

void gles_render_hud(const CycleStats& st, const TaskOutput& task) {
  (void)task;
  if (g_huPos < 0 || g_hud_prog == 0 || g_ui_prog == 0) return;
  const int W = g_win_w, H = g_win_h;
  const float s = std::max(3.f, H / 95.f);    // main text scale
  const float s2 = std::max(3.f, H / 150.f);  // small text scale
  const double tnow = now_sec();

  // one-time camera-gesture hint (renderer-side, no loop changes needed)
  static bool s_hint = false;
  if (!s_hint) {
    s_hint = true;
    gles_toast("1 FINGER DREHEN - 2 FINGER ZOOM", TOAST_BLUE);
  }

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // ================= pass 1: rounded-rect UI (one draw call) =================
  std::vector<float> ui;

  // ---- cycle-budget strip along the top edge ----
  {
    const float frac = std::min(1.0f, (float)st.t_total / 10000.f);
    float r[4] = {0.f, 0.f, W * frac, std::max(4.f, 0.005f * H)};
    const float* c = st.t_total > 10000 ? kAcc[TOAST_RED]
                    : st.t_total > 8000 ? kAcc[TOAST_AMBER]
                                        : kAcc[TOAST_GREEN];
    ui_rect(ui, r, c[0], c[1], c[2], 0.95f, 2.f, 0.f, W, H);
  }

  // ---- status card (top left) ----
  float card[4];
  ui_status_rect(card);
  ui_rect(ui, card, 0.095f, 0.105f, 0.135f, 0.82f, 0.012f * W, 0.f, W, H);
  ui_rect(ui, card, 1.f, 1.f, 1.f, 0.07f, 0.012f * W, 1.2f, W, H);  // hairline
  {
    // row 1: state dot + state + phase (right)
    const bool blink = (sinf((float)(tnow * 8.0)) > 0.f);
    int scol = TOAST_GREEN;
    const char* sname = "AKTIV";
    if (g_diag.paused) { scol = TOAST_AMBER; sname = "PAUSIERT"; }
    else if (g_diag.halted) { scol = TOAST_RED; sname = "HALT"; }
    else if (g_diag.finetuning) { scol = TOAST_ORANGE; sname = "FINETUNE"; }
    const float dotR = 0.0045f * W;
    float dot[4] = {card[0] + 0.012f * W, card[1] + 0.020f * H,
                    card[0] + 0.012f * W + 2 * dotR, card[1] + 0.020f * H + 2 * dotR};
    const float* dc = kAcc[scol];
    const float dalpha = g_diag.finetuning ? (blink ? 1.f : 0.35f) : 1.f;
    ui_rect(ui, dot, dc[0], dc[1], dc[2], dalpha, dotR, 0.f, W, H);
    // row 2: progress bar (right of the labels) + counts
    const float bx0 = card[0] + 0.185f * W, bx1 = card[2] - 0.012f * W;
    const float frac = g_diag.total > 0
                           ? (float)g_diag.sorted / (float)g_diag.total : 0.f;
    ui_bar(ui, bx0, card[1] + 0.064f * H, bx1 - bx0, 0.020f * H, frac,
           TOAST_GREEN, W, H);
  }

  // ---- perf chips (right of the card) ----
  for (int i = 0; i < 3; ++i) {
    float r[4];
    ui_chip_rect(i, r);
    ui_rect(ui, r, 0.095f, 0.105f, 0.135f, 0.72f, 0.45f * (r[3] - r[1]), 0.f,
            W, H);
  }

  // ---- action bar (bottom): color-coded buttons ----
  static const int kBtnCol[BTN_COUNT] = {TOAST_GREEN, TOAST_RED, TOAST_ORANGE,
                                         TOAST_BLUE, TOAST_VIOLET, TOAST_TEAL};
  static const int kBtnIcon[BTN_COUNT] = {IC_PLAY, IC_STOP, IC_BOLT, IC_PLUS,
                                          IC_CHIP, IC_BARS};
  const int pressed = g_pressed_btn.load();
  for (int i = 0; i < BTN_COUNT; ++i) {
    float r[4];
    ui_button_rect(i, r);
    int col = kBtnCol[i];
    if (i == BTN_START) col = g_diag.paused ? TOAST_GREEN : TOAST_AMBER;
    float br = 1.f;
    if (i == BTN_FINE && g_diag.finetuning) br = 1.f + 0.35f * (sinf((float)(tnow * 9.0)) * 0.5f + 0.5f);
    if (i == pressed) {  // pressed: shrink + brighten
      const float cx = 0.5f * (r[0] + r[2]), cy = 0.5f * (r[1] + r[3]);
      r[0] = cx + (r[0] - cx) * 0.94f; r[2] = cx + (r[2] - cx) * 0.94f;
      r[1] = cy + (r[1] - cy) * 0.94f; r[3] = cy + (r[3] - cy) * 0.94f;
      br *= 1.35f;
    }
    if ((i == BTN_NN && gles_window_open(WIN_NN)) ||
        (i == BTN_MOT && gles_window_open(WIN_MOT)))
      br *= 1.2f;  // toggle buttons look "in" while their window is open
    const float* c = kAcc[col];
    ui_rect(ui, r, std::min(1.f, c[0] * br), std::min(1.f, c[1] * br),
            std::min(1.f, c[2] * br), 0.92f, 0.012f * W, 0.f, W, H);
    if ((i == BTN_NN && gles_window_open(WIN_NN)) ||
        (i == BTN_MOT && gles_window_open(WIN_MOT))) {
      float rr[4];
      ui_button_rect(i, rr);
      float dot[4] = {rr[2] - 0.009f * W, rr[1] + 0.008f * H, rr[2] - 0.004f * W,
                      rr[1] + 0.016f * H};
      ui_rect(ui, dot, 1.f, 1.f, 1.f, 0.95f, 0.0025f * W, 0.f, W, H);
    }
  }

  // ================= windows (panels + bars, still pass 1) =================
  const bool open_nn = gles_window_open(WIN_NN);
  const bool open_mot = gles_window_open(WIN_MOT);
  float wr[4] = {0};
  int wopen = -1;
  if (open_nn) { ui_window_rect(WIN_NN, wr); wopen = WIN_NN; }
  else if (open_mot) { ui_window_rect(WIN_MOT, wr); wopen = WIN_MOT; }
  if (wopen >= 0) {
    ui_rect(ui, wr, 0.075f, 0.082f, 0.105f, 0.90f, 0.012f * W, 0.f, W, H);
    ui_rect(ui, wr, 1.f, 1.f, 1.f, 0.08f, 0.012f * W, 1.2f, W, H);
    const float* ac = kAcc[wopen == WIN_NN ? TOAST_VIOLET : TOAST_TEAL];
    float strip[4] = {wr[0], wr[1], wr[0] + 0.006f * W, wr[3]};
    ui_rect(ui, strip, ac[0], ac[1], ac[2], 0.95f, 0.f, 0.f, W, H);
    // content bars (NN live values / MOT countdowns) + close button + wbtns
    ui_render_window_bars(ui, wopen, wr, W, H, tnow);
  }

  // ================= toast =================
  char tmsg[48];
  double tage = -1.0;
  int tcol = TOAST_GRAY;
  {
    std::lock_guard<std::mutex> lk(g_toast_mtx);
    if (g_toast_t0 > 0) tage = tnow - g_toast_t0;
    snprintf(tmsg, sizeof tmsg, "%s", g_toast_msg);
    tcol = g_toast_col;
  }
  if (tage >= 0 && tage < g_toast_dur && tmsg[0]) {
    const float fade = tage < 0.15f ? (float)(tage / 0.15)
                       : tage > g_toast_dur - 0.4f
                           ? (float)((g_toast_dur - tage) / 0.4) : 1.f;
    const float tw = text_width(tmsg, s);
    const float bh = 0.056f * H;
    const float pw = tw + 4.5f * s;
    const float px = 0.5f * (W - pw);
    const float py = H - 0.115f * H - 0.020f * H - bh - 0.014f * H;
    float r[4] = {px, py, px + pw, py + bh};
    const float* c = kAcc[tcol & 7];
    ui_rect(ui, r, 0.08f, 0.085f, 0.11f, 0.92f * fade, 0.5f * bh, 0.f, W, H);
    float dot[4] = {px + 0.9f * s, py + 0.5f * (bh - 2 * s), px + 0.9f * s + 2 * s,
                    py + 0.5f * (bh + 2 * s)};
    ui_rect(ui, dot, c[0], c[1], c[2], fade, s, 0.f, W, H);
  }

  // pip label chip background (must be part of the pass-1 UI batch)
  float pip_chip[4] = {0};
  {
    float pr[4];
    ui_pip_rect(pr);
    const char* lbl = gles_pip_big() ? "KAMERA - TIPPEN ZUM KLEIN" : "KAMERA";
    const float tw = text_width(lbl, s2);
    pip_chip[0] = pr[0] + 0.006f * W;
    pip_chip[1] = pr[3] - 0.036f * H;
    pip_chip[2] = pip_chip[0] + tw + 2.2f * s2;
    pip_chip[3] = pr[3] - 0.008f * H;
    ui_rect(ui, pip_chip, 0.06f, 0.065f, 0.085f, 0.85f, 0.006f * W, 0.f, W, H);
  }

  glUseProgram(g_ui_prog);
  ui_draw(g_ui_vbo, ui);
  glUseProgram(g_hud_prog);

  // ================= pass 2: text + icons (flat, color batches) =============
  std::vector<float> t_white, t_gray;
  std::vector<float> t_acc[8];

  // ---- button icons + labels ----
  static const char* kLabels[BTN_COUNT] = {"START", "STOP", "FINE", "NEU",
                                           "NN", "MOT"};
  {
    const float cs = std::max(3.f, H / 190.f);
    for (int i = 0; i < BTN_COUNT; ++i) {
      float r[4];
      ui_button_rect(i, r);
      const int col = (i == BTN_START) ? (g_diag.paused ? TOAST_GREEN
                                                        : TOAST_AMBER)
                                       : kBtnCol[i];
      const int icon = (i == BTN_START) ? (g_diag.paused ? IC_PLAY : IC_PAUSE)
                                        : kBtnIcon[i];
      const char* lbl = kLabels[i];
      if (i == BTN_START) lbl = g_diag.paused ? "START" : "PAUSE";
      const float ih = 7.f * cs, th = 5.f * s;
      const float iy = r[1] + 0.10f * (r[3] - r[1]);
      const float iw = 7.f * cs;
      icon_quads(t_acc[col], icon, (r[0] + r[2]) * 0.5f - iw * 0.5f, iy, cs,
                 W, H);
      const float tw = text_width(lbl, s);
      text_quads(t_acc[col], lbl, (r[0] + r[2]) * 0.5f - tw * 0.5f,
                 iy + ih + 0.14f * (r[3] - r[1]), s, W, H);
    }
  }

  // ---- status card text ----
  {
    const float pad = 0.012f * W;
    int scol = TOAST_GREEN;
    const char* sname = "AKTIV";
    if (g_diag.paused) { scol = TOAST_AMBER; sname = "PAUSIERT"; }
    else if (g_diag.halted) { scol = TOAST_RED; sname = "HALT"; }
    else if (g_diag.finetuning) { scol = TOAST_ORANGE; sname = "FINETUNE"; }
    text_quads(t_acc[scol], sname, card[0] + pad + 0.013f * W,
               card[1] + 0.016f * H, s, W, H);
    char lbuf[48];
    snprintf(lbuf, sizeof lbuf, "%s", kPhaseName[task.phase & 7]);
    const float pw = text_width(lbuf, s2);
    text_quads(t_gray, lbuf, card[2] - pad - pw, card[1] + 0.021f * H, s2, W, H);
    snprintf(lbuf, sizeof lbuf, "SORTIERT %d/%d", g_diag.sorted, g_diag.total);
    text_quads(t_white, lbuf, card[0] + pad, card[1] + 0.058f * H, s2, W, H);
    snprintf(lbuf, sizeof lbuf, "GESTAPELT %d", g_diag.stacked);
    text_quads(t_white, lbuf, card[0] + pad, card[1] + 0.098f * H, s2, W, H);
    snprintf(lbuf, sizeof lbuf, "ZYK %ld", g_diag.cycles);
    const float cw = text_width(lbuf, s2);
    text_quads(t_gray, lbuf, card[2] - pad - cw, card[1] + 0.098f * H, s2, W, H);
  }

  // ---- chips text ----
  for (int i = 0; i < 3; ++i) {
    float r[4];
    ui_chip_rect(i, r);
    char cb[24];
    int col = TOAST_GRAY;
    if (i == 0) {
      snprintf(cb, sizeof cb, "%.0f FPS", g_fps > 0.5f ? g_fps : 0.f);
      col = TOAST_GREEN;
    } else if (i == 1) {
      snprintf(cb, sizeof cb, "%.1f MS", (float)st.t_total / 1000.f);
      col = st.t_total > 10000 ? TOAST_RED : st.t_total > 8000 ? TOAST_AMBER
                                                               : TOAST_GREEN;
    } else {
      snprintf(cb, sizeof cb, "B%d G%d", g_diag.bind, g_diag.gl_errs);
      col = TOAST_GRAY;
    }
    const float tw = text_width(cb, s2);
    text_quads(t_acc[col], cb, (r[0] + r[2]) * 0.5f - tw * 0.5f,
               (r[1] + r[3]) * 0.5f - 2.5f * s2, s2, W, H);
  }

  // ---- window text (titles + content) ----
  if (wopen >= 0) ui_render_window_text(wopen, wr, W, H, s, s2, t_white, t_gray,
                                        t_acc, tnow);

  // ---- pip label chip text (rect drawn in pass 1) ----
  {
    const char* lbl = gles_pip_big() ? "KAMERA - TIPPEN ZUM KLEIN" : "KAMERA";
    text_quads(t_gray, lbl, pip_chip[0] + 1.1f * s2, pip_chip[1] + 0.9f * s2,
               s2, W, H);
  }

  // ---- toast text ----
  if (tage >= 0 && tage < g_toast_dur && tmsg[0]) {
    const float fade = tage < 0.15f ? (float)(tage / 0.15)
                       : tage > g_toast_dur - 0.4f
                           ? (float)((g_toast_dur - tage) / 0.4) : 1.f;
    const float tw = text_width(tmsg, s);
    const float bh = 0.056f * H;
    const float pw = tw + 4.5f * s;
    const float px = 0.5f * (W - pw);
    const float py = H - 0.115f * H - 0.020f * H - bh - 0.014f * H;
    text_quads(t_white, tmsg, px + 2.6f * s, py + 0.5f * (bh - 5.f * s), s, W,
               H);
    (void)fade;
  }

  // draw the batches
  glUniform4f(g_huColor, 0.60f, 0.64f, 0.70f, 1.f);
  stream_draw_2f(g_hud_vbo, g_huPos, t_gray);
  for (int c = 0; c < 8; ++c) {
    if (t_acc[c].empty()) continue;
    glUniform4f(g_huColor, kAcc[c][0], kAcc[c][1], kAcc[c][2], 1.f);
    stream_draw_2f(g_hud_vbo, g_huPos, t_acc[c]);
  }
  glUniform4f(g_huColor, 1.f, 1.f, 1.f, 1.f);
  stream_draw_2f(g_hud_vbo, g_huPos, t_white);
  glDisable(GL_BLEND);
}

// ---------------- v1.3.0 overlay windows (NN info / motion manager) ----------------
// pixel coords are TOP-left origin (same convention as touches + text)

static void ui_window_rect(int which, float r[4]) {
  const int W = g_win_w, H = g_win_h;
  const float s2 = std::max(3.f, H / 150.f);
  const float lh = 6.5f * s2;
  const int lines = (which == WIN_NN) ? 12 : 7;
  r[0] = 0.014f * W;
  r[1] = 0.170f * H;                       // below the status card
  r[2] = r[0] + 0.46f * W;
  r[3] = r[1] + 0.042f * H + lines * lh
       + (which == WIN_MOT ? 0.092f * H : 0.034f * H);
  const float maxb = H - 0.152f * H;       // keep above the action bar
  if (r[3] > maxb) r[3] = maxb;
}

// window buttons: btn 0 = close (X, top-right); NN btn 1 = KAM (bottom row);
// MOT btns 1..4 = AUFZ/UMW/TRAIN/LOESCH (bottom row). ids match v1.3.0.
static bool ui_window_button_rect(int win, int btn, float r[4]) {
  float wr[4];
  ui_window_rect(win, wr);
  const int W = g_win_w, H = g_win_h;
  if (btn == 0) {                          // close: square top-right
    const float bs = 0.040f * H;
    r[2] = wr[2] - 0.010f * W;
    r[0] = r[2] - bs;
    r[1] = wr[1] + 0.011f * H;
    r[3] = r[1] + bs;
    return true;
  }
  const int n = (win == WIN_NN) ? 1 : 4;   // bottom-row buttons
  const int k = btn - 1;
  if (k < 0 || k >= n) return false;
  const float m = 0.012f * W, gap = 0.008f * W, bh = 0.056f * H;
  const float bw = ((wr[2] - wr[0]) - 2 * m - (n - 1) * gap) / (float)n;
  r[0] = wr[0] + m + k * (bw + gap);
  r[2] = r[0] + bw;
  r[3] = wr[3] - 0.012f * H;
  r[1] = r[3] - bh;
  return true;
}

int gles_hit_window_button(float x, float y) {
  for (int w = 0; w < WIN_COUNT; ++w) {
    if (!gles_window_open(w)) continue;
    float wr[4];
    ui_window_rect(w, wr);
    const int n = (w == WIN_NN) ? 2 : 5;
    for (int b = 0; b < n; ++b) {
      float r[4];
      if (!ui_window_button_rect(w, b, r)) continue;
      if (x >= r[0] && x <= r[2] && y >= r[1] && y <= r[3]) {
        if (w == WIN_NN) return b == 0 ? 1 : 2;   // 1 close, 2 cam reset
        return 3 + b;  // 3 close, 4 AUFZ, 5 UMW, 6 TRAIN, 7 LOESCH
      }
    }
    // inside the window body (not a button): consume so the scene does not
    // orbit under the finger
    if (x >= wr[0] && x <= wr[2] && y >= wr[1] && y <= wr[3]) return 8;
  }
  return 0;
}

// ---------------- picture-in-picture robot camera ----------------
int gles_gl_errs() { return g_gl_errs.load(); }

void gles_push_pip(const uint8_t* rgb, int w, int h) {
  std::lock_guard<std::mutex> lk(g_pip_mtx);
  g_pip_buf.assign(rgb, rgb + (size_t)w * h * 3);
  g_pip_w = w;
  g_pip_h = h;
  g_pip_has = true;
}

static void gles_draw_pip() {
  if (!g_pip_prog || !g_pip_has || g_pip_w <= 0) return;
  std::vector<uint8_t> frame;
  {
    std::lock_guard<std::mutex> lk(g_pip_mtx);
    if (!g_pip_has) return;
    frame = g_pip_buf;  // 20 KB copy, keeps the GL upload off the loop thread
  }
  const int W = g_win_w, H = g_win_h;
  // v1.4.0: rect from the shared ui_pip_rect (draw + hit-test + label chip
  // all agree; tapping toggles the large view)
  float pr[4];
  ui_pip_rect(pr);
  const float x0 = pr[0], y0 = pr[1];
  const float pw = pr[2] - pr[0], ph = pr[3] - pr[1];
  if (g_pip_tex == 0 || g_pip_w != kEvW || g_pip_h != kEvH) return;

  // v1.3.0 FLICKER FIX (field report: "Kamera-Bild wird kurz Vollbild").
  // The old path drew an NDC quad sized from window pixels — any transient
  // state inconsistency (resize window racing a frame, driver-side buffer
  // churn) turned the quad into a fullscreen camera flash. The image can now
  // PHYSICALLY not leave its rectangle: it is drawn as a fullscreen quad
  // THROUGH a viewport+scissor locked to the PiP rect. Nothing outside the
  // rect is ever touched, whatever happens to the vertex data.
  const int ix0 = std::max(0, (int)x0), iy0 = std::max(0, (int)y0);
  const int ix1 = std::min(W, (int)(x0 + pw + 3.f) + 3);
  const int iy1 = std::min(H, (int)(y0 + ph + 3.f) + 3);
  const int iw = ix1 - ix0, ih = iy1 - iy0;
  if (iw < 8 || ih < 8 || W < 64 || H < 64) return;  // degenerate guard

  glBindTexture(GL_TEXTURE_2D, g_pip_tex);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, g_pip_w, g_pip_h, GL_RGB,
                  GL_UNSIGNED_BYTE, frame.data());

  // white 2-px border behind the camera image: drawn with the HUD program
  // under a scissor locked to the border rect (also cannot leak)
  {
    glViewport(ix0, iy0, iw, ih);
    glEnable(GL_SCISSOR_TEST);
    glScissor(ix0, iy0, iw, ih);
    glUseProgram(g_hud_prog);
    glUniform4f(g_huColor, 1.f, 1.f, 1.f, 1.f);
    // fullscreen quad in this viewport == the border rect
    static const float bquad[12] = {-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1};
    stream_draw_2f(g_hud_vbo, g_huPos,
                   std::vector<float>(bquad, bquad + 12));
  }

  // camera image: fullscreen quad through a viewport+scissor = exactly the
  // PiP rectangle (border 3 px shows around it)
  glViewport(ix0 + 3, iy0 + 3, std::max(2, iw - 6), std::max(2, ih - 6));
  glScissor(ix0 + 3, iy0 + 3, std::max(2, iw - 6), std::max(2, ih - 6));
  // uv: row 0 of the buffer is the image TOP -> v=0 at the quad top
  static const float q[6][4] = {{-1.f, 1.f, 0.f, 0.f}, {1.f, 1.f, 1.f, 0.f},
                                {1.f, -1.f, 1.f, 1.f}, {-1.f, 1.f, 0.f, 0.f},
                                {1.f, -1.f, 1.f, 1.f}, {-1.f, -1.f, 0.f, 1.f}};
  std::vector<float> v;
  for (const auto& c : q) v.insert(v.end(), c, c + 4);

  glUseProgram(g_pip_prog);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, g_pip_tex);
  glUniform1i(g_pip_uTex, 0);
  if (!g_pip_vbo) glGenBuffers(1, &g_pip_vbo);
  glBindBuffer(GL_ARRAY_BUFFER, g_pip_vbo);
  glBufferData(GL_ARRAY_BUFFER, v.size() * sizeof(float), v.data(),
               GL_STREAM_DRAW);
  glEnableVertexAttribArray((GLuint)g_pip_aPos);
  glEnableVertexAttribArray((GLuint)g_pip_aUV);
  glVertexAttribPointer((GLuint)g_pip_aPos, 2, GL_FLOAT, GL_FALSE, 16, (void*)0);
  glVertexAttribPointer((GLuint)g_pip_aUV, 2, GL_FLOAT, GL_FALSE, 16, (void*)8);
  glDrawArrays(GL_TRIANGLES, 0, (GLint)(v.size() / 4));
  glDisableVertexAttribArray((GLuint)g_pip_aPos);
  glDisableVertexAttribArray((GLuint)g_pip_aUV);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // restore full-screen state — the HUD after this frame depends on it
  glViewport(0, 0, W, H);
  glDisable(GL_SCISSOR_TEST);
  glBindTexture(GL_TEXTURE_2D, 0);
}

// ---------------- touch buttons + UI state ----------------

// ONE layout for the bottom button bar — shared by the HUD draw and the
// touch hit-test (v1.1.0 had two copies that could drift apart)
void ui_button_rect(int i, float r[4]) {
  const int W = g_win_w, H = g_win_h;
  const float m = 0.012f * W;
  const float gap = 0.009f * W;
  const float bw = (W - 2 * m - (BTN_COUNT - 1) * gap) / (float)BTN_COUNT;
  const float bh = 0.115f * H;
  // pixel coords with TOP-left origin (matches Android touch coords)
  r[0] = m + i * (bw + gap);
  r[1] = H - bh - 0.020f * H;
  r[2] = r[0] + bw;
  r[3] = r[1] + bh;
}

int gles_hit_button(float x, float y) {
  for (int i = 0; i < BTN_COUNT; ++i) {
    float r[4];
    ui_button_rect(i, r);
    if (x >= r[0] && x <= r[2] && y >= r[1] && y <= r[3]) return i;
  }
  return -1;
}

PcsUiState gles_take_ui() {
  std::lock_guard<std::mutex> lk(g_ui_mtx);
  PcsUiState out = g_ui;
  g_ui = PcsUiState{};
  return out;
}

void gles_push_ui(const PcsUiState& s) {
  std::lock_guard<std::mutex> lk(g_ui_mtx);
  g_ui.toggle_run |= s.toggle_run;
  g_ui.stop |= s.stop;
  g_ui.fine |= s.fine;
  g_ui.new_episode |= s.new_episode;
}

void gles_set_diag(int bind, int gl_errs, long cycles, int sorted, int total,
                   int stacked, bool paused, bool halted, bool finetuning) {
  std::lock_guard<std::mutex> lk(g_ui_mtx);
  DiagState d;
  d.bind = bind;
  d.gl_errs = gl_errs;
  d.cycles = cycles;
  d.sorted = sorted;
  d.total = total;
  d.stacked = stacked;
  d.paused = paused;
  d.halted = halted;
  d.finetuning = finetuning;
  g_diag = d;
}

// ---------------- event-camera FBO pass ----------------
static GLuint g_fbo = 0, g_fbo_tex = 0;
static uint8_t* g_readback = nullptr;

void gles_render_event_frame(SimGlue& glue, bool jitter, uint8_t* rgb, int w, int h) {
  const mjModel* m = glue.model();
  std::vector<float> pos, quat;
  copy_poses(pos, quat);
  if (pos.size() < (size_t)3 * m->nbody) return;  // no snapshot yet
  if (!g_fbo) {
    glGenFramebuffers(1, &g_fbo);
    glGenTextures(1, &g_fbo_tex);
    glBindTexture(GL_TEXTURE_2D, g_fbo_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindFramebuffer(GL_FRAMEBUFFER, g_fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g_fbo_tex, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    g_readback = new uint8_t[w * h * 3];
  }
  GLint prev_fb = 0;
  glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev_fb);  // restore on exit
  glBindFramebuffer(GL_FRAMEBUFFER, g_fbo);
  glViewport(0, 0, w, h);
  glClearColor(0.45f, 0.38f, 0.30f, 1.f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);

  glUseProgram(g_prog);
  // top-down camera (cam_event), optional micro-jitter for refresh pulses
  const float jx = jitter ? 0.008f : 0.f, jy = jitter ? 0.004f : 0.f;
  float eye[3] = {0.42f + jx, 0.f + jy, 1.05f}, ctr[3] = {0.42f + jx, jy, 0.f};
  float up[3] = {0, 1, 0};
  Mat4 view = Mat4::lookAt(eye, ctr, up);
  Mat4 proj = Mat4::perspective(78.f * (float)M_PI / 180.f, (float)w / h, 0.05f, 5.f);
  Mat4 vp = proj * view;  // same fixed column-major path as the view render
  glUniformMatrix4fv(g_uMVP, 1, GL_FALSE, vp.m);
  for (int g = 0; g < m->ngeom; ++g) {
    if (m->geom_type[g] == mjGEOM_PLANE) continue;
    // perception needs shape, not beauty: render the CHEAP collision meshes
    // (group 3) + primitives (group 0), skip the high-poly visual meshes
    // (group 2, ~500k faces) — the 100 Hz pass stays far inside budget.
    if (m->geom_group[g] == 2) continue;
    // skip translucent markers (zone plates share cube hue families and
    // would pollute the color-binned event pipeline)
    float a = 1.f;
    const int matid = m->geom_matid[g];
    if (matid >= 0) a = (float)m->mat_rgba[4 * matid + 3];
    else a = (float)m->geom_rgba[4 * g + 3];
    if (a < 0.95f) continue;
    draw_geom(m, pos, quat, g, 1.f);
  }
  glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, g_readback);
  glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)prev_fb);
  // flip vertically (GL origin bottom-left)
  for (int y = 0; y < h; ++y)
    memcpy(rgb + 3 * y * w, g_readback + 3 * (h - 1 - y) * w, 3 * w);
}

void out_stats_set(const CycleStats& st) { g_last_stats = st; }
void out_task_set(const TaskOutput& t) { g_last_task = t; }

// ---------------- status / error screen (never-black guarantee) ----------------
// Own program + geometry on the calling thread's context; works BEFORE
// gles_init so loading and error states are always visible.
static GLuint g_status_prog = 0;
static GLint g_status_pos = -1, g_status_col = -1;

void gles_set_window_size(int w, int h) {
  if (w > 0 && h > 0) { g_win_w = w; g_win_h = h; }
}
int gles_win_w() { return g_win_w; }
int gles_win_h() { return g_win_h; }

// 3x5 pixel glyphs, rows top->bottom (bit 2 = leftmost column)
static const uint8_t kFont[47][5] = {
    {7, 5, 5, 5, 7},   // 0  (0)
    {2, 6, 2, 2, 7},   // 1
    {7, 1, 7, 4, 7},   // 2
    {7, 1, 7, 1, 7},   // 3
    {5, 5, 7, 1, 1},   // 4
    {7, 4, 7, 1, 7},   // 5
    {7, 4, 7, 5, 7},   // 6
    {7, 1, 1, 2, 2},   // 7
    {7, 5, 7, 5, 7},   // 8
    {7, 5, 7, 1, 7},   // 9
    {2, 5, 7, 5, 5},   // A  (10)
    {6, 5, 6, 5, 6},   // B
    {3, 4, 4, 4, 3},   // C
    {6, 5, 5, 5, 6},   // D
    {7, 4, 6, 4, 7},   // E
    {7, 4, 6, 4, 4},   // F
    {3, 4, 5, 5, 3},   // G
    {5, 5, 7, 5, 5},   // H
    {7, 2, 2, 2, 7},   // I
    {1, 1, 1, 5, 2},   // J
    {5, 5, 6, 5, 5},   // K
    {4, 4, 4, 4, 7},   // L
    {5, 7, 7, 5, 5},   // M
    {6, 5, 5, 5, 5},   // N
    {7, 5, 5, 5, 7},   // O
    {6, 5, 6, 4, 4},   // P
    {7, 5, 5, 7, 1},   // Q
    {6, 5, 6, 5, 5},   // R
    {3, 4, 2, 1, 6},   // S
    {7, 2, 2, 2, 2},   // T
    {5, 5, 5, 5, 7},   // U
    {5, 5, 5, 5, 2},   // V
    {5, 5, 7, 7, 5},   // W
    {5, 5, 2, 5, 5},   // X
    {5, 5, 2, 2, 2},   // Y
    {7, 1, 2, 4, 7},   // Z
    {0, 0, 7, 0, 0},   // -  (36)
    {0, 2, 0, 2, 0},   // :
    {1, 1, 2, 4, 4},   // /
    {0, 0, 0, 0, 2},   // .
    {2, 2, 7, 5, 7},   // ä  (40) — bit 0 col: umlaut dots approximated
    {2, 5, 7, 5, 7},   // ö
    {5, 5, 7, 7, 5},   // ü
    {2, 5, 7, 7, 5},   // A-umlaut alt (unused)
    {5, 5, 2, 5, 2},   // ß-like (unused)
    {0, 0, 0, 0, 0},   // space (44)
    {7, 5, 5, 5, 5}};  // F-umlaut alt (unused)

static const uint8_t* glyph_of(char c) {
  if (c >= '0' && c <= '9') return kFont[c - '0'];
  if (c >= 'A' && c <= 'Z') return kFont[c - 'A' + 10];
  if (c >= 'a' && c <= 'z') return kFont[c - 'a' + 10];  // map lower -> upper
  switch (c) {
    case '-': return kFont[36];
    case ':': return kFont[37];
    case '/': return kFont[38];
    case '.': return kFont[39];
    case ' ': return kFont[44];
    case 0xE4: return kFont[40];  // ä
    case 0xF6: return kFont[41];  // ö
    case 0xFC: return kFont[42];  // ü
    case 0xDF: return kFont[43];  // ß
  }
  return kFont[44];  // space for unknowns
}

// append one text line as screen-space quad triangles
static void text_quads(std::vector<float>& v, const char* s, float x0,
                       float y0, float scale, int win_w, int win_h) {
  float gx = 0;
  for (const char* p = s; *p; ++p, gx += 3 * scale + scale) {
    const uint8_t* g = glyph_of(*p);
    for (int r = 0; r < 5; ++r) {
      for (int c = 0; c < 3; ++c) {
        if (!((g[r] >> (2 - c)) & 1)) continue;
        const float px = x0 + gx + c * scale;
        const float py = y0 + r * scale;
        const float xa = 2.f * px / win_w - 1.f;
        const float xb = 2.f * (px + scale) / win_w - 1.f;
        const float ya = 1.f - 2.f * py / win_h;
        const float yb = 1.f - 2.f * (py + scale) / win_h;
        v.insert(v.end(), {xa, ya, xb, ya, xb, yb});
        v.insert(v.end(), {xa, ya, xb, yb, xa, yb});
      }
    }
  }
}

static float text_width(const char* s, float scale) {
  int n = 0;
  for (const char* p = s; *p; ++p) ++n;
  return n > 0 ? n * 3 * scale + (n - 1) * scale : 0.f;
}

void gles_render_status(int code, int win_w, int win_h, const char* sub) {
  if (!g_status_prog) {
    g_status_prog = make_program(kHudVS, kHudFS);
    g_status_pos = glGetAttribLocation(g_status_prog, "aPos");
    g_status_col = glGetUniformLocation(g_status_prog, "uColor");
  }
  glViewport(0, 0, win_w, win_h);
  const bool error = code >= 2;
  if (error) {
    glClearColor(0.42f, 0.06f, 0.06f, 1.f);  // dark red: fatal
  } else {
    glClearColor(0.52f, 0.40f, 0.08f, 1.f);  // amber: loading
  }
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glDisable(GL_DEPTH_TEST);

  char text[3] = {'L', '0', 0};
  if (error) {
    text[0] = 'E';
    text[1] = (char)('0' + (code % 10));
  }

  const int scale = std::max(8, std::min(win_w, win_h) / 24);
  const float gh = 5.f * scale;
  const float gw = text_width(text, (float)scale);
  // optional hex sub-line (e.g. "0X3009") in half-size glyphs below the code
  float sub_scale = 0.f;
  if (sub && *sub) sub_scale = std::max(4.f, scale * 0.5f);
  const float sub_h = sub_scale > 0.f ? 5.f * sub_scale + scale : 0.f;

  const float x0 = (win_w - gw) * 0.5f;
  const float y0 = (win_h - gh - sub_h) * 0.5f;

  std::vector<float> v;
  text_quads(v, text, x0, y0, (float)scale, win_w, win_h);
  if (sub_scale > 0.f) {
    const float sw = text_width(sub, sub_scale);
    text_quads(v, sub, (win_w - sw) * 0.5f, y0 + gh + scale, sub_scale, win_w,
               win_h);
  }
  if (v.empty()) return;
  glUseProgram(g_status_prog);
  glUniform4f(g_status_col, 1.f, 1.f, 1.f, 1.f);
  if (!g_status_vbo) glGenBuffers(1, &g_status_vbo);
  stream_draw_2f(g_status_vbo, g_status_pos, v);
}
