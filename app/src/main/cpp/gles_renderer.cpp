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
// v1.3.0 overlay windows (NN info + motion manager)
static void ui_window_rect(int which, float r[4]);
static bool ui_window_button_rect(int win, int btn, float r[4]);
static void gles_render_windows();

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
  if (g_huPos < 0 || g_hud_prog == 0) return;
  const int W = g_win_w, H = g_win_h;

  glUseProgram(g_hud_prog);

  // ---- batch 1: button fills + budget bar (dark, then accent) ----
  std::vector<float> v;
  for (int i = 0; i < BTN_COUNT; ++i) {
    float r[4];
    ui_button_rect(i, r);
    const float xa = 2.f * r[0] / W - 1.f, xb = 2.f * r[2] / W - 1.f;
    const float ya = 1.f - 2.f * r[1] / H, yb = 1.f - 2.f * r[3] / H;
    v.insert(v.end(), {xa, ya, xb, ya, xb, yb});
    v.insert(v.end(), {xa, ya, xb, yb, xa, yb});
  }
  // cycle-budget bar along the top edge (green ok / amber tight / red over)
  {
    const float frac = std::min(1.0f, (float)st.t_total / 10000.f);
    const bool over = st.t_total > 10000;
    const float xa = -1.f, xb = -1.f + 2.f * frac;
    const float ya = 1.f, yb = 1.f - 2.f * std::max(4.f, 0.005f * H) / H;
    (void)over;
    v.insert(v.end(), {xa, ya, xb, ya, xb, yb});
    v.insert(v.end(), {xa, ya, xb, yb, xa, yb});
  }
  glUniform4f(g_huColor, 0.13f, 0.14f, 0.16f, 0.92f);
  stream_draw_2f(g_hud_vbo, g_huPos, v);

  // ---- batch 2: accent underline on the START button + bar color ----
  v.clear();
  {
    float r[4];
    ui_button_rect(BTN_START, r);
    const float xa = 2.f * r[0] / W - 1.f, xb = 2.f * r[2] / W - 1.f;
    const float ya = 1.f - 2.f * (r[3] - 0.012f * H) / H, yb = 1.f - 2.f * r[3] / H;
    v.insert(v.end(), {xa, ya, xb, ya, xb, yb});
    v.insert(v.end(), {xa, ya, xb, yb, xa, yb});
    // budget bar recolor pass: draw over with the true color
    const float frac = std::min(1.0f, (float)st.t_total / 10000.f);
    const float bxa = -1.f, bxb = -1.f + 2.f * frac;
    const float bya = 1.f, byb = 1.f - 2.f * std::max(4.f, 0.005f * H) / H;
    v.insert(v.end(), {bxa, bya, bxb, bya, bxb, byb});
    v.insert(v.end(), {bxa, bya, bxb, byb, bxa, byb});
  }
  glUniform4f(g_huColor, g_diag.paused ? 0.95f : 0.20f,
              g_diag.paused ? 0.75f : 0.85f, g_diag.paused ? 0.10f : 0.30f,
              1.f);
  stream_draw_2f(g_hud_vbo, g_huPos, v);

  // ---- batch 3: white text (labels, diag, counters) ----
  const float s = std::max(3.f, H / 90.f);  // glyph pixel scale
  v.clear();
  static const char* kLabels[BTN_COUNT] = {"START", "STOP", "FINE", "NEU",
                                           "NN", "MOT"};
  for (int i = 0; i < BTN_COUNT; ++i) {
    float r[4];
    ui_button_rect(i, r);
    const char* lbl = kLabels[i];
    if (i == BTN_START) lbl = g_diag.paused ? "START" : "PAUSE";
    if (i == BTN_NN && gles_window_open(WIN_NN)) lbl = "NN X";
    if (i == BTN_MOT && gles_window_open(WIN_MOT)) lbl = "MOT X";
    const float tw = text_width(lbl, s);
    text_quads(v, lbl, (r[0] + r[2]) * 0.5f - tw * 0.5f,
               (r[1] + r[3]) * 0.5f - 2.5f * s, s, W, H);
  }
  // diagnostics top-left (visible WITHOUT adb)
  char l1[48], l2[48];
  snprintf(l1, sizeof l1, "SORTIERT %d/%d GESTAPELT %d", g_diag.sorted,
           g_diag.total, g_diag.stacked);
  const char* mode = g_diag.finetuning
                         ? "FINE"
                         : g_diag.halted ? "HALT"
                                         : (g_diag.paused ? "PAUSE" : "AKTIV");
  snprintf(l2, sizeof l2, "ZYK %ld B%d G%d %s", g_diag.cycles, g_diag.bind,
           g_diag.gl_errs, mode);
  text_quads(v, l1, 0.02f * W, 0.03f * H + 6.f * s, s, W, H);
  text_quads(v, l2, 0.02f * W, 0.03f * H, s, W, H);
  glUniform4f(g_huColor, 1.f, 1.f, 1.f, 1.f);
  stream_draw_2f(g_hud_vbo, g_huPos, v);

  // ---- v1.3.0: overlay windows (NN info / motion manager) ----
  gles_render_windows();
}

// ---------------- v1.3.0 overlay windows (NN info / motion manager) ----------------
// Pixel coords are TOP-left origin (same convention as touches + text).

static void push_rect(std::vector<float>& v, const float r[4], int W, int H) {
  const float xa = 2.f * r[0] / W - 1.f, xb = 2.f * r[2] / W - 1.f;
  const float ya = 1.f - 2.f * r[1] / H, yb = 1.f - 2.f * r[3] / H;
  v.insert(v.end(), {xa, ya, xb, ya, xb, yb});
  v.insert(v.end(), {xa, ya, xb, yb, xa, yb});
}

static void ui_window_rect(int which, float r[4]) {
  const int W = g_win_w, H = g_win_h;
  const float s2 = std::max(3.f, H / 140.f);
  const float lh = 6.5f * s2;              // text line height
  const int lines = (which == WIN_NN) ? 12 : 9;
  r[0] = 0.02f * W;
  r[1] = 0.115f * H;
  r[2] = r[0] + 0.42f * W;
  r[3] = r[1] + lines * lh + 0.105f * H;   // text + button row + margins
  if (r[3] > H - 0.155f * H) r[3] = H - 0.155f * H;  // keep above the bar
}

static bool ui_window_button_rect(int win, int btn, float r[4]) {
  const int W = g_win_w, H = g_win_h;
  const int n = (win == WIN_NN) ? 2 : 5;
  if (btn < 0 || btn >= n) return false;
  float wr[4];
  ui_window_rect(win, wr);
  const float m = 0.008f * W;
  const float gap = 0.006f * W;
  const float bh = 0.052f * H;
  const float bw = ((wr[2] - wr[0]) - 2 * m - (n - 1) * gap) / (float)n;
  r[0] = wr[0] + m + btn * (bw + gap);
  r[1] = wr[3] - bh - 0.010f * H;
  r[2] = r[0] + bw;
  r[3] = r[1] + bh;
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

static void gles_render_windows() {
  const bool open_nn = gles_window_open(WIN_NN);
  const bool open_mot = gles_window_open(WIN_MOT);
  if ((!open_nn && !open_mot) || g_hud_prog == 0) return;
  const int W = g_win_w, H = g_win_h;
  const float s2 = std::max(3.f, H / 140.f);
  const float lh = 6.5f * s2;

  // live state snapshot (loop thread publishes under g_ui_mtx)
  PcsNnState nn;
  PcsMotState mot;
  {
    std::lock_guard<std::mutex> lk(g_ui_mtx);
    nn = g_nn;
    mot = g_mot;
  }

  glUseProgram(g_hud_prog);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // ---- batch A: panels ----
  std::vector<float> v;
  float wr_nn[4] = {0}, wr_mot[4] = {0};
  if (open_nn) { ui_window_rect(WIN_NN, wr_nn); push_rect(v, wr_nn, W, H); }
  if (open_mot) { ui_window_rect(WIN_MOT, wr_mot); push_rect(v, wr_mot, W, H); }
  glUniform4f(g_huColor, 0.08f, 0.09f, 0.11f, 0.86f);
  stream_draw_2f(g_hud_vbo, g_huPos, v);

  // ---- batch B: in-window buttons (slate) + close buttons (red) ----
  v.clear();
  for (int w = 0; w < WIN_COUNT; ++w) {
    if (!(w == WIN_NN ? open_nn : open_mot)) continue;
    const int n = (w == WIN_NN) ? 2 : 5;
    for (int b = 0; b < n; ++b) {
      float r[4];
      if (!ui_window_button_rect(w, b, r)) continue;
      if (!((w == WIN_NN && b == 0) || (w == WIN_MOT && b == 0)))
        push_rect(v, r, W, H);
    }
  }
  glUniform4f(g_huColor, 0.22f, 0.24f, 0.29f, 0.96f);
  stream_draw_2f(g_hud_vbo, g_huPos, v);
  v.clear();
  for (int w = 0; w < WIN_COUNT; ++w) {
    if (!(w == WIN_NN ? open_nn : open_mot)) continue;
    float r[4];
    if (ui_window_button_rect(w, 0, r)) push_rect(v, r, W, H);
  }
  glUniform4f(g_huColor, 0.62f, 0.18f, 0.14f, 0.96f);
  stream_draw_2f(g_hud_vbo, g_huPos, v);

  // ---- batch C: text ----
  v.clear();
  static const char* kNnStatic[] = {
      "EVENT-KAMERA 96X72 - 576 BINS",
      "ALIF-LSNN: 128 NEURONE ADAPTIV",
      "FEP-CODER: 32D EMBEDDING",
      "SOFT-MOE: 8 PROTOTYPEN",
      "KAN-MLP 24-8-8 + LORA R4 LYAP",
      "",
  };
  if (open_nn) {
    text_quads(v, "NEURONALES NETZ - 100 HZ", wr_nn[0] + 0.010f * W,
               wr_nn[1] + 0.012f * H, s2, W, H);
    float ty = wr_nn[1] + 0.012f * H + 2 * lh;
    for (const char* l : kNnStatic) {
      text_quads(v, l, wr_nn[0] + 0.010f * W, ty, s2, W, H);
      ty += lh;
    }
    char ln[48];
    snprintf(ln, sizeof ln, "EVENTS %u  SPIKES %u", nn.events, nn.spikes);
    text_quads(v, ln, wr_nn[0] + 0.010f * W, ty, s2, W, H);
    ty += lh;
    snprintf(ln, sizeof ln, "EMB %.3f  FEP-E %.4f", nn.emb_norm,
             nn.free_energy);
    text_quads(v, ln, wr_nn[0] + 0.010f * W, ty, s2, W, H);
    ty += lh;
    snprintf(ln, sizeof ln, "LORA ETA %.3f  V %.4f", nn.lora_eta, nn.lora_v);
    text_quads(v, ln, wr_nn[0] + 0.010f * W, ty, s2, W, H);
    ty += lh;
    snprintf(ln, sizeof ln, "MOE-MAX %.2f  PHASE %s", nn.moe_max,
             kPhaseName[nn.phase & 7]);
    text_quads(v, ln, wr_nn[0] + 0.010f * W, ty, s2, W, H);
    ty += lh;
    snprintf(ln, sizeof ln, "T US: P%u E%u S%u M%u", nn.t_phys, nn.t_event,
             nn.t_snn, nn.t_mlp);
    text_quads(v, ln, wr_nn[0] + 0.010f * W, ty, s2, W, H);
  }
  if (open_mot) {
    text_quads(v, "MOTION-MANAGER", wr_mot[0] + 0.010f * W,
               wr_mot[1] + 0.012f * H, s2, W, H);
    float ty = wr_mot[1] + 0.012f * H + 2 * lh;
    char ln[48];
    text_quads(v, mot.msg[0] ? mot.msg : "BEREIT", wr_mot[0] + 0.010f * W, ty,
               s2, W, H);
    ty += lh;
    snprintf(ln, sizeof ln, "CLIPS %d  SAMPLES %d", mot.clips, mot.samples);
    text_quads(v, ln, wr_mot[0] + 0.010f * W, ty, s2, W, H);
    ty += lh;
    snprintf(ln, sizeof ln, "MERKMAL 32D  N %.2f", mot.last_feat_norm);
    text_quads(v, ln, wr_mot[0] + 0.010f * W, ty, s2, W, H);
    ty += lh;
    snprintf(ln, sizeof ln, "LORA-UPDATES %d", mot.updates);
    text_quads(v, ln, wr_mot[0] + 0.010f * W, ty, s2, W, H);
    ty += lh;
    text_quads(v, "", wr_mot[0] + 0.010f * W, ty, s2, W, H);
    ty += lh;
    text_quads(v, "AUFZ: LETZTE 2.5 S AUFNEHMEN", wr_mot[0] + 0.010f * W, ty,
               s2, W, H);
    ty += lh;
    text_quads(v, "UMW: IN 32D-DATENSATZ UMWANDELN", wr_mot[0] + 0.010f * W,
               ty, s2, W, H);
    ty += lh;
    text_quads(v, "TRAIN: LORA-LYAPUNOV UPDATES", wr_mot[0] + 0.010f * W, ty,
               s2, W, H);
  }
  glUniform4f(g_huColor, 1.f, 1.f, 1.f, 1.f);
  stream_draw_2f(g_hud_vbo, g_huPos, v);

  // ---- batch D: button labels ----
  v.clear();
  static const char* kNnBtn[2] = {"X", "KAM"};
  static const char* kMotBtn[5] = {"X", "AUFZ", "UMW", "TRAIN", "LOESCH"};
  for (int w = 0; w < WIN_COUNT; ++w) {
    if (!(w == WIN_NN ? open_nn : open_mot)) continue;
    const int n = (w == WIN_NN) ? 2 : 5;
    for (int b = 0; b < n; ++b) {
      float r[4];
      if (!ui_window_button_rect(w, b, r)) continue;
      const char* lbl = (w == WIN_NN) ? kNnBtn[b] : kMotBtn[b];
      const float tw = text_width(lbl, s2);
      text_quads(v, lbl, (r[0] + r[2]) * 0.5f - tw * 0.5f,
                 (r[1] + r[3]) * 0.5f - 2.5f * s2, s2, W, H);
    }
  }
  glUniform4f(g_huColor, 1.f, 1.f, 1.f, 1.f);
  stream_draw_2f(g_hud_vbo, g_huPos, v);

  glDisable(GL_BLEND);
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
  const float ph = 0.30f * H;
  const float pw = ph * (float)g_pip_w / g_pip_h;  // keep source aspect
  const float x1 = W - 0.015f * W, x0 = x1 - pw;
  const float y0 = 0.06f * H, y1 = y0 + ph;
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
  const float m = 0.014f * W;
  const float gap = 0.010f * W;
  const float bw = (W - 2 * m - (BTN_COUNT - 1) * gap) / (float)BTN_COUNT;
  const float bh = 0.105f * H;
  // pixel coords with TOP-left origin (matches Android touch coords)
  r[0] = m + i * (bw + gap);
  r[1] = H - bh - 0.018f * H;
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
