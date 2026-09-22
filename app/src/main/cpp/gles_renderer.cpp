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
    "void main(){ float d = max(0.35, dot(normalize(vNrm), normalize(vec3(0.4,0.5,0.8))));\n"
    "  gl_FragColor = vec4(uColor.rgb * d, uColor.a); }";

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

  LOGW("gles_init done %dx%d", win_w, win_h);
}

// draw one geom (box or capsule) with color; poses come from the snapshot
static void draw_geom(const mjModel* m, const std::vector<float>& pos,
                      const std::vector<float>& quat, int g, float alpha) {
  const int body = m->geom_bodyid[g];
  float M[16];
  body_transform(body, pos, quat, M);
  const mjtNum* size = m->geom_size + 3 * g;
  const int type = m->geom_type[g];
  const float* rgba = m->geom_rgba + 4 * g;
  float col[4] = {(float)rgba[0], (float)rgba[1], (float)rgba[2],
                  alpha < 0.f ? (float)rgba[3] : alpha};

  Mat4 Mb, scale = Mat4::identity();
  std::memcpy(Mb.m, M, sizeof(Mb.m));
  if (type == mjGEOM_CAPSULE || type == mjGEOM_CYLINDER) {
    // MuJoCo capsule: along local z, half-length size[1]; our VBO: along x
    scale.m[0] = (float)size[0]; scale.m[5] = (float)size[0]; scale.m[10] = (float)size[1];
    Mat4 rot = Mat4::identity();   // align z-axis with x-axis
    rot.m[0] = 0; rot.m[2] = -1; rot.m[8] = 1; rot.m[10] = 0;
    Mat4 model = Mb * rot * scale;
    glUniformMatrix4fv(g_uModel, 1, GL_FALSE, model.m);
  } else if (type == mjGEOM_BOX) {
    scale.m[0] = (float)size[0]; scale.m[5] = (float)size[1]; scale.m[10] = (float)size[2];
    Mat4 model = Mb * scale;
    glUniformMatrix4fv(g_uModel, 1, GL_FALSE, model.m);
  } else if (type == mjGEOM_SPHERE) {
    scale.m[0] = scale.m[5] = scale.m[10] = (float)size[0];
    Mat4 model = Mb * scale;
    glUniformMatrix4fv(g_uModel, 1, GL_FALSE, model.m);
  } else if (type == mjGEOM_PLANE) {
    scale.m[0] = (float)size[0]; scale.m[5] = (float)size[1];
    scale.m[10] = 0.01f;  // thin slab — MuJoCo planes have no thickness
    Mat4 model = Mb * scale;
    glUniformMatrix4fv(g_uModel, 1, GL_FALSE, model.m);
  } else {
    return;
  }
  glUniform4fv(g_uColor, 1, col);
  GLuint vbo = g_cube_vbo;
  int count = 36;
  if (type == mjGEOM_CAPSULE || type == mjGEOM_CYLINDER) {
    vbo = g_cyl_vbo; count = g_cyl_count;
  } else if (type == mjGEOM_SPHERE) {
    vbo = g_sph_vbo; count = g_sph_count;
  }
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glEnableVertexAttribArray((GLuint)g_aPos);
  glEnableVertexAttribArray((GLuint)g_aNrm);
  glVertexAttribPointer((GLuint)g_aPos, 3, GL_FLOAT, GL_FALSE, 24, (void*)0);
  glVertexAttribPointer((GLuint)g_aNrm, 3, GL_FLOAT, GL_FALSE, 24, (void*)12);
  glDrawArrays(GL_TRIANGLES, 0, count);
  glDisableVertexAttribArray((GLuint)g_aPos);
  glDisableVertexAttribArray((GLuint)g_aNrm);
}

void gles_render_view(SimGlue& glue, Controller& ctrl) {
  const mjModel* m = glue.model();
  std::vector<float> pos, quat;
  copy_poses(pos, quat);
  const bool have_poses = pos.size() >= (size_t)3 * m->nbody;
  // NEVER early-return: an invisible failure mode (grey screen) is worse
  // than a degraded frame. Without poses we still clear + show the HUD.
  glViewport(0, 0, g_win_w, g_win_h);
  glClearColor(0.09f, 0.10f, 0.12f, 1.f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);

  if (have_poses) {
    glUseProgram(g_prog);
    float eye[3] = {1.35f, -1.15f, 0.95f}, ctr[3] = {0.42f, 0.f, 0.30f},
          up[3] = {0, 0, 1};
    Mat4 view = Mat4::lookAt(eye, ctr, up);
    Mat4 proj = Mat4::perspective(50.f * (float)M_PI / 180.f,
                                  (float)g_win_w / g_win_h, 0.05f, 10.f);
    Mat4 vp = proj * view;
    glUniformMatrix4fv(g_uMVP, 1, GL_FALSE, vp.m);

    // draw everything visible: floor, table, zones, arm, hand, cubes.
    // v1.0.3 skipped contype==0 geoms — that hid the colored sorting zones.
    for (int g = 0; g < m->ngeom; ++g) {
      const float* rgba = m->geom_rgba + 4 * g;
      if (rgba[3] < 0.05f) continue;  // fully transparent only
      draw_geom(m, pos, quat, g, 1.f);
    }
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

  // ---- shared button layout (top-left origin, same space as touches) ----
  auto button_rect = [&](int i, float r[4]) {
    const float m = 0.018f * W;
    const float gap = 0.014f * W;
    const float bw = (W - 2 * m - 3 * gap) / 4.f;
    const float bh = 0.115f * H;
    r[0] = m + i * (bw + gap);
    r[1] = H - bh - 0.018f * H;
    r[2] = r[0] + bw;
    r[3] = r[1] + bh;
  };

  glUseProgram(g_hud_prog);

  // ---- batch 1: button fills + budget bar (dark, then accent) ----
  std::vector<float> v;
  for (int i = 0; i < BTN_COUNT; ++i) {
    float r[4];
    button_rect(i, r);
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
  if (!v.empty()) {
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, v.size() * sizeof(float), v.data(),
                 GL_STREAM_DRAW);
    glEnableVertexAttribArray((GLuint)g_huPos);
    glVertexAttribPointer((GLuint)g_huPos, 2, GL_FLOAT, GL_FALSE, 8, (void*)0);
    glDrawArrays(GL_TRIANGLES, 0, (GLint)(v.size() / 2));
    glDisableVertexAttribArray((GLuint)g_huPos);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDeleteBuffers(1, &vbo);
  }

  // ---- batch 2: accent underline on the START button + bar color ----
  v.clear();
  {
    float r[4];
    button_rect(BTN_START, r);
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
  if (!v.empty()) {
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, v.size() * sizeof(float), v.data(),
                 GL_STREAM_DRAW);
    glEnableVertexAttribArray((GLuint)g_huPos);
    glVertexAttribPointer((GLuint)g_huPos, 2, GL_FLOAT, GL_FALSE, 8, (void*)0);
    glDrawArrays(GL_TRIANGLES, 0, (GLint)(v.size() / 2));
    glDisableVertexAttribArray((GLuint)g_huPos);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDeleteBuffers(1, &vbo);
  }

  // ---- batch 3: white text (labels, diag, counters) ----
  const float s = std::max(3.f, H / 90.f);  // glyph pixel scale
  v.clear();
  static const char* kLabels[BTN_COUNT][2] = {
      {"START", "PAUSE"}, {"STOP", "STOP"}, {"FINE", "FINE"}, {"NEU", "NEU"}};
  for (int i = 0; i < BTN_COUNT; ++i) {
    float r[4];
    button_rect(i, r);
    const char* lbl = kLabels[i][g_diag.paused && i == BTN_START ? 1 : 0];
    if (i == BTN_START) lbl = g_diag.paused ? "START" : "PAUSE";
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
  if (!v.empty()) {
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, v.size() * sizeof(float), v.data(),
                 GL_STREAM_DRAW);
    glEnableVertexAttribArray((GLuint)g_huPos);
    glVertexAttribPointer((GLuint)g_huPos, 2, GL_FLOAT, GL_FALSE, 8, (void*)0);
    glDrawArrays(GL_TRIANGLES, 0, (GLint)(v.size() / 2));
    glDisableVertexAttribArray((GLuint)g_huPos);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDeleteBuffers(1, &vbo);
  }
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

  glBindTexture(GL_TEXTURE_2D, g_pip_tex);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, g_pip_w, g_pip_h, GL_RGB,
                  GL_UNSIGNED_BYTE, frame.data());

  const float xa = 2.f * x0 / W - 1.f, xb = 2.f * x1 / W - 1.f;
  const float ya = 1.f - 2.f * y0 / H, yb = 1.f - 2.f * y1 / H;
  // uv: row 0 of the buffer is the image TOP -> v=0 at the quad top
  const float q[4][4] = {{xa, ya, 0.f, 0.f}, {xb, ya, 1.f, 0.f},
                         {xb, yb, 1.f, 1.f}, {xa, yb, 0.f, 1.f}};
  const int idx[6] = {0, 1, 2, 0, 2, 3};
  std::vector<float> v;
  for (int i : idx) v.insert(v.end(), q[i], q[i] + 4);

  // white 2-px border behind the camera image (makes the PiP readable
  // against the grey table)
  {
    const float bx0 = 2.f * (x0 - 3.f) / W - 1.f, bx1 = 2.f * (x1 + 3.f) / W - 1.f;
    const float bya = 1.f - 2.f * (y0 - 3.f) / H, byb = 1.f - 2.f * (y1 + 3.f) / H;
    std::vector<float> bv;
    bv.insert(bv.end(), {bx0, bya, bx1, bya, bx1, byb});
    bv.insert(bv.end(), {bx0, bya, bx1, byb, bx0, byb});
    glUseProgram(g_hud_prog);
    glUniform4f(g_huColor, 1.f, 1.f, 1.f, 1.f);
    GLuint bbo;
    glGenBuffers(1, &bbo);
    glBindBuffer(GL_ARRAY_BUFFER, bbo);
    glBufferData(GL_ARRAY_BUFFER, bv.size() * sizeof(float), bv.data(), GL_STREAM_DRAW);
    glEnableVertexAttribArray((GLuint)g_huPos);
    glVertexAttribPointer((GLuint)g_huPos, 2, GL_FLOAT, GL_FALSE, 8, (void*)0);
    glDrawArrays(GL_TRIANGLES, 0, (GLint)(bv.size() / 2));
    glDisableVertexAttribArray((GLuint)g_huPos);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDeleteBuffers(1, &bbo);
  }
  glUseProgram(g_pip_prog);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, g_pip_tex);
  glUniform1i(g_pip_uTex, 0);
  GLuint vbo;
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
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
  glDeleteBuffers(1, &vbo);
  glBindTexture(GL_TEXTURE_2D, 0);
}

// ---------------- touch buttons + UI state ----------------

int gles_hit_button(float x, float y) {
  const int W = g_win_w, H = g_win_h;
  const float m = 0.018f * W;
  const float gap = 0.014f * W;
  const float bw = (W - 2 * m - 3 * gap) / 4.f;
  const float bh = 0.115f * H;
  const float y0 = H - bh - 0.018f * H;
  for (int i = 0; i < BTN_COUNT; ++i) {
    const float x0 = m + i * (bw + gap);
    if (x >= x0 && x <= x0 + bw && y >= y0 && y <= y0 + bh) return i;
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
  GLuint vbo;
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, v.size() * sizeof(float), v.data(), GL_STREAM_DRAW);
  glEnableVertexAttribArray((GLuint)g_status_pos);
  glVertexAttribPointer((GLuint)g_status_pos, 2, GL_FLOAT, GL_FALSE, 8, (void*)0);
  glDrawArrays(GL_TRIANGLES, 0, (GLint)(v.size() / 2));
  glDisableVertexAttribArray((GLuint)g_status_pos);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glDeleteBuffers(1, &vbo);
}
