// gles_renderer.cpp — GLES 3.0 renderer: scene view + event-camera FBO + HUD
// Draws the scene as colored primitives (capsules/boxes) directly from
// mjData kinematics — no meshes, no textures, minimal GPU load.
#include "gles_renderer.h"
#include <android/log.h>
#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <android/log.h>
#include <cmath>
#include <cstring>
#include <vector>

#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, "panda-sorter", __VA_ARGS__)

using namespace pcs;

// ---------------- tiny math (column-major mat4) ----------------
struct Mat4 {
  float m[16];
  Mat4 operator*(const Mat4& o) const {
    Mat4 r{};
    for (int row = 0; row < 4; ++row)
      for (int c = 0; c < 4; ++c) {
        float acc = 0;
        for (int k = 0; k < 4; ++k) acc += m[4*row+k] * o.m[4*k+c];
        r.m[4*row+c] = acc;
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

static void quat_to_mat(const double q[4], float R[9]) {
  // MuJoCo quat = (w,x,y,z)
  const float w=(float)q[0],x=(float)q[1],y=(float)q[2],z=(float)q[3];
  R[0]=1-2*(y*y+z*z); R[1]=2*(x*y-w*z); R[2]=2*(x*z+w*y);
  R[3]=2*(x*y+w*z);   R[4]=1-2*(x*x+z*z); R[5]=2*(y*z-w*x);
  R[6]=2*(x*z-w*y);   R[7]=2*(y*z+w*x);   R[8]=1-2*(x*x+y*y);
}

static void body_transform(const mjModel* m, const mjData* d, int body,
                           float M[16]) {
  const mjtNum* p = d->xpos + 3 * body;
  float R[9];
  quat_to_mat(d->xquat + 4 * body, R);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) M[4*i+j] = (float)R[3*i+j];
    M[12+i] = (float)p[i];
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
    "  gl_Position = uMVP * vec4(aPos,1.0); }";
static const char* kFS =
    "precision mediump float;\n"
    "uniform vec4 uColor;\n"
    "varying vec3 vNrm;\n"
    "varying vec4 vPos;\n"
    "void main(){ float d = max(0.35, dot(normalize(vNrm), normalize(vec3(0.4,0.5,0.8))));\n"
    "  gl_FragColor = vec4(uColor.rgb * d, uColor.a); }";

static const char* kHudVS =
    "attribute vec2 aPos;\n"
    "void main(){ gl_Position = vec4(aPos,0.0,1.0); }";
static const char* kHudFS =
    "precision mediump float;\n"
    "uniform vec4 uColor;\n"
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

static GLuint make_program(const char* vs, const char* fs) {
  GLuint v = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(v, 1, &vs, nullptr); glCompileShader(v);
  GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(f, 1, &fs, nullptr); glCompileShader(f);
  GLuint p = glCreateProgram();
  glAttachShader(p, v); glAttachShader(p, f); glLinkProgram(p);
  glDeleteShader(v); glDeleteShader(f);
  return p;
}

static void push_cube(std::vector<float>& v) {
  // 6 faces, normal + 4 positions (triangle strip per face)
  static const float F[6][15] = {
      { 1,0,0,  1,-1,-1, 1,1,-1, 1,1,1, 1,-1,1},
      {-1,0,0, -1,-1,1, -1,1,1, -1,1,-1, -1,-1,-1},
      {0, 1,0,  1,1,-1, -1,1,-1, -1,1,1, 1,1,1},
      {0,-1,0, -1,-1,-1, 1,-1,-1, 1,-1,1, -1,-1,1},
      {0,0, 1,  1,-1,1, -1,-1,1, -1,1,1, 1,1,1},
      {0,0,-1, -1,-1,-1, 1,-1,-1, 1,1,-1, -1,1,-1}};
  for (const auto& f : F)
    for (int i = 0; i < 15; ++i) v.push_back(f[i]);
}

static void push_cylinder(std::vector<float>& v) {
  // unit cylinder along +x, r=1, length 1 centered at origin; 12 segments
  const int N = 12;
  for (int i = 0; i < N; ++i) {
    const float a0 = 2.f * (float)M_PI * i / N, a1 = 2.f * (float)M_PI * (i + 1) / N;
    const float c0 = cosf(a0), s0 = sinf(a0), c1 = cosf(a1), s1 = sinf(a1);
    // side quad (two tris), normal = radial
    v.insert(v.end(), {c0, s0, 0.f,  c1, s1, 0.f,  c1, s1, 1.f});
    v.insert(v.end(), {c0, s0, 0.f,  c1, s1, 1.f,  c0, s0, 1.f});
    // caps (normal ±x)
    v.insert(v.end(), {1.f, 0.f, 0.f,  1.f, 0.f, 0.f,  1.f, 0.f, 0.f});
    v.insert(v.end(), {0.5f, c0, s0,  0.5f, c1, s1,  0.5f, 0.f, 0.f});
    v.insert(v.end(), {-1.f, 0.f, 0.f, -1.f, 0.f, 0.f, -1.f, 0.f, 0.f});
    v.insert(v.end(), {-0.5f, c1, s1, -0.5f, c0, s0, -0.5f, 0.f, 0.f});
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
  g_cyl_count = (int)v.size() / 3;
  glGenBuffers(1, &g_cyl_vbo);
  glBindBuffer(GL_ARRAY_BUFFER, g_cyl_vbo);
  glBufferData(GL_ARRAY_BUFFER, v.size() * 4, v.data(), GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glDisable(GL_DEPTH_TEST);
  LOGW("gles_init done %dx%d", win_w, win_h);
}

// draw one geom (box or capsule) with color
static void draw_geom(const mjModel* m, const mjData* d, int g, float alpha) {
  const int body = m->geom_bodyid[g];
  float M[16];
  body_transform(m, d, body, M);
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
  } else if (type == mjGEOM_PLANE) {
    scale.m[0] = (float)size[0]; scale.m[5] = (float)size[1];
    Mat4 model = Mb * scale;
    glUniformMatrix4fv(g_uModel, 1, GL_FALSE, model.m);
  } else {
    return;
  }
  glUniform4fv(g_uColor, 1, col);
  GLuint vbo = (type == mjGEOM_CAPSULE || type == mjGEOM_CYLINDER) ? g_cyl_vbo : g_cube_vbo;
  int count = (type == mjGEOM_CAPSULE || type == mjGEOM_CYLINDER) ? g_cyl_count : 36;
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
  const mjData* d = glue.data();
  glViewport(0, 0, g_win_w, g_win_h);
  glClearColor(0.09f, 0.10f, 0.12f, 1.f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);

  glUseProgram(g_prog);
  float eye[3] = {1.35f, -1.15f, 0.95f}, ctr[3] = {0.42f, 0.f, 0.30f}, up[3] = {0, 0, 1};
  Mat4 view = Mat4::lookAt(eye, ctr, up);
  Mat4 proj = Mat4::perspective(50.f * (float)M_PI / 180.f,
                                (float)g_win_w / g_win_h, 0.05f, 10.f);
  Mat4 vp = proj * view;
  glUniformMatrix4fv(g_uMVP, 1, GL_FALSE, vp.m);

  // floor + table + zones + arm + hand + cubes (skip invisible geoms)
  for (int g = 0; g < m->ngeom; ++g) {
    if (m->geom_type[g] == mjGEOM_PLANE) {
      draw_geom(m, d, g, 1.f);
    } else if (m->geom_contype[g] != 0 || m->geom_conaffinity[g] != 0) {
      draw_geom(m, d, g, 1.f);
    }
  }
  glDisable(GL_DEPTH_TEST);
  gles_render_hud(g_last_stats, g_last_task);
}

void gles_render_hud(const CycleStats& st, const TaskOutput& task) {
  (void)st; (void)task;
  glUseProgram(g_hud_prog);
  glUniform4f(g_huColor, 1.f, 1.f, 1.f, 1.f);
  // tiny status bar: one triangle strip at the bottom whose width encodes the
  // cycle time fraction (full width = 10 ms budget used)
  const float w = 2.f * ((float)st.t_total / 10000.f);
  float quad[8] = {-1.f, -1.f, -1.f + w, -1.f, -1.f, -0.965f, -1.f + w, -0.965f};
  GLuint vbo;
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_DYNAMIC_DRAW);
  glEnableVertexAttribArray((GLuint)g_huPos);
  glVertexAttribPointer((GLuint)g_huPos, 2, GL_FLOAT, GL_FALSE, 8, (void*)0);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  glDisableVertexAttribArray((GLuint)g_huPos);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glDeleteBuffers(1, &vbo);
}

// ---------------- event-camera FBO pass ----------------
static GLuint g_fbo = 0, g_fbo_tex = 0;
static uint8_t* g_readback = nullptr;

void gles_render_event_frame(SimGlue& glue, bool jitter, uint8_t* rgb, int w, int h) {
  const mjModel* m = glue.model();
  const mjData* d = glue.data();
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
  Mat4 vp;
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) {
      float acc = 0;
      for (int k = 0; k < 4; ++k) acc += proj.m[4*k+c] * view.m[4*r+k];
      vp.m[4*r+c] = acc;
    }
  glUniformMatrix4fv(g_uMVP, 1, GL_FALSE, vp.m);
  for (int g = 0; g < m->ngeom; ++g) {
    if (m->geom_type[g] == mjGEOM_PLANE) continue;
    draw_geom(m, d, g, 1.f);
  }
  glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, g_readback);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  // flip vertically (GL origin bottom-left)
  for (int y = 0; y < h; ++y)
    memcpy(rgb + 3 * y * w, g_readback + 3 * (h - 1 - y) * w, 3 * w);
}

void out_stats_set(const CycleStats& st) { g_last_stats = st; }
void out_task_set(const TaskOutput& t) { g_last_task = t; }
