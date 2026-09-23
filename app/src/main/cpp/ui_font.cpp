// ui_font.cpp — stb_truetype based Roboto renderer for the PCS HUD
#include "ui_font.h"
#include <android/log.h>
#include <cmath>
#include <cstring>
#include <mutex>

#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, "panda-sorter", __VA_ARGS__)

#define STB_TRUETYPE_IMPLEMENTATION
#define STBTT_STATIC
#include "third_party/stb/stb_truetype.h"

namespace uifont {

static const int kAtlasW = 1024;
static const int kAtlasH = 1024;
static const int kGlyphPx = 96;    // rasterized em size in the atlas

// codepoints: ASCII + German + a few symbols
static const uint32_t kCodepoints[] = {
    0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x2B,
    0x2C, 0x2D, 0x2E, 0x2F, 0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,
    0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F, 0x40, 0x41, 0x42, 0x43,
    0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x4B, 0x4C, 0x4D, 0x4E, 0x4F,
    0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x5B,
    0x5C, 0x5D, 0x5E, 0x5F, 0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67,
    0x68, 0x69, 0x6A, 0x6B, 0x6C, 0x6D, 0x6E, 0x6F, 0x70, 0x71, 0x72, 0x73,
    0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x7B, 0x7C, 0x7D, 0x7E,
    0xC4, 0xD6, 0xDC, 0xDF, 0xE4, 0xF6, 0xFC, 0xDF, 0xB0, 0xB5, 0x2022, 0x2026, 0x2014, 0x2192};
static const int kNumCp = (int)(sizeof(kCodepoints) / sizeof(kCodepoints[0]));

struct Glyph {
  float u0, v0, u1, v1;   // atlas UV
  float w, h;             // size in atlas px
  float xoff, yoff;       // bearing at kGlyphPx
  float advance;          // advance at kGlyphPx
};
static Glyph g_glyphs[kNumCp];
static stbtt_fontinfo g_font;
static uint8_t* g_ttf = nullptr;
static GLuint g_tex = 0;
static bool g_ready = false;
static float g_atlas_scale = 1.f;   // font units -> atlas px
static std::mutex g_mtx;

// Roboto cap height / em ratio ~0.711; we treat "pixel height px" as the CAP
// height (matches the visual size of the old pixel font rows).
static const float kCapEm = 0.711f;

static int cp_index(uint32_t cp) {
  for (int i = 0; i < kNumCp; ++i)
    if (kCodepoints[i] == cp) return i;
  return -1;
}

// UTF-8 -> codepoint (1 or 2 byte sequences; ASCII fast path)
static uint32_t next_cp(const char*& p) {
  const uint8_t c = (uint8_t)*p++;
  if (c < 0x80) return c;
  if ((c & 0xE0) == 0xC0) {
    const uint8_t c2 = (uint8_t)*p++;
    return ((uint32_t)(c & 0x1F) << 6) | (c2 & 0x3F);
  }
  if ((c & 0xF0) == 0xE0) {
    const uint8_t c2 = (uint8_t)*p++;
    const uint8_t c3 = (uint8_t)*p++;
    return ((uint32_t)(c & 0x0F) << 12) | ((uint32_t)(c2 & 0x3F) << 6)
         | (c3 & 0x3F);
  }
  return 0x3F;  // '?'
}

bool init(AAssetManager* am) {
  std::lock_guard<std::mutex> lk(g_mtx);
  if (g_ready) return true;
  if (!am) return false;

  // ---- load Regular (+ Medium for later use) from assets ----
  AAsset* a = AAssetManager_open(am, "fonts/UiFont-Regular.ttf",
                                 AASSET_MODE_BUFFER);
  if (!a) {
    LOGW("ui_font: fonts/UiFont-Regular.ttf missing — pixel font fallback");
    return false;
  }
  const off_t len = AAsset_getLength(a);
  g_ttf = new uint8_t[len];
  memcpy(g_ttf, AAsset_getBuffer(a), (size_t)len);
  AAsset_close(a);

  if (!stbtt_InitFont(&g_font, g_ttf, 0)) {
    LOGW("ui_font: stbtt_InitFont failed");
    delete[] g_ttf; g_ttf = nullptr;
    return false;
  }

  // ---- rasterize the atlas (R8) ----
  std::vector<uint8_t> atlas((size_t)kAtlasW * kAtlasH, 0);
  const float scale = stbtt_ScaleForPixelHeight(&g_font, (float)kGlyphPx);
  g_atlas_scale = scale;
  int pen_x = 1, pen_y = 1, row_h = 0;
  for (int i = 0; i < kNumCp; ++i) {
    int w = 0, h = 0, xoff = 0, yoff = 0;
    uint8_t* bmp = stbtt_GetCodepointBitmap(&g_font, 0, scale,
                                            (int)kCodepoints[i], &w, &h,
                                            &xoff, &yoff);
    if (pen_x + w + 1 > kAtlasW) {   // next row
      pen_x = 1;
      pen_y += row_h + 1;
      row_h = 0;
    }
    if (!bmp || pen_y + h + 1 > kAtlasH) {  // out of atlas space: skip
      if (bmp) STBTT_free(bmp, nullptr);
      continue;
    }
    uint8_t* dst = atlas.data() + (size_t)pen_y * kAtlasW + pen_x;
    for (int r = 0; r < h; ++r)
      memcpy(dst + (size_t)r * kAtlasW, bmp + (size_t)r * w, (size_t)w);
    Glyph& gl = g_glyphs[i];
    gl.u0 = (float)pen_x / kAtlasW;             gl.v0 = (float)pen_y / kAtlasH;
    gl.u1 = (float)(pen_x + w) / kAtlasW;       gl.v1 = (float)(pen_y + h) / kAtlasH;
    gl.w = (float)w; gl.h = (float)h;
    gl.xoff = (float)xoff; gl.yoff = (float)yoff;
    int adv = 0, lsb = 0;
    stbtt_GetCodepointHMetrics(&g_font, (int)kCodepoints[i], &adv, &lsb);
    gl.advance = (float)adv * scale;
    STBTT_free(bmp, nullptr);
    pen_x += w + 1;
    if (h > row_h) row_h = h;
  }

  // ---- upload ----
  glGenTextures(1, &g_tex);
  glBindTexture(GL_TEXTURE_2D, g_tex);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, kAtlasW, kAtlasH, 0, GL_RED,
               GL_UNSIGNED_BYTE, atlas.data());
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glBindTexture(GL_TEXTURE_2D, 0);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

  g_ready = true;
  LOGW("ui_font: Roboto atlas ready (%d glyphs, %dx%d)", kNumCp, kAtlasW,
       kAtlasH);
  return true;
}

void shutdown() {
  if (g_tex) glDeleteTextures(1, &g_tex);
  g_tex = 0;
  delete[] g_ttf; g_ttf = nullptr;
  g_ready = false;
}

bool ready() { return g_ready; }
GLuint atlas_texture() { return g_tex; }
int atlas_glyph_px() { return kGlyphPx; }

float text(std::vector<float>& out, const char* s, float x, float y, float px,
           int win_w, int win_h) {
  if (!g_ready || !s || !*s) return 0.f;
  const float k = px / ((float)kGlyphPx * kCapEm);  // atlas px -> screen px
  float pen = x;
  const char* p = s;
  uint32_t prev = 0;
  while (*p) {
    const uint32_t cp = next_cp(p);
    int gi = cp_index(cp);
    if (gi < 0) gi = cp_index('?');
    const Glyph& gl = g_glyphs[gi];
    if (prev) {  // kerning (font units -> atlas px -> screen px)
      const int kern = stbtt_GetCodepointKernAdvance(&g_font, (int)prev,
                                                     (int)cp);
      pen += (float)kern * g_atlas_scale * k;
    }
    const float gx = pen + gl.xoff * k;
    const float gy = y + gl.yoff * k;
    const float gw = gl.w * k, gh = gl.h * k;
    const float xa = 2.f * gx / win_w - 1.f;
    const float xb = 2.f * (gx + gw) / win_w - 1.f;
    const float ya = 1.f - 2.f * gy / win_h;
    const float yb = 1.f - 2.f * (gy + gh) / win_h;
    // quad: (x,y,u,v) — v grows downward in the atlas
    out.insert(out.end(), {xa, ya, gl.u0, gl.v0});
    out.insert(out.end(), {xb, ya, gl.u1, gl.v0});
    out.insert(out.end(), {xb, yb, gl.u1, gl.v1});
    out.insert(out.end(), {xa, ya, gl.u0, gl.v0});
    out.insert(out.end(), {xb, yb, gl.u1, gl.v1});
    out.insert(out.end(), {xa, yb, gl.u0, gl.v1});
    pen += gl.advance * k;
    prev = cp;
  }
  return pen - x;
}

float width(const char* s, float px) {
  if (!g_ready || !s || !*s) return 0.f;
  const float k = px / ((float)kGlyphPx * kCapEm);
  float pen = 0.f;
  const char* p = s;
  while (*p) {
    const uint32_t cp = next_cp(p);
    int gi = cp_index(cp);
    if (gi < 0) gi = cp_index('?');
    pen += g_glyphs[gi].advance * k;
  }
  return pen;
}

}  // namespace uifont
