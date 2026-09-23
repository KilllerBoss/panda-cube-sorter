// ui_font.h — real TrueType text rendering (Roboto via stb_truetype)
// v1.5.0 "normale Schrift": replaces the 3x5 pixel font for the whole HUD.
// One glyph atlas (1024x1024, ~96 px glyphs, R8) is built once at init; text
// is drawn as UV quads in per-color batches with bilinear filtering.
#pragma once
#include <android/asset_manager.h>
#include <GLES3/gl3.h>
#include <vector>

namespace uifont {

// Load assets/fonts/UiFont-Regular.ttf (+ Medium) and rasterize the glyph
// atlas. Call ONCE with a GL context current (share group). Safe no-op if
// assets are missing (then width()/text() degrade to 0 and the HUD falls
// back to uppercase spacing only).
bool init(AAssetManager* am);
void shutdown();
bool ready();

// Append one UTF-8 text line as NDC quads into `out` (4 floats per vertex:
// ndc_x, ndc_y, u, v; 2 triangles per glyph). y = TOP of the text box.
// px = requested pixel height of the em box. Returns the advance width.
float text(std::vector<float>& out, const char* s, float x, float y, float px,
           int win_w, int win_h);

// advance width of a string at the same pixel height (without drawing)
float width(const char* s, float px);

GLuint atlas_texture();
int atlas_glyph_px();

}  // namespace uifont
