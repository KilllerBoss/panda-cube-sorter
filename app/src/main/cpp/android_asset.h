// android_asset.h — AAssetManager helpers
#pragma once
#include <android/asset_manager.h>
#include <cstdint>
#include <vector>

inline bool read_asset(AAssetManager* am, const char* name, std::vector<uint8_t>& out) {
  AAsset* a = AAssetManager_open(am, name, AASSET_MODE_BUFFER);
  if (!a) return false;
  const off_t len = AAsset_getLength(a);
  out.resize((size_t)len);
  const void* buf = AAsset_getBuffer(a);
  if (buf) {
    memcpy(out.data(), buf, (size_t)len);
  } else {
    AAsset_read(a, out.data(), (size_t)len);
  }
  AAsset_close(a);
  return true;
}
