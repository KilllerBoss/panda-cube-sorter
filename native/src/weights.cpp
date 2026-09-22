// pcs/weights.cpp — binary weight-blob loader
#include "pcs/weights.h"
#include "pcs/event_camera.h"

namespace pcs {
namespace {

struct SecEntry { char name[8]; uint32_t offset; uint32_t bytes; };

uint32_t rd_u32(const uint8_t* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
float rd_f32(const uint8_t* p) { float f; memcpy(&f, p, 4); return f; }

bool expect(const Weights& w, const SecEntry& s, const char* name, size_t n_floats, std::string& err) {
  if (memcmp(s.name, name, 8) != 0) { err = std::string("section mismatch: got '") +
      std::string(s.name, 8) + "' want " + name; return false; }
  if (s.bytes != n_floats * 4) { err = std::string(name) + ": size mismatch"; return false; }
  return true;
}

}  // namespace

bool Weights::load_from_memory(const uint8_t* d, size_t size) {
  last_error.clear();
  if (size < 8) { last_error = "blob too small"; return false; }
  if (memcmp(d, "PCSW", 4) != 0) { last_error = "bad magic"; return false; }
  uint16_t version = (uint16_t)d[4] | ((uint16_t)d[5] << 8);
  if (version != 1) { last_error = "unsupported version"; return false; }
  uint32_t nsec = rd_u32(d + 8);
  const uint8_t* p = d + 12;
  if (size < 12 + (size_t)nsec * 16) { last_error = "truncated table"; return false; }

  std::vector<SecEntry> secs(nsec);
  for (uint32_t i = 0; i < nsec; ++i) {
    memcpy(secs[i].name, p, 8);
    secs[i].offset = rd_u32(p + 8);
    secs[i].bytes  = rd_u32(p + 12);
    p += 16;
    if ((size_t)secs[i].offset + secs[i].bytes > size) { last_error = "section oob"; return false; }
  }

  auto ptr = [&](const SecEntry& s) { return d + s.offset; };

  for (const SecEntry& s : secs) {
    if (memcmp(s.name, "PROTO", 8) == 0) {
      if (!expect(*this, s, "PROTO", (size_t)kNumProto * kDof * kProtoLen, last_error)) return false;
      proto.assign((const float*)ptr(s), (const float*)(ptr(s) + s.bytes));
    } else if (memcmp(s.name, "LSNN_W", 8) == 0) {
      if (!expect(*this, s, "LSNN_W", (size_t)kLsnnIn * kLsnnN, last_error)) return false;
      lsnn_w.assign((const float*)ptr(s), (const float*)(ptr(s) + s.bytes));
    } else if (memcmp(s.name, "LSNN_B", 8) == 0) {
      if (!expect(*this, s, "LSNN_B", kLsnnN, last_error)) return false;
      lsnn_b.assign((const float*)ptr(s), (const float*)(ptr(s) + s.bytes));
    } else if (memcmp(s.name, "LSNN_G", 8) == 0) {
      if (!expect(*this, s, "LSNN_G", kLsnnN, last_error)) return false;
      lsnn_gain.assign((const float*)ptr(s), (const float*)(ptr(s) + s.bytes));
    } else if (memcmp(s.name, "PRED_FB", 8) == 0) {
      if (!expect(*this, s, "PRED_FB", (size_t)kLsnnN * kNumBins, last_error)) return false;
      pred_fb.assign((const float*)ptr(s), (const float*)(ptr(s) + s.bytes));
    } else if (memcmp(s.name, "EMB_PROJ", 8) == 0) {
      if (!expect(*this, s, "EMB_PROJ", (size_t)(kNumBins + kLsnnN) * kEmbDim, last_error)) return false;
      emb_proj.assign((const float*)ptr(s), (const float*)(ptr(s) + s.bytes));
    } else if (memcmp(s.name, "SMOE_W", 8) == 0) {
      if (!expect(*this, s, "SMOE_W", (size_t)kEmbDim * kMlpOut, last_error)) return false;
      smoe_w.assign((const float*)ptr(s), (const float*)(ptr(s) + s.bytes));
    } else if (memcmp(s.name, "SMOE_B", 8) == 0) {
      if (!expect(*this, s, "SMOE_B", kMlpOut, last_error)) return false;
      smoe_b.assign((const float*)ptr(s), (const float*)(ptr(s) + s.bytes));
    } else if (memcmp(s.name, "KAN_W1", 8) == 0) {
      if (!expect(*this, s, "KAN_W1", (size_t)kL1Feats * kMlpHidden, last_error)) return false;
      w1.assign((const float*)ptr(s), (const float*)(ptr(s) + s.bytes));
    } else if (memcmp(s.name, "KAN_B1", 8) == 0) {
      if (!expect(*this, s, "KAN_B1", kMlpHidden, last_error)) return false;
      b1.assign((const float*)ptr(s), (const float*)(ptr(s) + s.bytes));
    } else if (memcmp(s.name, "KAN_W2", 8) == 0) {
      if (!expect(*this, s, "KAN_W2", (size_t)kL2Feats * kMlpOut, last_error)) return false;
      w2.assign((const float*)ptr(s), (const float*)(ptr(s) + s.bytes));
    } else if (memcmp(s.name, "KAN_B2", 8) == 0) {
      if (!expect(*this, s, "KAN_B2", kMlpOut, last_error)) return false;
      b2.assign((const float*)ptr(s), (const float*)(ptr(s) + s.bytes));
    } else if (memcmp(s.name, "HINGE_T1", 8) == 0) {
      if (!expect(*this, s, "HINGE_T1", (size_t)kMlpIn * kHinges1, last_error)) return false;
      hinge_t1.assign((const float*)ptr(s), (const float*)(ptr(s) + s.bytes));
    } else if (memcmp(s.name, "HINGE_T2", 8) == 0) {
      if (!expect(*this, s, "HINGE_T2", (size_t)kMlpHidden * kHinges1, last_error)) return false;
      hinge_t2.assign((const float*)ptr(s), (const float*)(ptr(s) + s.bytes));
    } else if (memcmp(s.name, "DEC_W", 8) == 0) {
      if (!expect(*this, s, "DEC_W", (size_t)kEmbDim * kDecOut, last_error)) return false;
      dec_w.assign((const float*)ptr(s), (const float*)(ptr(s) + s.bytes));
    } else if (memcmp(s.name, "DEC_B", 8) == 0) {
      if (!expect(*this, s, "DEC_B", kDecOut, last_error)) return false;
      dec_b.assign((const float*)ptr(s), (const float*)(ptr(s) + s.bytes));
    } else if (memcmp(s.name, "LORA_A", 8) == 0) {
      if (!expect(*this, s, "LORA_A", (size_t)kMlpHidden * kLoraRank, last_error)) return false;
      lora_a.assign((const float*)ptr(s), (const float*)(ptr(s) + s.bytes));
    } else if (memcmp(s.name, "LORA_B", 8) == 0) {
      if (!expect(*this, s, "LORA_B", (size_t)kLoraRank * kMlpOut, last_error)) return false;
      lora_b.assign((const float*)ptr(s), (const float*)(ptr(s) + s.bytes));
    } else if (memcmp(s.name, "IN_SCALE", 8) == 0) {
      if (!expect(*this, s, "IN_SCALE", kMlpIn, last_error)) return false;
      in_scale.assign((const float*)ptr(s), (const float*)(ptr(s) + s.bytes));
    } else if (memcmp(s.name, "META", 8) == 0) {
      if (s.bytes < 7 * 4) { last_error = "META too small"; return false; }
      const uint8_t* q = ptr(s);
      lambda_v = rd_f32(q); rho_a = rd_f32(q + 4); beta_th = rd_f32(q + 8);
      vth0     = rd_f32(q + 12); snn_eta = rd_f32(q + 16); pred_eta = rd_f32(q + 20);
      out_scale = rd_f32(q + 24);
    }
  }

  // required sections present?
  if (proto.empty() || lsnn_w.empty() || pred_fb.empty() || emb_proj.empty() ||
      w1.empty() || w2.empty() || dec_w.empty() || smoe_w.empty() ||
      hinge_t1.empty() || hinge_t2.empty() || in_scale.empty()) {
    last_error = "missing required sections"; return false;
  }
  return true;
}

bool Weights::load_from_file(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) { last_error = "cannot open weights file"; return false; }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> buf((size_t)n);
  size_t rd = fread(buf.data(), 1, (size_t)n, f);
  fclose(f);
  if (rd != (size_t)n) { last_error = "short read"; return false; }
  return load_from_memory(buf.data(), buf.size());
}

}  // namespace pcs
