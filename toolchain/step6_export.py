#!/usr/bin/env python3
"""toolchain/step6_export.py — assemble weights.bin (binary blob for the app).

Layout:
  u32 magic 'PCSW', u16 version=1, u16 endian=0x4949, u32 num_sections
  per section: char name[8], u32 offset, u32 bytes   (offsets from blob start)
  sections: float32 little-endian payloads, META = 8 floats.

Also writes weights_meta.json for documentation/debug.
"""
import json
import os
import struct
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa


def section(name: str, data: np.ndarray) -> bytes:
    assert len(name) <= 8
    raw = data.astype(np.float32).tobytes()
    return struct.pack("<8sII", name.encode("ascii"), 0, len(raw)), raw


def main():
    init = np.load(os.path.join(HERE, "init_weights.npz"))
    km = np.load(os.path.join(HERE, "kan_compiled.npz"))
    prot = np.load(os.path.join(HERE, "prototypes.npz"))
    heads = np.load(os.path.join(HERE, "heads.npz"))

    secs = []
    secs.append(section("PROTO", prot["P"].reshape(-1)))
    secs.append(section("LSNN_W", init["lsnn_w"].reshape(-1)))
    secs.append(section("LSNN_B", init["lsnn_b"]))
    secs.append(section("LSNN_G", init["lsnn_gain"]))
    secs.append(section("PRED_FB", init["pred_fb"].reshape(-1)))
    secs.append(section("EMB_PROJ", init["emb_proj"].reshape(-1)))
    secs.append(section("SMOE_W", heads["smoe_w"].reshape(-1)))
    secs.append(section("SMOE_B", heads["smoe_b"]))
    secs.append(section("KAN_W1", km["w1"].reshape(-1)))
    secs.append(section("KAN_B1", km["b1"]))
    secs.append(section("KAN_W2", km["w2"].reshape(-1)))
    secs.append(section("KAN_B2", km["b2"]))
    secs.append(section("HINGE_T1", km["hinge_t1"].reshape(-1)))
    secs.append(section("HINGE_T2", km["hinge_t2"].reshape(-1)))
    secs.append(section("DEC_W", heads["dec_w"].reshape(-1)))
    secs.append(section("DEC_B", heads["dec_b"]))
    secs.append(section("LORA_A", init["lora_a"].reshape(-1)))
    secs.append(section("LORA_B", init["lora_b"].reshape(-1)))
    secs.append(section("IN_SCALE", km["in_scale"]))
    secs.append(section("META", init["meta"][:8]))

    header_size = 12 + 16 * len(secs)
    table = b""
    blob_parts = []
    offset = header_size
    for (tbl, raw) in secs:
        name = tbl[:8].decode("ascii").rstrip("\x00")
        size = struct.unpack("<I", tbl[12:16])[0]
        table += struct.pack("<8sII", name.encode("ascii"), offset, size)
        blob_parts.append(raw)
        offset += size

    blob = struct.pack("<4sHHI", b"PCSW", 1, 0x4949, len(secs)) + table + b"".join(blob_parts)

    out = os.path.join(HERE, "..", "weights", "weights.bin")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(blob)
    meta = dict(
        version=1, sections=len(secs), bytes=len(blob),
        proto=[C.NUM_PROTO, C.DOF, C.PROTO_LEN],
        lsnn=[C.LSNN_IN, C.LSNN_N], emb=C.EMB_DIM,
        mlp=[C.MLP_IN, C.MLP_HIDDEN, C.MLP_OUT],
        lora_rank=C.LORA_RANK, dec_out=C.DEC_OUT,
        meta_floats=[float(x) for x in init["meta"][:8]],
        meta_names=["lambda_v", "rho_a", "beta_th", "vth0", "snn_eta",
                    "pred_eta", "out_scale", "reserved"],
    )
    with open(os.path.join(HERE, "..", "weights", "weights_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[step6] weights.bin: {len(blob)} bytes, {len(secs)} sections")


if __name__ == "__main__":
    main()
