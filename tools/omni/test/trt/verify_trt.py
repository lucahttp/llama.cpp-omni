#!/usr/bin/env python3
"""GGUF (C++) vs TRT (Python) vision encoder precision verification.

Usage:
  python verify_trt.py --gguf vision.gguf --plan vision.plan --image test.jpg
  python verify_trt.py --gguf vision.gguf --plan vision.plan --image test.jpg --dump-bin ./build/bin/llama-omni-dump-vision-emb
"""
import argparse
import os
import struct
import subprocess
import sys

import numpy as np
import tensorrt as trt
from cuda.bindings import driver

PS, NQ, RD = 14, 64, 4096
MAX_PATCHES = 1024


def main():
    p = argparse.ArgumentParser(description="GGUF vs TRT vision encoder precision check")
    p.add_argument("--gguf", required=True, help="Path to vision GGUF model")
    p.add_argument("--plan", required=True, help="Path to TRT engine (.plan)")
    p.add_argument("--image", required=True, help="Path to test image (JPEG)")
    p.add_argument("--dump-bin", default=None, help="Path to llama-omni-dump-vision-emb binary")
    args = p.parse_args()

    # Auto-find dump binary if not specified
    dump_bin = args.dump_bin
    if not dump_bin:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(script_dir, "../../../build4/bin/llama-omni-dump-vision-emb"),
            os.path.join(script_dir, "../../../build/bin/llama-omni-dump-vision-emb"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                dump_bin = c
                break
    if not dump_bin:
        p.error("cannot find llama-omni-dump-vision-emb; specify --dump-bin")

    # --- 1. GGUF dump ---
    proc = subprocess.run([dump_bin, args.gguf, args.image],
                          capture_output=True, timeout=60)
    data = proc.stdout
    H, W = struct.unpack_from("<II", data, 0)
    n_pixels = 3 * H * W
    prep = np.frombuffer(data, offset=8, count=n_pixels, dtype=np.float32).copy()
    gguf_emb = np.frombuffer(data, offset=8 + n_pixels * 4,
                             count=NQ * RD, dtype=np.float32).copy()
    print(f"GGUF: {H}x{W} -> {gguf_emb.size} floats, L2={float(np.linalg.norm(gguf_emb)):.3f}")

    # --- 2. Build patch-layout pixel_values ---
    pos_w, pos_h = W // PS, H // PS
    nx = MAX_PATCHES * PS  # 14336
    ny = PS                 # 14
    inp_raw = np.zeros((3, ny, nx), dtype=np.float32)
    prep_3d = prep.reshape(3, H, W)

    patch_idx = 0
    for i in range(0, H, PS):
        for j in range(0, W, PS):
            if patch_idx >= MAX_PATCHES:
                break
            for pi in range(PS):
                for pj in range(PS):
                    inp_raw[:, pi, patch_idx * PS + pj] = prep_3d[:, i + pi, j + pj]
            patch_idx += 1

    # --- 3. position_ids ---
    positions = np.zeros(max(pos_h * pos_w, MAX_PATCHES), dtype=np.int32)
    bucket_h = [int(np.floor(70.0 * i / pos_h)) for i in range(pos_h)]
    bucket_w = [int(np.floor(70.0 * i / pos_w)) for i in range(pos_w)]
    for i in range(pos_h):
        for j in range(pos_w):
            positions[i * pos_w + j] = bucket_h[i] * 70 + bucket_w[j]

    # --- 4. pos_embed_2d ---
    def get_2d_sincos(ed, gh, gw):
        gm = np.meshgrid(np.arange(gw, dtype=np.float32),
                         np.arange(gh, dtype=np.float32))
        def g1d(d, pos):
            omega = np.arange(d // 2, dtype=np.float32) / (d / 2)
            omega = 1.0 / 10000**omega
            out = np.einsum("hw,d->hwd", pos, omega)
            return np.concatenate([np.sin(out), np.cos(out)], axis=-1)
        return np.concatenate([g1d(ed // 2, gm[0]), g1d(ed // 2, gm[1])], axis=-1)

    pe = get_2d_sincos(RD, pos_h, pos_w).reshape(pos_h * pos_w, RD)
    pos_embed = np.zeros(max(pos_h * pos_w, MAX_PATCHES) * RD, dtype=np.float32)
    for i in range(pos_h * pos_w):
        pos_embed[i * RD:(i + 1) * RD] = pe[i]

    # --- 5. TRT inference ---
    driver.cuInit(0)
    _, cu_ctx = driver.cuCtxCreate(None, 0, 0)

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(args.plan, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    ec = engine.create_execution_context()

    inp_4d = inp_raw.reshape(1, 3, ny, nx)
    pid_2d = positions[:MAX_PATCHES].reshape(1, -1)
    pe_3d = pos_embed[:MAX_PATCHES * RD].reshape(1, MAX_PATCHES, RD)

    _, d_in = driver.cuMemAlloc(inp_4d.nbytes)
    _, d_pid = driver.cuMemAlloc(pid_2d.nbytes)
    _, d_pe = driver.cuMemAlloc(pe_3d.nbytes)
    _, d_out = driver.cuMemAlloc(NQ * RD * 4)

    driver.cuMemcpyHtoD(d_in, inp_4d.ctypes.data, inp_4d.nbytes)
    driver.cuMemcpyHtoD(d_pid, pid_2d.ctypes.data, pid_2d.nbytes)
    driver.cuMemcpyHtoD(d_pe, pe_3d.ctypes.data, pe_3d.nbytes)

    ec.set_tensor_address("pixel_values", d_in)
    ec.set_tensor_address("position_ids", d_pid)
    ec.set_tensor_address("pos_embed_2d", d_pe)
    ec.set_tensor_address("f", d_out)

    _, stream = driver.cuStreamCreate(0)
    ec.execute_async_v3(stream)
    driver.cuStreamSynchronize(stream)

    trt_emb = np.empty(NQ * RD, dtype=np.float32)
    driver.cuMemcpyDtoH(trt_emb.ctypes.data, d_out, NQ * RD * 4)

    driver.cuMemFree(d_in)
    driver.cuMemFree(d_pid)
    driver.cuMemFree(d_pe)
    driver.cuMemFree(d_out)
    driver.cuStreamDestroy(stream)
    driver.cuCtxDestroy(cu_ctx)

    # --- 6. Compare ---
    cos = float(np.dot(gguf_emb, trt_emb) /
                (np.linalg.norm(gguf_emb) * np.linalg.norm(trt_emb) + 1e-12))
    diff = np.abs(gguf_emb - trt_emb)
    mae = float(np.mean(diff))

    print(f"\n{'=' * 60}")
    print(f"  GGUF vs TRT precision")
    print(f"{'=' * 60}")
    print(f"  Cosine similarity:   {cos:.10f}")
    print(f"  MAE:                 {mae:.8f}")
    print(f"  GGUF L2:             {float(np.linalg.norm(gguf_emb)):.3f}")
    print(f"  TRT  L2:             {float(np.linalg.norm(trt_emb)):.3f}")
    print(f"  GGUF[:4] = {[float(x) for x in gguf_emb[:4]]}")
    print(f"  TRT [:4] = {[float(x) for x in trt_emb[:4]]}")

    sd = np.sort(diff)
    for q in [50, 90, 95, 99, 100]:
        idx = min(int(len(sd) * q / 100), len(sd) - 1)
        print(f"  p{q} AE:              {float(sd[idx]):.8f}")

    if cos > 0.999:
        print(f"\n  PASS (cos={cos:.6f} > 0.999)")
    elif cos > 0.99:
        print(f"\n  MARGINAL (cos={cos:.6f} > 0.99)")
    else:
        print(f"\n  FAIL (cos={cos:.6f} < 0.99)")


if __name__ == "__main__":
    main()
