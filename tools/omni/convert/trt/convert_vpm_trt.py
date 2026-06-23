#!/usr/bin/env python3
"""Export VisionEncoder (ViT + Resampler) from GGUF to TensorRT engine.

Loads weights from a GGUF vision model, builds a PyTorch model with three
inputs (pixel_values, position_ids, pos_embed_2d), exports it to ONNX,
then compiles a FP16 TensorRT engine.

Usage:
  python vit.py --gguf vision.gguf --plan vision.plan
  GGUF_PATH=vision.gguf PLAN_PATH=vision.plan python vit.py  # alternate
"""

import argparse
import os
import sys
import time
import numpy as np
import onnx
import tensorrt as trt
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add the repo root to sys.path so gguf-py is importable
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.normpath(os.path.join(_script_dir, "../../../.."))
sys.path.insert(0, os.path.join(_repo_root, "gguf-py"))
from gguf import GGUFReader, GGMLQuantizationType


def parse_args():
    p = argparse.ArgumentParser(description="GGUF -> TensorRT engine exporter")
    p.add_argument("--gguf", default=os.environ.get("GGUF_PATH", ""),
                   help="Path to vision GGUF model")
    p.add_argument("--onnx", default=os.environ.get("ONNX_PATH", "/tmp/vit.onnx"),
                   help="Intermediate ONNX output path")
    p.add_argument("--plan", default=os.environ.get("PLAN_PATH", "/tmp/vit.plan"),
                   help="Output TRT engine path (.plan)")
    p.add_argument("--trt-lib-dir", default=os.environ.get("TRT_LIB_DIR", ""),
                   help="Path to directory containing libnvinfer.so")
    args = p.parse_args()
    if not args.gguf:
        p.error("--gguf is required (or set GGUF_PATH env var)")
    return args


def main():
    args = parse_args()

    if args.trt_lib_dir:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{args.trt_lib_dir}:{existing}"

    # ---- Load GGUF ----
    print(f"Loading GGUF: {args.gguf}")
    reader = GGUFReader(args.gguf)

    def load_tensor(t):
        shape = [int(x) for x in t.shape]
        dt = np.float16 if t.tensor_type != GGMLQuantizationType.F32 else np.float32
        arr = np.frombuffer(t.data, dt).astype(np.float32)
        return torch.from_numpy(arr.reshape(shape) if len(shape) > 1 else arr)

    W = {t.name: load_tensor(t) for t in reader.tensors}
    print(f"  {len(W)} tensors loaded")

    # ---- Model constants ----
    IMG_H, IMG_W = 14, 14336          # patch-layout input
    PS = 14                            # patch_size
    NP = IMG_W // PS                    # num_patches = 1024
    D, FF = 1152, 4304                 # hidden_size, intermediate_size
    VIT_NH, NL = 16, 27                # num_attention_heads, num_hidden_layers
    VIT_DH = D // VIT_NH
    RD, NQ = 4096, 64                  # resampler embed_dim, num_queries
    RS_NH = 32                         # resampler heads
    RS_DH = RD // RS_NH
    EP = 1e-6

    def param(t):
        return nn.Parameter(t)

    # ---- ViT Layer ----
    class ViTLayer(nn.Module):
        def __init__(self, idx):
            super().__init__()
            p = f"v.blk.{idx}."
            self.q  = param(W[p + "attn_q.weight"]);  self.qb = param(W[p + "attn_q.bias"])
            self.k  = param(W[p + "attn_k.weight"]);  self.kb = param(W[p + "attn_k.bias"])
            self.v  = param(W[p + "attn_v.weight"]);  self.vb = param(W[p + "attn_v.bias"])
            self.o  = param(W[p + "attn_out.weight"]); self.ob = param(W[p + "attn_out.bias"])
            self.l1w = param(W[p + "ln1.weight"]);     self.l1b = param(W[p + "ln1.bias"])
            self.l2w = param(W[p + "ln2.weight"]);     self.l2b = param(W[p + "ln2.bias"])
            self.fu = param(W[p + "ffn_up.weight"]);   self.fd = param(W[p + "ffn_down.weight"])
        def forward(self, x):
            B, T, _ = x.shape
            y = F.layer_norm(x, (D,), self.l1w, self.l1b, EP)
            q = F.linear(y, self.q, self.qb).view(B, T, VIT_NH, VIT_DH).transpose(1, 2)
            k = F.linear(y, self.k, self.kb).view(B, T, VIT_NH, VIT_DH).transpose(1, 2)
            v = F.linear(y, self.v, self.vb).view(B, T, VIT_NH, VIT_DH).transpose(1, 2)
            x = x + F.linear(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, T, D), self.o, self.ob)
            y = F.layer_norm(x, (D,), self.l2w, self.l2b, EP)
            h = F.gelu(F.linear(y, self.fu), approximate="tanh")
            return x + F.linear(h, self.fd)

    # ---- VPM (ViT) ----
    class VPM(nn.Module):
        def __init__(self):
            super().__init__()
            pw = W["v.patch_embd.weight"]
            self.pc = nn.Conv2d(3, D, PS, stride=PS, bias=True)
            self.pc.weight = param(pw.permute(3, 2, 0, 1))
            self.pc.bias   = param(W["v.patch_embd.bias"])
            self.pos_embd   = param(W["v.position_embd.weight"].T.contiguous())
            self.layers     = nn.ModuleList([ViTLayer(i) for i in range(NL)])
            self.plw = param(W["v.post_ln.weight"])
            self.plb = param(W["v.post_ln.bias"])
        def forward(self, img, position_ids):
            x = self.pc(img).flatten(2).transpose(1, 2)
            x = x + F.embedding(position_ids.long(), self.pos_embd)
            for layer in self.layers:
                x = layer(x)
            return F.layer_norm(x, (D,), self.plw, self.plb, EP)

    # ---- Resampler ----
    class Resampler(nn.Module):
        def __init__(self):
            super().__init__()
            self.kv   = param(W["resampler.kv.weight"].T)
            self.lk_w = param(W["resampler.ln_kv.weight"]);  self.lk_b = param(W["resampler.ln_kv.bias"])
            self.query = param(W["resampler.query"].T)
            self.lq_w = param(W["resampler.ln_q.weight"]);   self.lq_b = param(W["resampler.ln_q.bias"])
            self.qw = param(W["resampler.attn.q.weight"]);      self.qb = param(W["resampler.attn.q.bias"])
            self.kw = param(W["resampler.attn.k.weight"]);      self.kb = param(W["resampler.attn.k.bias"])
            self.vw = param(W["resampler.attn.v.weight"]);      self.vb = param(W["resampler.attn.v.bias"])
            self.ow = param(W["resampler.attn.out.weight"]);    self.ob = param(W["resampler.attn.out.bias"])
            self.lp_w = param(W["resampler.ln_post.weight"]);   self.lp_b = param(W["resampler.ln_post.bias"])
            self.pw = param(W["resampler.proj.weight"])
        def forward(self, x, pos_embed_2d):
            B = x.size(0)
            kv_base = F.layer_norm(F.linear(x, self.kv), (RD,), self.lk_w, self.lk_b, EP)
            k = kv_base + pos_embed_2d
            v = kv_base
            q = F.layer_norm(self.query[:NQ].unsqueeze(0).expand(B, -1, -1), (RD,), self.lq_w, self.lq_b, EP)
            Q = F.linear(q, self.qw, self.qb).view(B, -1, RS_NH, RS_DH).transpose(1, 2)
            K = F.linear(k, self.kw, self.kb).view(B, -1, RS_NH, RS_DH).transpose(1, 2)
            V = F.linear(v, self.vw, self.vb).view(B, -1, RS_NH, RS_DH).transpose(1, 2)
            x = F.linear(F.scaled_dot_product_attention(Q, K, V).transpose(1, 2).reshape(B, -1, RD), self.ow, self.ob)
            return F.linear(F.layer_norm(x, (RD,), self.lp_w, self.lp_b, EP), self.pw)

    # ---- Full model ----
    class VisionEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.vpm = VPM()
            self.rs  = Resampler()
        def forward(self, pixel_values, position_ids, pos_embed_2d):
            return self.rs(self.vpm(pixel_values, position_ids), pos_embed_2d)

    model = VisionEncoder().eval().cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params / 1e6:.0f}M params")

    # ---- Warmup & benchmark (PyTorch) ----
    K = 50
    inp_img = torch.randn(1, 3, IMG_H, IMG_W, device="cuda")
    inp_pid = torch.zeros(1, NP, dtype=torch.int32, device="cuda")
    inp_pe  = torch.zeros(1, NP, RD, device="cuda")

    with torch.no_grad():
        output = model(inp_img, inp_pid, inp_pe)
        print(f"  Input  {list(inp_img.shape)} + pos_ids + pe")
        print(f"  Output {list(output.shape)}")
        for _ in range(5):
            model(inp_img, inp_pid, inp_pe)
    torch.cuda.synchronize()

    pt_ms = []
    with torch.no_grad():
        for _ in range(K):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(inp_img, inp_pid, inp_pe)
            torch.cuda.synchronize()
            pt_ms.append((time.perf_counter() - t0) * 1000)
    print(f"PT FP32: {np.mean(pt_ms):.1f} ms")

    # ---- Export ONNX ----
    print("Exporting ONNX...")
    torch.onnx.export(model, (inp_img, inp_pid, inp_pe), args.onnx,
                      input_names=["pixel_values", "position_ids", "pos_embed_2d"],
                      output_names=["f"], opset_version=18)
    onnx.save(onnx.load(args.onnx), args.onnx, save_as_external_data=False)
    print(f"  {os.path.getsize(args.onnx) / 1e6:.1f} MB")

    # ---- Build TRT engine ----
    TRT_LOG = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOG)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOG)
    with open(args.onnx, "rb") as f:
        parser.parse(f.read())

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    profile.set_shape("pixel_values",  (1, 3, IMG_H, IMG_W), (1, 3, IMG_H, IMG_W), (1, 3, IMG_H, IMG_W))
    profile.set_shape("position_ids",  (1, NP),               (1, NP),               (1, NP))
    profile.set_shape("pos_embed_2d",  (1, NP, RD),           (1, NP, RD),           (1, NP, RD))
    config.add_optimization_profile(profile)

    print("Building TRT engine...")
    serialized = builder.build_serialized_network(network, config)
    assert serialized, "TRT build failed"
    with open(args.plan, "wb") as f:
        f.write(serialized)
    plan_mb = os.path.getsize(args.plan) / 1e6
    print(f"  {plan_mb:.1f} MB")

    # ---- Verify PT vs TRT ----
    print("Verifying...")
    runtime = trt.Runtime(TRT_LOG)
    engine = runtime.deserialize_cuda_engine(serialized)
    exec_ctx = engine.create_execution_context()

    d_img  = torch.zeros(inp_img.numel(),  dtype=torch.float32, device="cuda")
    d_pid  = torch.zeros(inp_pid.numel(),  dtype=torch.int32,   device="cuda")
    d_pe   = torch.zeros(inp_pe.numel(),   dtype=torch.float32, device="cuda")
    d_out  = torch.zeros(output.numel(),   dtype=torch.float32, device="cuda")

    exec_ctx.set_tensor_address("pixel_values",  d_img.data_ptr())
    exec_ctx.set_tensor_address("position_ids",  d_pid.data_ptr())
    exec_ctx.set_tensor_address("pos_embed_2d",  d_pe.data_ptr())
    exec_ctx.set_tensor_address("f",             d_out.data_ptr())

    d_img.copy_(inp_img.flatten().view_as(d_img))
    d_pid.copy_(inp_pid.flatten().view_as(d_pid))
    d_pe.copy_(inp_pe.flatten().view_as(d_pe))

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        exec_ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    trt_out = d_out.cpu().numpy()
    pt_out  = output.cpu().numpy().flatten()
    cos = float(np.dot(pt_out, trt_out) / (np.linalg.norm(pt_out) * np.linalg.norm(trt_out) + 1e-12))
    print(f"  PT vs TRT cos: {cos:.10f}")

    # ---- TRT benchmark ----
    trt_ms = []
    for _ in range(K):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        exec_ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        trt_ms.append((time.perf_counter() - t0) * 1000)

    print(f"\n  ViT+Resampler  {n_params / 1e6:.1f}M params")
    print(f"  Engine         {plan_mb:.1f} MB  FP16")
    print(f"  PT  FP32       {np.mean(pt_ms):.1f} ms")
    print(f"  TRT FP16       {np.mean(trt_ms):.1f} ms")
    print(f"  Speedup        {np.mean(pt_ms) / np.mean(trt_ms):.1f}x")


if __name__ == "__main__":
    main()
