# Running MiniCPM-o 4.5 Full-Duplex Voice on Windows with AMD ROCm / HIP

This guide details how to build and run **MiniCPM-o 4.5** with real-time **bidirectional Full-Duplex audio conversation** on Windows natively using **AMD Radeon GPUs** (RX 6000 series `gfx1030`, RX 7000 series `gfx1100`, etc.) via **ROCm 7.x / HIP**.

---

## 📑 Architecture Overview

The system consists of three coordinated layers:

1. **`llama-omni-server` (C++ Backend):**
   - High-performance LLM streaming inference (GGUF Q4_K_M).
   - Native hardware-accelerated **Audition APM** audio encoder.
   - **TTS Flow Model** + **Token2Wav Vocoder** for real-time speech synthesis.
   - Parallel pipeline (`encoder_thread`, `llm_thread`, `tts_thread`, `t2w_thread`).
2. **`worker.py` & `gateway.py` (MiniCPM-o-Demo):**
   - Session management, KV Cache sliding window preservation.
   - WebSocket streaming router (`/ws/duplex`).
   - Reference audio & voice cloning manager (`/api/default_ref_audio`, `/api/presets`).
3. **Web Frontend:**
   - Real-time Audio Duplex Interface (`https://localhost:8006/audio-duplex/audio_duplex.html`).
   - Live microphone waveform, dynamic latency control, voice presets.

---

## 🛠️ Prerequisites

1. **Operating System:** Windows 10 / 11 (64-bit).
2. **GPU:** AMD Radeon RX 6000 / 7000 / 8000 series (e.g. RX 6800 XT `gfx1030`, RX 7900 XTX `gfx1100`) with >= 12-16 GB VRAM.
3. **Software Toolchain:**
   - **Visual Studio 2022** (Desktop development with C++ & Windows 10/11 SDK).
   - **CMake** (>= 3.21) and **Ninja** (`pip install ninja`).
   - **Python** (3.10 - 3.12).
   - **ROCm / HIP SDK for Windows**: Official AMD ROCm TheRock tarball for Windows gfx103X (RDNA2): [therock-dist-windows-gfx103X-all-7.14.0.tar.gz](https://repo.amd.com/rocm/tarball-multi-arch/therock-dist-windows-gfx103X-all-7.14.0.tar.gz) (or via [lemonade-sdk/llamacpp-rocm](https://github.com/lemonade-sdk/llamacpp-rocm)).

---

## 🔧 Essential C++ Fixes in `llama.cpp-omni`

When building `llama.cpp-omni` on Windows for real-time duplex streaming, several fixes are required:

1. **Windows Sockets & SSL (`tools/server/server-omni.cpp`):**
   - Initialize Winsock with `WSAStartup(MAKEWORD(2, 2), &wsaData)` in `main()`.
   - Use plain `httplib::Server` when SSL certificate paths are empty to avoid aborting.

2. **Duplex Mutex Deadlock Elimination (`tools/server/server-omni.cpp`):**
   - Removed broad `state.octx_mutex` locks around `stream_decode()` and `stream_prefill()`.
   - Thread safety is internally managed by `dup->llm_mtx`, `dup->encoder_mtx`, and `text_mtx`. Removing the outer transport lock allows concurrent audio ingestion and text/speech decoding.

3. **Duplex Prefill Routing & Nullptr Guard (`tools/omni/omni.cpp`):**
   - Route all duplex prefill requests (including `index = 0`) to `duplex_prefill(...)` when `system_prompt_initialized == true`.
   - Guard `ctx_omni->llm_thread_info` access in simplex fallback to prevent null pointer dereference.

4. **ROCm Device Library Paths (`ggml/src/ggml-hip/CMakeLists.txt`):**
   - Pass `--rocm-path` and `--rocm-device-lib-path=${ROCM_PATH}/lib/llvm/amdgcn/bitcode` to `ggml-hip` compiler options.

---

## ⚡ Building `llama-omni-server` with ROCm / HIP

Open **PowerShell** and execute:

```powershell
# 1. Set ROCm paths (adjust to your extracted TheRock ROCm SDK directory)
$env:ROCM_PATH = "C:\Users\lucas\.cache\lemonade\bin\therock\gfx103X-7.14.0"
$env:HIP_PATH  = "C:\Users\lucas\.cache\lemonade\bin\therock\gfx103X-7.14.0"
$env:PATH      = "$env:ROCM_PATH\bin;$env:ROCM_PATH\lib\llvm\bin;" + $env:PATH

cd C:\Users\lucas\llama.cpp-omni

# 2. Configure with CMake & Ninja
cmake -B build_hip -G "Ninja" `
  -DCMAKE_BUILD_TYPE=Release `
  -DGGML_HIP=ON `
  -DAMDGPU_TARGETS="gfx1030" `
  -DCMAKE_C_COMPILER="$env:ROCM_PATH/lib/llvm/bin/clang.exe" `
  -DCMAKE_CXX_COMPILER="$env:ROCM_PATH/lib/llvm/bin/clang++.exe" `
  -DCMAKE_RC_COMPILER="C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/rc.exe"

# 3. Compile the server executable
ninja -C build_hip llama-omni-server -j 8
```

This outputs `build_hip/bin/llama-omni-server.exe`, `build_hip/bin/ggml-hip.dll`, and `build_hip/bin/omni.dll`.

---

## 📦 Setting Up `MiniCPM-o-Demo`

### 1. Install Python Dependencies

```powershell
pip install fastapi uvicorn websockets httpx numpy soundfile librosa pyyaml ninja
```

### 2. Configure `config.json`

Edit `MiniCPM-o-Demo/config.json`:

```json
{
  "backend": "cpp",
  "llamacpp_root": "C:/Users/lucas/llama.cpp-omni",
  "model_dir": "C:/Users/lucas/.comni/models/MiniCPM-o-4_5-gguf",
  "llm_model": "MiniCPM-o-4_5-Q4_K_M.gguf",
  "ctx_size": 4096,
  "n_gpu_layers": 99,
  "gateway_port": 8006,
  "playback_delay_ms": 200
}
```

---

## 🚀 Running the Services

### Terminal 1: Start Worker (GPU Backend)
```powershell
cd C:\Users\lucas\MiniCPM-o-Demo
python worker.py --port 22400
```
*Look for: `vision using ROCm0 backend` and `flowGGUFModelLoader: backend=ROCm0`.*

### Terminal 2: Start Gateway
```powershell
cd C:\Users\lucas\MiniCPM-o-Demo
python gateway.py
```

---

## 🎙️ Using Full-Duplex Audio in Browser

1. Open **`https://localhost:8006/audio_duplex`** (or `https://localhost:8006/audio-duplex/audio_duplex.html`).
2. Accept the self-signed HTTPS certificate (required for browser microphone capture).
3. Choose a voice preset:
   - **English Call:** Uses `ref_en_dlc_1.wav` for natural English conversational tone.
   - **中文通话:** Standard Chinese assistant voice.
   - **Advanced ▴:** Enter a custom system prompt or upload your own `.wav` sample for instant voice cloning.
4. Click **Start** and begin speaking!
