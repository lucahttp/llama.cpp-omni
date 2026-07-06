#include "trt_vocoder.h"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>

namespace omni {
namespace vocoder {

static constexpr int kNfft          = 16;
static constexpr int kHop           = 4;
static constexpr int kNumFreqs      = 9;
static constexpr int kPad           = 8;
static constexpr int kMelCh         = 80;
static constexpr int kStftCh        = 18;
static constexpr int kSamplesPerMel = 480;
static constexpr float kClipLimit   = 0.99f;

namespace {

class TrtLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char * msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::fprintf(stderr, "[TRT] %s\n", msg);
    }
};
static TrtLogger g_logger;

} // anonymous namespace

TrtVocoder::TrtVocoder() = default;

TrtVocoder::~TrtVocoder() {
    if (d_mel_)  cudaFree(d_mel_);
    if (d_src_)  cudaFree(d_src_);
    if (d_stft_) cudaFree(d_stft_);
    if (stream_) cudaStreamDestroy(stream_);
    delete ctx_;
    delete engine_;
    delete runtime_;
}

bool TrtVocoder::init(const TrtVocoderConfig & cfg) {
    cfg_ = cfg;

    std::ifstream f(cfg_.engine_path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "[TRT] cannot open engine: %s\n", cfg_.engine_path.c_str());
        return false;
    }
    f.seekg(0, std::ios::end);
    std::streamsize sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> plan((size_t)sz);
    f.read(plan.data(), sz);
    std::fprintf(stderr, "[TRT] loaded engine: %s (%.1f MB)\n",
                 cfg_.engine_path.c_str(), (double)sz / 1e6);

    runtime_ = nvinfer1::createInferRuntime(g_logger);
    if (!runtime_) {
        std::fprintf(stderr, "[TRT] createInferRuntime failed\n");
        return false;
    }
    engine_ = runtime_->deserializeCudaEngine(plan.data(), (size_t)sz);
    if (!engine_) {
        std::fprintf(stderr, "[TRT] deserializeCudaEngine failed\n");
        return false;
    }
    ctx_ = engine_->createExecutionContext();
    if (!ctx_) {
        std::fprintf(stderr, "[TRT] createExecutionContext failed\n");
        return false;
    }

    int n_io = engine_->getNbIOTensors();
    std::fprintf(stderr, "[TRT] engine has %d IO tensors:\n", n_io);
    for (int i = 0; i < n_io; i++) {
        const char * name = engine_->getIOTensorName(i);
        auto mode = engine_->getTensorIOMode(name);
        auto dims = engine_->getTensorShape(name);
        const char * mode_str = (mode == nvinfer1::TensorIOMode::kINPUT) ? "IN" : "OUT";
        std::fprintf(stderr, "[TRT]   [%d] %s (%s) dims=", i, name, mode_str);
        for (int d = 0; d < dims.nbDims; d++)
            std::fprintf(stderr, "%lld ", (long long)dims.d[d]);
        std::fprintf(stderr, "\n");

        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            if (std::strcmp(name, "mel") == 0) {
                mel_name_   = name;
                dynamic_mel_ = (dims.nbDims == 3 && dims.d[2] < 0);
            } else if (std::strcmp(name, "source_stft") == 0) {
                src_name_      = name;
                has_src_input_ = true;
                dynamic_src_   = (dims.nbDims == 3 && dims.d[2] < 0);
            }
        } else {
            stft_name_ = name;
        }
    }

    nvinfer1::Dims mel_shape;
    mel_shape.nbDims = 3;
    mel_shape.d[0] = 1;
    mel_shape.d[1] = kMelCh;
    mel_shape.d[2] = cfg_.max_mel_frames;
    if (!ctx_->setInputShape(mel_name_.c_str(), mel_shape)) {
        std::fprintf(stderr, "[TRT] setInputShape(%s) failed\n", mel_name_.c_str());
        return false;
    }

    src_frames_ = cfg_.max_mel_frames * kSamplesPerMel / kHop + 1;
    if (has_src_input_) {
        nvinfer1::Dims src_shape = engine_->getTensorShape(src_name_.c_str());
        if (!dynamic_src_ && src_shape.nbDims == 3 && src_shape.d[2] > 0)
            src_frames_ = src_shape.d[2];
        src_shape.nbDims = 3;
        src_shape.d[0] = 1;
        src_shape.d[1] = kStftCh;
        src_shape.d[2] = src_frames_;
        if (!ctx_->setInputShape(src_name_.c_str(), src_shape)) {
            std::fprintf(stderr, "[TRT] setInputShape(%s) failed\n", src_name_.c_str());
            return false;
        }
    }

    cudaStreamCreate(&stream_);
    mel_bytes_  = kMelCh  * cfg_.max_mel_frames * sizeof(float);
    src_bytes_  = has_src_input_ ? kStftCh * src_frames_ * sizeof(float) : 0;
    nvinfer1::Dims out_shape = ctx_->getTensorShape(stft_name_.c_str());
    stft_frames_ = has_src_input_ ? src_frames_
                                  : (cfg_.max_mel_frames * kSamplesPerMel / kHop + 1);
    if (out_shape.nbDims == 3 && out_shape.d[2] > 0)
        stft_frames_ = out_shape.d[2];
    stft_bytes_ = kStftCh * stft_frames_ * sizeof(float);

    if (cudaMalloc(&d_mel_, mel_bytes_) != cudaSuccess) {
        std::fprintf(stderr, "[TRT] cudaMalloc mel failed\n");
        return false;
    }
    if (has_src_input_ && cudaMalloc(&d_src_, src_bytes_) != cudaSuccess) {
        std::fprintf(stderr, "[TRT] cudaMalloc source_stft failed\n");
        return false;
    }
    if (cudaMalloc(&d_stft_, stft_bytes_) != cudaSuccess) {
        std::fprintf(stderr, "[TRT] cudaMalloc stft output failed\n");
        return false;
    }
    if (!ctx_->setTensorAddress(mel_name_.c_str(),  d_mel_)  ||
        (has_src_input_ && !ctx_->setTensorAddress(src_name_.c_str(), d_src_)) ||
        !ctx_->setTensorAddress(stft_name_.c_str(), d_stft_)) {
        std::fprintf(stderr, "[TRT] setTensorAddress failed\n");
        return false;
    }

    window_.resize(kNfft);
    window_sq_.resize(kNfft);
    for (int i = 0; i < kNfft; ++i) {
        window_[i]    = 0.5f - 0.5f * std::cos(2.0f * M_PI * i / (float)kNfft);
        window_sq_[i] = window_[i] * window_[i];
    }

    ready_ = true;
    std::fprintf(stderr, "[TRT] vocoder ready: max_mel_frames=%d, src_frames=%d, stft_frames=%d\n",
                 cfg_.max_mel_frames, src_frames_, stft_frames_);
    return true;
}

bool TrtVocoder::infer(const float * mel_bct, int T_mel,
                        std::vector<float> & wave_bt_out, int64_t & out_T_audio) {
    return infer(mel_bct, T_mel, nullptr, 0, wave_bt_out, out_T_audio);
}

bool TrtVocoder::infer(const float * mel_bct, int T_mel,
                        const float * source_stft_tcb, int T_source_frame,
                        std::vector<float> & wave_bt_out, int64_t & out_T_audio) {
    if (!ready_) return false;
    if (!mel_bct || T_mel <= 0) return false;
    if (T_mel > cfg_.max_mel_frames) {
        std::fprintf(stderr, "[TRT] T_mel=%d exceeds max_mel_frames=%d\n", T_mel, cfg_.max_mel_frames);
        return false;
    }

    const int T_mel_engine = dynamic_mel_ ? T_mel : cfg_.max_mel_frames;
    nvinfer1::Dims mel_shape;
    mel_shape.nbDims = 3;
    mel_shape.d[0] = 1;
    mel_shape.d[1] = kMelCh;
    mel_shape.d[2] = T_mel_engine;
    if (!ctx_->setInputShape(mel_name_.c_str(), mel_shape)) {
        std::fprintf(stderr, "[TRT] setInputShape(%s, T=%d) failed\n", mel_name_.c_str(), T_mel_engine);
        return false;
    }

    int T_src_engine = src_frames_;
    if (has_src_input_ && dynamic_src_) {
        T_src_engine = (source_stft_tcb && T_source_frame > 0)
                           ? T_source_frame
                           : (T_mel_engine * kSamplesPerMel / kHop + 1);
        if (T_src_engine > src_frames_) {
            std::fprintf(stderr, "[TRT] T_source=%d exceeds max=%d\n", T_src_engine, src_frames_);
            return false;
        }
    }
    if (has_src_input_) {
        nvinfer1::Dims src_shape;
        src_shape.nbDims = 3;
        src_shape.d[0] = 1;
        src_shape.d[1] = kStftCh;
        src_shape.d[2] = T_src_engine;
        if (!ctx_->setInputShape(src_name_.c_str(), src_shape)) {
            std::fprintf(stderr, "[TRT] setInputShape(%s, T=%d) failed\n", src_name_.c_str(), T_src_engine);
            return false;
        }
    }

    nvinfer1::Dims out_shape = ctx_->getTensorShape(stft_name_.c_str());
    int T_frame = stft_frames_;
    if (out_shape.nbDims == 3 && out_shape.d[2] > 0)
        T_frame = out_shape.d[2];
    if (T_frame <= 0 || T_frame > stft_frames_) {
        std::fprintf(stderr, "[TRT] invalid output T_frame=%d (max=%d)\n", T_frame, stft_frames_);
        return false;
    }

    const size_t mel_cur_bytes  = kMelCh  * T_mel_engine * sizeof(float);
    const size_t src_cur_bytes  = has_src_input_ ? kStftCh * T_src_engine * sizeof(float) : 0;
    const size_t stft_cur_bytes = kStftCh * T_frame      * sizeof(float);

    std::vector<float> mel_buf(kMelCh * T_mel_engine, 0.0f);
    int T_copy = std::min(T_mel, T_mel_engine);
    for (int c = 0; c < kMelCh; ++c)
        std::memcpy(mel_buf.data() + (size_t)c * T_mel_engine,
                    mel_bct + (size_t)c * T_mel,
                    (size_t)T_copy * sizeof(float));

    cudaMemcpyAsync(d_mel_, mel_buf.data(), mel_cur_bytes, cudaMemcpyHostToDevice, stream_);

    std::vector<float> src_buf;
    if (has_src_input_) {
        if (source_stft_tcb && T_source_frame > 0) {
            src_buf.assign(kStftCh * T_src_engine, 0.0f);
            const int T_src_copy = std::min(T_source_frame, T_src_engine);
            for (int c = 0; c < kStftCh; ++c)
                std::memcpy(src_buf.data() + (size_t)c * T_src_engine,
                            source_stft_tcb + (size_t)c * T_source_frame,
                            (size_t)T_src_copy * sizeof(float));
            cudaMemcpyAsync(d_src_, src_buf.data(), src_cur_bytes, cudaMemcpyHostToDevice, stream_);
        } else {
            cudaMemsetAsync(d_src_, 0, src_cur_bytes, stream_);
        }
    }

    if (!ctx_->enqueueV3(stream_)) {
        std::fprintf(stderr, "[TRT] enqueueV3 failed\n");
        return false;
    }
    cudaError_t cu_err = cudaStreamSynchronize(stream_);
    if (cu_err != cudaSuccess) {
        std::fprintf(stderr, "[TRT] cudaStreamSynchronize error: %s\n", cudaGetErrorString(cu_err));
        return false;
    }

    std::vector<float> stft_buf(kStftCh * T_frame);
    cudaMemcpy(stft_buf.data(), d_stft_, stft_cur_bytes, cudaMemcpyDeviceToHost);

    int T_frame_valid = T_frame;
    if (has_src_input_ && T_source_frame > 0)
        T_frame_valid = std::min(T_frame, T_source_frame);
    else if (T_mel > 0)
        T_frame_valid = std::min(T_frame, (T_mel * kSamplesPerMel) / kHop + 1);
    if (T_frame_valid <= 0)
        return false;

    const int audio_len_full = (T_frame_valid - 1) * kHop + kNfft;
    std::vector<float> overlap_buf((size_t)audio_len_full, 0.0f);
    std::vector<float> window_sum((size_t)audio_len_full, 0.0f);

    for (int frame = 0; frame < T_frame_valid; ++frame) {
        float real_vals[9], imag_vals[9];
        for (int f = 0; f < kNumFreqs; ++f) {
            const float mag_log   = stft_buf[(size_t)f * T_frame + frame];
            const float raw_phase = stft_buf[(size_t)(kNumFreqs + f) * T_frame + frame];

            float mag = std::exp(mag_log);
            mag = std::max(0.0f, std::min(mag, 100.0f));
            if (!std::isfinite(mag))
                mag = (mag_log > 0.0f) ? 100.0f : 0.0f;

            float phase = std::sin(raw_phase);
            if (!std::isfinite(phase))
                phase = 0.0f;
            real_vals[f] = mag * std::cos(phase);
            imag_vals[f] = mag * std::sin(phase);
        }

        const int base = frame * kHop;
        for (int n = 0; n < kNfft; ++n) {
            float v = real_vals[0] + real_vals[kNumFreqs - 1] * ((n & 1) ? -1.0f : 1.0f);
            for (int k = 1; k < kNumFreqs - 1; ++k) {
                const float angle = 2.0f * M_PI * (float)k * (float)n / (float)kNfft;
                v += 2.0f * (real_vals[k] * std::cos(angle) - imag_vals[k] * std::sin(angle));
            }
            overlap_buf[(size_t)base + n]  += (v / (float)kNfft) * window_[n];
            window_sum[(size_t)base + n]   += window_sq_[n];
        }
    }

    const int trim = 2 * kPad;
    if (audio_len_full <= trim)
        return false;

    const int audio_len = audio_len_full - trim;
    wave_bt_out.assign((size_t)audio_len, 0.0f);
    for (int i = 0; i < audio_len; ++i) {
        const int src_i   = i + kPad;
        const float denom = std::max(window_sum[(size_t)src_i], 1e-8f);
        float v = overlap_buf[(size_t)src_i] / denom;
        if (!std::isfinite(v))
            v = 0.0f;
        wave_bt_out[(size_t)i] = std::max(-kClipLimit, std::min(v, kClipLimit));
    }
    out_T_audio = (int64_t)wave_bt_out.size();
    return true;
}

} // namespace vocoder
} // namespace omni
