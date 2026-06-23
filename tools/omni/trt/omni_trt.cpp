#include "omni_trt.h"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>

#define TRT_CUDA_CHECK(call)                                                  \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "[TRT] CUDA error at %s:%d: %s (%s)\n",     \
                         __FILE__, __LINE__, cudaGetErrorString(err_), #call);\
            return false;                                                     \
        }                                                                     \
    } while (0)

#define TRT_CUDA_CHECK_WARN(call)                                             \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess)                                              \
            std::fprintf(stderr, "[TRT] CUDA error at %s:%d: %s (%s)\n",     \
                         __FILE__, __LINE__, cudaGetErrorString(err_), #call);\
    } while (0)

#define TRT_CHECK(call, msg)                                                  \
    do {                                                                      \
        if (!(call)) {                                                        \
            std::fprintf(stderr, "[TRT] %s failed at %s:%d\n",               \
                         msg, __FILE__, __LINE__);                            \
            return false;                                                     \
        }                                                                     \
    } while (0)

namespace omni {
namespace vision {

namespace {
class TrtLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char * msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::fprintf(stderr, "[TRT] %s\n", msg);
    }
};
static TrtLogger g_logger;
} // namespace

OmniTrt::OmniTrt() = default;

OmniTrt::~OmniTrt() {
    if (d_img_)  TRT_CUDA_CHECK_WARN(cudaFree(d_img_));
    if (d_pid_)  TRT_CUDA_CHECK_WARN(cudaFree(d_pid_));
    if (d_pe_)   TRT_CUDA_CHECK_WARN(cudaFree(d_pe_));
    if (d_feat_) TRT_CUDA_CHECK_WARN(cudaFree(d_feat_));
    if (stream_) TRT_CUDA_CHECK_WARN(cudaStreamDestroy(stream_));
    delete ctx_;
    delete engine_;
    delete runtime_;
}

bool OmniTrt::init(const char * engine_path) {
    engine_path_ = engine_path;

    std::ifstream f(engine_path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "[TRT] cannot open engine: %s\n", engine_path);
        return false;
    }
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> plan(sz);
    f.read(plan.data(), sz);
    std::fprintf(stderr, "[TRT] loaded engine: %s (%.1f MB)\n", engine_path, sz / 1e6);

    runtime_ = nvinfer1::createInferRuntime(g_logger);
    TRT_CHECK(runtime_, "createInferRuntime");

    engine_ = runtime_->deserializeCudaEngine(plan.data(), sz);
    TRT_CHECK(engine_, "deserializeCudaEngine");

    ctx_ = engine_->createExecutionContext();
    TRT_CHECK(ctx_, "createExecutionContext");

    // Set shapes for all 3 inputs
    // TRT engine expects [1, 3, 14, 14336]
    {
        nvinfer1::Dims img_dims{};
        img_dims.nbDims = 4;
        img_dims.d[0] = 1; img_dims.d[1] = 3;
        img_dims.d[2] = 14; img_dims.d[3] = 14336;
        TRT_CHECK(ctx_->setInputShape("pixel_values", img_dims), "setInputShape pixel_values");
    }
    {
        nvinfer1::Dims pid_dims{};
        pid_dims.nbDims = 2;
        pid_dims.d[0] = 1; pid_dims.d[1] = kNumPatches;
        TRT_CHECK(ctx_->setInputShape("position_ids", pid_dims), "setInputShape position_ids");
    }
    {
        nvinfer1::Dims pe_dims{};
        pe_dims.nbDims = 3;
        pe_dims.d[0] = 1; pe_dims.d[1] = kNumPatches; pe_dims.d[2] = kOutDim;
        TRT_CHECK(ctx_->setInputShape("pos_embed_2d", pe_dims), "setInputShape pos_embed_2d");
    }

    TRT_CUDA_CHECK(cudaStreamCreate(&stream_));

    img_bytes_  = 1 * 3 * 14 * 14336 * sizeof(float);
    pid_bytes_  = 1 * kNumPatches * sizeof(int32_t);
    pe_bytes_   = 1 * kNumPatches * kOutDim * sizeof(float);
    feat_bytes_ = 1 * kNumQueries * kOutDim * sizeof(float);

    TRT_CUDA_CHECK(cudaMalloc(&d_img_,  img_bytes_));
    TRT_CUDA_CHECK(cudaMalloc(&d_pid_,  pid_bytes_));
    TRT_CUDA_CHECK(cudaMalloc(&d_pe_,   pe_bytes_));
    TRT_CUDA_CHECK(cudaMalloc(&d_feat_, feat_bytes_));

    ready_ = true;
    std::fprintf(stderr, "[TRT] vision ready, img=14x14336, pos=%d, out=%dx%d\n",
                 kNumPatches, kNumQueries, kOutDim);
    return true;
}

bool OmniTrt::run(const float * pixel_values, const int32_t * position_ids,
                    const float * pos_embed_2d, float * features_out) {
    if (!ready_) return false;

    TRT_CUDA_CHECK(cudaMemcpyAsync(d_img_, pixel_values, img_bytes_,
                                    cudaMemcpyHostToDevice, stream_));
    TRT_CUDA_CHECK(cudaMemcpyAsync(d_pid_, position_ids, pid_bytes_,
                                    cudaMemcpyHostToDevice, stream_));
    TRT_CUDA_CHECK(cudaMemcpyAsync(d_pe_, pos_embed_2d, pe_bytes_,
                                    cudaMemcpyHostToDevice, stream_));

    ctx_->setTensorAddress("pixel_values", d_img_);
    ctx_->setTensorAddress("position_ids", d_pid_);
    ctx_->setTensorAddress("pos_embed_2d", d_pe_);
    ctx_->setTensorAddress("f", d_feat_);
    TRT_CHECK(ctx_->enqueueV3(stream_), "enqueueV3");
    TRT_CUDA_CHECK(cudaStreamSynchronize(stream_));

    TRT_CUDA_CHECK(cudaMemcpy(features_out, d_feat_, feat_bytes_,
                               cudaMemcpyDeviceToHost));

    return true;
}

} // namespace vision
} // namespace omni
