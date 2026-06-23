#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace nvinfer1 {
class IRuntime;
class ICudaEngine;
class IExecutionContext;
}
typedef struct CUstream_st *cudaStream_t;

namespace omni {
namespace vision {

class OmniTrt {
public:
    OmniTrt();
    ~OmniTrt();

    bool init(const char * engine_path);
    bool ready() const { return ready_; }

    // Three-input interface:
    //   pixel_values:  [1, 3, 14, 14336]  (CHW float32, patch layout)
    //   position_ids:  [num_patches]       (int32, bucketized)
    //   pos_embed_2d:  [num_patches, out_dim]  (float32, 2D sincos)
    bool run(const float * pixel_values, const int32_t * position_ids,
             const float * pos_embed_2d, float * features_out);

private:
    bool            ready_ = false;
    std::string     engine_path_;

    nvinfer1::IRuntime          * runtime_ = nullptr;
    nvinfer1::ICudaEngine       * engine_  = nullptr;
    nvinfer1::IExecutionContext * ctx_     = nullptr;
    cudaStream_t                  stream_  = nullptr;

    void * d_img_     = nullptr;
    void * d_pid_     = nullptr;
    void * d_pe_      = nullptr;
    void * d_feat_    = nullptr;
    size_t img_bytes_  = 0;
    size_t pid_bytes_  = 0;
    size_t pe_bytes_   = 0;
    size_t feat_bytes_ = 0;

    static constexpr int kNumPatches = 1024;
    static constexpr int kOutDim     = 4096;
    static constexpr int kNumQueries = 64;
};

} // namespace vision
} // namespace omni
