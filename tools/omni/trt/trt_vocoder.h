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
namespace vocoder {

struct TrtVocoderConfig {
    std::string engine_path;
    int         max_mel_frames = 100;
};

class TrtVocoder {
public:
    TrtVocoder();
    ~TrtVocoder();

    bool init(const TrtVocoderConfig & cfg);
    bool ready() const { return ready_; }

    bool infer(const float * mel_bct, int T_mel,
               std::vector<float> & wave_bt_out, int64_t & out_T_audio);
    bool infer(const float * mel_bct, int T_mel,
               const float * source_stft_tcb, int T_source_frame,
               std::vector<float> & wave_bt_out, int64_t & out_T_audio);

private:
    bool             ready_ = false;
    TrtVocoderConfig cfg_;

    nvinfer1::IRuntime          * runtime_ = nullptr;
    nvinfer1::ICudaEngine       * engine_  = nullptr;
    nvinfer1::IExecutionContext * ctx_     = nullptr;
    cudaStream_t                  stream_  = 0;

    void * d_mel_        = nullptr;
    void * d_src_        = nullptr;
    void * d_stft_       = nullptr;
    size_t mel_bytes_    = 0;
    size_t src_bytes_    = 0;
    size_t stft_bytes_   = 0;
    bool   has_src_input_ = false;
    bool   dynamic_mel_   = false;
    bool   dynamic_src_   = false;
    int    src_frames_    = 0;
    int    stft_frames_   = 0;
    std::string mel_name_;
    std::string src_name_;
    std::string stft_name_;

    std::vector<float> window_;
    std::vector<float> window_sq_;
};

} // namespace vocoder
} // namespace omni
