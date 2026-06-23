// Standalone TRT vision encoder benchmark.
// Build: cmake -DENABLE_TRT_VISION=ON ... && cmake --build . --target llama-omni-test-trt-vision
// Run:   OMNI_TRT_VISION_ENGINE=<plan> ./bin/llama-omni-test-trt-vision

#include "trt/omni_trt.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main() {
    const char * engine_path = std::getenv("OMNI_TRT_VISION_ENGINE");
    if (!engine_path || !engine_path[0]) {
        std::fprintf(stderr, "Usage: OMNI_TRT_VISION_ENGINE=<plan> %s\n",
                     "llama-omni-test-trt-vision");
        return 1;
    }

    omni::vision::OmniTrt trt_vision;
    if (!trt_vision.init(engine_path)) {
        std::fprintf(stderr, "OmniTrt init failed\n");
        return 1;
    }

    // Fixed input shapes: [1, 3, 14, 14336] — patch layout
    constexpr int kImgH = 14, kImgW = 14336;
    constexpr int kNumPatches = 1024;
    constexpr int kOutDim = 4096;
    constexpr int kNumQueries = 64;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    int n_pixels = 3 * kImgH * kImgW;
    std::vector<float> pixel_values(n_pixels);
    std::vector<int32_t> position_ids(kNumPatches, 0);
    std::vector<float> pos_embed(kNumPatches * kOutDim, 0.0f);
    for (auto & v : pixel_values) v = dist(rng);

    // Warmup
    std::vector<float> features(kNumQueries * kOutDim);
    if (!trt_vision.run(pixel_values.data(), position_ids.data(), pos_embed.data(), features.data())) {
        std::fprintf(stderr, "warmup run failed\n");
        return 1;
    }

    // Benchmark
    constexpr int kIters = 50;
    double total_ms = 0.0;
    double min_ms   = 1e9;
    double max_ms   = 0.0;

    for (int i = 0; i < kIters; ++i) {
        for (auto & v : pixel_values) v = dist(rng);

        auto t0 = std::chrono::steady_clock::now();
        if (!trt_vision.run(pixel_values.data(), position_ids.data(), pos_embed.data(), features.data())) {
            std::fprintf(stderr, "run failed at iter %d\n", i);
            return 1;
        }
        auto t1 = std::chrono::steady_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
        if (ms > max_ms) max_ms = ms;
    }

    double avg_ms = total_ms / kIters;

    std::fprintf(stderr, "=== TRT Vision Encoder Benchmark ===\n");
    std::fprintf(stderr, "  Engine:      %s\n", engine_path);
    std::fprintf(stderr, "  Input:       [1, 3, %d, %d]\n", kImgH, kImgW);
    std::fprintf(stderr, "  Output:      [1, %d, %d]\n", kNumQueries, kOutDim);
    std::fprintf(stderr, "  Iterations:  %d\n", kIters);
    std::fprintf(stderr, "  Min:         %.3f ms\n", min_ms);
    std::fprintf(stderr, "  Max:         %.3f ms\n", max_ms);
    std::fprintf(stderr, "  Avg:         %.3f ms\n", avg_ms);

    return 0;
}
