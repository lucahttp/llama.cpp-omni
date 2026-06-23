// Compare GGUF and TRT vision encoder outputs on the same input.
// Build: ./build_trt.sh llama-omni-test-trt-compare
#include "vision.h"
#include "ggml.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

static bool file_exists(const char * path) {
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <vision.gguf> <vision.plan> <image.jpg>\n", argv[0]);
        return 1;
    }
    const char * gguf_path = argv[1];
    const char * plan_path = argv[2];
    const char * img_path  = argv[3];

    ggml_time_init();

    // Load image
    std::ifstream f(img_path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "Cannot open: %s\n", img_path); return 1; }
    size_t sz = f.tellg(); f.seekg(0, std::ios::beg);
    std::vector<unsigned char> buf(sz); f.read((char*)buf.data(), sz);

    struct vision_image_u8 * img_u8 = vision_image_u8_init();
    vision_image_load_from_bytes(buf.data(), sz, img_u8);

    // --- 1. GGUF encode ---
    fprintf(stderr, "=== GGUF encode ===\n");
    {
        vision_context_params vp;
        vp.use_gpu = true; vp.verbosity = GGML_LOG_LEVEL_WARN; vp.coreml_model_path = nullptr;
        struct vision_ctx * ctx = vision_init(gguf_path, vp);
        if (!ctx) { fprintf(stderr, "vision_init GGUF failed\n"); return 1; }

        vision_image_f32_batch batch;
        vision_image_preprocess(ctx, img_u8, &batch);

        int n_tokens = vision_n_output_tokens(ctx);
        int n_embd = vision_n_mmproj_embd(ctx);
        int total = n_tokens * n_embd;
        std::vector<float> emb(total);
        vision_image_batch_encode(ctx, 4, &batch, emb.data());

        double s = 0; for (float v : emb) s += v;
        double s2 = 0; for (float v : emb) s2 += (double)v * v;
        fprintf(stderr, "GGUF: %dx%d img -> %dx%d emb, mean=%.6f L2=%.3f\n",
                batch.entries[0]->nx, batch.entries[0]->ny, n_tokens, n_embd, s/total, std::sqrt(s2));

        // Write to stdout
        fwrite(emb.data(), sizeof(float), total, stdout);
        vision_free(ctx);
    }

    // --- 2. TRT encode ---
    fprintf(stderr, "\n=== TRT encode ===\n");
    {
        vision_context_params vp;
        vp.use_gpu = true; vp.verbosity = GGML_LOG_LEVEL_WARN; vp.coreml_model_path = nullptr;
        struct vision_ctx * ctx = vision_init(gguf_path, vp);
        if (!ctx) { fprintf(stderr, "vision_init TRT failed\n"); return 1; }

        // Set TRT engine path (must be done before encode)
        vision_set_trt_vision_engine_path(ctx, plan_path);

        vision_image_f32_batch batch;
        vision_image_preprocess(ctx, img_u8, &batch);

        int n_tokens = vision_n_output_tokens(ctx);
        int n_embd = vision_n_mmproj_embd(ctx);
        int total = n_tokens * n_embd;
        std::vector<float> emb(total);
        vision_image_batch_encode(ctx, 4, &batch, emb.data());

        double s = 0; for (float v : emb) s += v;
        double s2 = 0; for (float v : emb) s2 += (double)v * v;
        fprintf(stderr, "TRT:  %dx%d img -> %dx%d emb, mean=%.6f L2=%.3f\n",
                batch.entries[0]->nx, batch.entries[0]->ny, n_tokens, n_embd, s/total, std::sqrt(s2));

        // Write to stdout (appended after GGUF data)
        fwrite(emb.data(), sizeof(float), total, stdout);
        vision_free(ctx);
    }

    vision_image_u8_free(img_u8);
    return 0;
}
