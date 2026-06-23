// Simple dump_vision_emb: load vision GGUF, encode image, dump final embedding.
#include "vision.h"
#include "ggml.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "Usage: %s <vision.gguf> <image.jpg> [<image2.jpg> ...]\n", argv[0]); return 1; }
    ggml_time_init();
    vision_context_params vp;
    vp.use_gpu = true; vp.verbosity = GGML_LOG_LEVEL_WARN; vp.coreml_model_path = nullptr;

    struct vision_ctx * ctx = vision_init(argv[1], vp);
    if (!ctx) { fprintf(stderr, "vision_init failed\n"); return 1; }

    for (int i = 2; i < argc; ++i) {
        std::ifstream f(argv[i], std::ios::binary | std::ios::ate);
        size_t sz = f.tellg(); f.seekg(0, std::ios::beg);
        std::vector<unsigned char> buf(sz); f.read((char*)buf.data(), sz);

        struct vision_image_u8 * img = vision_image_u8_init();
        vision_image_load_from_bytes(buf.data(), sz, img);
        vision_image_f32_batch batch;
        vision_image_preprocess(ctx, img, &batch);

        struct vision_image_f32 * prep = batch.entries[0].get();
        int H = prep->ny, W = prep->nx, C = 3, n_pixels = C * H * W;
        uint32_t hdr[2] = {(uint32_t)H, (uint32_t)W};
        fwrite(hdr, sizeof(uint32_t), 2, stdout);
        fwrite(prep->buf.data(), sizeof(float), n_pixels, stdout);

        int n_tokens = vision_n_output_tokens(ctx);
        int n_embd = vision_n_mmproj_embd(ctx);
        std::vector<float> emb(n_tokens * n_embd);
        vision_image_batch_encode(ctx, 4, &batch, emb.data());

        double s = 0; for (float v : emb) s += v;
        double s2 = 0; for (float v : emb) s2 += (double)v*v;
        fprintf(stderr, "emb: mean=%.6f L2=%.3f\n", s/emb.size(), std::sqrt(s2));

        fwrite(emb.data(), sizeof(float), emb.size(), stdout);
        fflush(stdout);
        vision_image_u8_free(img);
    }
    vision_free(ctx);
    return 0;
}
