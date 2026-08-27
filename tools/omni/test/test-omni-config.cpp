#include "arg.h"
#include "common.h"
#include "omni.h"

#undef NDEBUG
#include <cassert>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

static constexpr uint64_t GiB = 1024ULL * 1024ULL * 1024ULL;

class temporary_model_tree {
  public:
    temporary_model_tree() {
        const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
        root = std::filesystem::temp_directory_path() / ("omni-config-test-" + std::to_string(unique));
        std::filesystem::create_directories(root);
    }

    ~temporary_model_tree() {
        std::error_code error;
        std::filesystem::remove_all(root, error);
    }

    void touch(const std::filesystem::path & relative_path) const {
        const auto path = root / relative_path;
        std::filesystem::create_directories(path.parent_path());
        std::ofstream file(path);
        file << "test";
    }

    void write(const std::filesystem::path & relative_path, const std::string & contents) const {
        const auto path = root / relative_path;
        std::filesystem::create_directories(path.parent_path());
        std::ofstream file(path);
        file << contents;
    }

    std::filesystem::path root;
};

static void create_complete_model_tree(const temporary_model_tree & tree) {
    for (const auto & relative_path : {
             std::filesystem::path("MiniCPM-o-4_5-Q4_K_M.gguf"),
             std::filesystem::path("vision/MiniCPM-o-4_5-vision-F16.gguf"),
             std::filesystem::path("audio/MiniCPM-o-4_5-audio-F16.gguf"),
             std::filesystem::path("tts/MiniCPM-o-4_5-tts-F16.gguf"),
             std::filesystem::path("tts/MiniCPM-o-4_5-projector-F16.gguf"),
             std::filesystem::path("token2wav-gguf/encoder.gguf"),
             std::filesystem::path("token2wav-gguf/flow_matching.gguf"),
             std::filesystem::path("token2wav-gguf/flow_extra.gguf"),
             std::filesystem::path("token2wav-gguf/hifigan2.gguf"),
             std::filesystem::path("token2wav-gguf/prompt_cache.gguf") }) {
        tree.touch(relative_path);
    }
}

static std::string complete_config() {
    return R"json({
  "schema_version": 1,
  "llm": {
    "model": "MiniCPM-o-4_5-Q4_K_M.gguf",
    "quantization": "Q4_K_M",
    "device": "CUDA0",
    "n_gpu_layers": 37
  },
  "vision": {
    "model": "vision/MiniCPM-o-4_5-vision-F16.gguf",
    "device": "CUDA1"
  },
  "audio": {
    "model": "audio/MiniCPM-o-4_5-audio-F16.gguf",
    "device": "CUDA1"
  },
  "tts": {
    "model": "tts/MiniCPM-o-4_5-tts-F16.gguf",
    "device": "CUDA0",
    "gpu_layers": -1
  },
  "projector": {
    "model": "tts/MiniCPM-o-4_5-projector-F16.gguf",
    "device": "CUDA1"
  },
  "token2wav": {
    "model_dir": "token2wav-gguf",
    "device": "CUDA0",
    "threads": 12
  },
  "n_ctx": 6144,
  "duplex": true,
  "async": false,
  "vpm_batch_encode": true
})json";
}

static omni::hardware_snapshot two_accelerators() {
    omni::hardware_snapshot hardware;
    hardware.devices.push_back({ "CUDA0", "NVIDIA H20", true, 24 * GiB, 96 * GiB, "GPU" });
    hardware.devices.push_back({ "CUDA1", "NVIDIA H20", true, 24 * GiB, 96 * GiB, "GPU" });
    return hardware;
}

static omni::config_result load_from_tree(const temporary_model_tree & tree,
                                                  const omni::hardware_snapshot & hardware) {
    return omni::load_config((tree.root / "omni-config.json").string(), hardware, tree.root.string());
}

static void test_config_loads_values_and_maps_logical_devices() {
    temporary_model_tree tree;
    create_complete_model_tree(tree);
    tree.write("omni-config.json", complete_config());

    const auto result = load_from_tree(tree, two_accelerators());

    assert(result.ok);
    assert(result.loaded.path == (tree.root / "omni-config.json").string());
    assert(result.loaded.llm_model == (tree.root / "MiniCPM-o-4_5-Q4_K_M.gguf").string());
    assert(result.loaded.llm_quantization == "Q4_K_M");
    assert(result.loaded.llm_device == "CUDA0");
    assert(result.loaded.n_gpu_layers == 37);
    assert(result.loaded.vision_device == "CUDA1");
    assert(result.loaded.audio_device == "CUDA1");
    assert(result.loaded.tts_device == "CUDA0");
    assert(result.loaded.tts_gpu_layers == -1);
    assert(result.loaded.token2wav_device == "gpu:0");
    assert(result.loaded.token2wav_threads == 12);
    assert(result.loaded.n_ctx == 6144);
    assert(result.loaded.duplex_mode);
    assert(!result.loaded.async_mode);
    assert(result.loaded.vpm_batch_encode);
}

static void test_config_explicit_overrides_win_over_json() {
    temporary_model_tree tree;
    create_complete_model_tree(tree);
    tree.touch("MiniCPM-o-4_5-F16.gguf");
    tree.touch("vision/custom-vision.gguf");
    tree.write("omni-config.json", complete_config());

    omni::config_overrides overrides;
    overrides.llm_model = (tree.root / "MiniCPM-o-4_5-F16.gguf").string();
    overrides.vision_model = (tree.root / "vision/custom-vision.gguf").string();
    overrides.n_gpu_layers = 20;
    overrides.n_ctx = 1024;
    overrides.token2wav_threads = 4;
    overrides.vpm_batch_encode = false;

    const auto result = omni::load_config((tree.root / "omni-config.json").string(), two_accelerators(),
                                          tree.root.string(), overrides);

    assert(result.ok);
    assert(result.loaded.llm_model == *overrides.llm_model);
    assert(result.loaded.llm_quantization == "F16");
    assert(result.loaded.vision_model == *overrides.vision_model);
    assert(result.loaded.n_gpu_layers == 20);
    assert(result.loaded.n_ctx == 1024);
    assert(result.loaded.token2wav_threads == 4);
    assert(!result.loaded.vpm_batch_encode);
}

static void test_config_reports_missing_file() {
    temporary_model_tree tree;
    create_complete_model_tree(tree);
    const auto result = load_from_tree(tree, two_accelerators());

    assert(!result.ok);
    assert(result.error.find("config file") != std::string::npos);
    assert(result.error.find("omni-config.json") != std::string::npos);
}

static void test_config_reports_invalid_json() {
    temporary_model_tree tree;
    create_complete_model_tree(tree);
    tree.write("omni-config.json", "{\"schema_version\":");

    const auto result = load_from_tree(tree, two_accelerators());

    assert(!result.ok);
    assert(result.error.find("invalid config JSON") != std::string::npos);
}

static void test_config_omits_optional_module() {
    temporary_model_tree tree;
    create_complete_model_tree(tree);
    auto config = complete_config();
    const auto marker = std::string("  \"audio\": {\n    \"model\": \"audio/MiniCPM-o-4_5-audio-F16.gguf\",\n    \"device\": \"CUDA1\"\n  },\n");
    config.replace(config.find(marker), marker.size(), "");
    tree.write("omni-config.json", config);

    const auto result = load_from_tree(tree, two_accelerators());

    assert(result.ok);
    assert(result.loaded.audio_model.empty());
}

static void test_config_requires_llm_object() {
    temporary_model_tree tree;
    create_complete_model_tree(tree);
    auto config = complete_config();
    const auto marker = std::string("  \"llm\": {\n    \"model\": \"MiniCPM-o-4_5-Q4_K_M.gguf\",\n    \"quantization\": \"Q4_K_M\",\n    \"device\": \"CUDA0\",\n    \"n_gpu_layers\": 37\n  },\n");
    config.replace(config.find(marker), marker.size(), "");
    tree.write("omni-config.json", config);

    const auto result = load_from_tree(tree, two_accelerators());

    assert(!result.ok);
    assert(result.error.find("llm") != std::string::npos);
}

static void test_config_skips_missing_optional_model_file() {
    temporary_model_tree tree;
    create_complete_model_tree(tree);
    std::filesystem::remove(tree.root / "vision/MiniCPM-o-4_5-vision-F16.gguf");
    tree.write("omni-config.json", complete_config());

    const auto result = load_from_tree(tree, two_accelerators());

    assert(result.ok);
    assert(result.loaded.llm_model.find("Q4_K_M") != std::string::npos);
    assert(result.loaded.vision_model.empty());
    assert(result.loaded.reason.find("skipped missing modules: vision") != std::string::npos);
}

static void test_config_reports_missing_model_file() {
    temporary_model_tree tree;
    create_complete_model_tree(tree);
    auto config = complete_config();
    const auto marker = "MiniCPM-o-4_5-Q4_K_M.gguf";
    config.replace(config.find(marker), std::string(marker).size(), "missing.gguf");
    tree.write("omni-config.json", config);

    const auto result = load_from_tree(tree, two_accelerators());

    assert(!result.ok);
    assert(result.error.find("llm model") != std::string::npos);
    assert(result.error.find("missing.gguf") != std::string::npos);
}

static void test_config_rejects_quantization_filename_mismatch() {
    temporary_model_tree tree;
    create_complete_model_tree(tree);
    auto config = complete_config();
    const auto marker = "\"quantization\": \"Q4_K_M\"";
    config.replace(config.find(marker), std::string(marker).size(), "\"quantization\": \"F16\"");
    tree.write("omni-config.json", config);

    const auto result = load_from_tree(tree, two_accelerators());

    assert(!result.ok);
    assert(result.error.find("quantization") != std::string::npos);
    assert(result.error.find("Q4_K_M") != std::string::npos);
}

static void test_config_reports_unavailable_logical_device() {
    temporary_model_tree tree;
    create_complete_model_tree(tree);
    tree.write("omni-config.json", complete_config());

    omni::hardware_snapshot hardware;
    hardware.devices.push_back({ "CUDA0", "NVIDIA H20", true, 24 * GiB, 96 * GiB, "GPU" });
    const auto result = load_from_tree(tree, hardware);

    assert(!result.ok);
    assert(result.error.find("CUDA1") != std::string::npos);
    assert(result.error.find("vision") != std::string::npos);
}

static void test_config_rejects_empty_device_value() {
    temporary_model_tree tree;
    create_complete_model_tree(tree);
    auto config = complete_config();
    const auto marker = "\"device\": \"CUDA0\"";
    config.replace(config.find(marker), std::string(marker).size(), "\"device\": \"\"");
    tree.write("omni-config.json", config);

    const auto result = load_from_tree(tree, two_accelerators());

    assert(!result.ok);
    assert(result.error.find("llm.device") != std::string::npos);
    assert(result.error.find("empty") != std::string::npos);
}

static void test_effective_config_reports_source_path() {
    temporary_model_tree tree;
    create_complete_model_tree(tree);
    tree.write("omni-config.json", complete_config());

    const auto result = load_from_tree(tree, two_accelerators());
    assert(result.ok);
    const auto output = omni::format_config(result.loaded);
    assert(output.find("config_path=" + result.loaded.path + "\n") != std::string::npos);
    assert(output.find("llm_quantization=Q4_K_M\n") != std::string::npos);
}

static void test_config_controls_session_options_when_present() {
    omni::config config;
    config.duplex_mode       = true;
    config.tts_gpu_layers    = -1;
    config.token2wav_device  = "gpu:0";
    config.token2wav_threads = 32;

    const auto locked = omni::resolve_session_options(&config, false, 100, "cpu", 4);
    assert(locked.duplex_mode);
    assert(!locked.async_mode);
    assert(locked.tts_gpu_layers == -1);
    assert(locked.token2wav_device == "gpu:0");
    assert(locked.token2wav_threads == 32);
    assert(locked.config_locked);
    assert(!omni::config_accepts_session_mode(&config, false));
    assert(omni::config_accepts_session_mode(&config, true));
    assert(omni::config_accepts_session_mode(nullptr, false));

    const auto legacy = omni::resolve_session_options(nullptr, false, 100, "cpu", 4);
    assert(!legacy.duplex_mode);
    assert(legacy.async_mode);
    assert(legacy.tts_gpu_layers == 100);
    assert(legacy.token2wav_device == "cpu");
    assert(legacy.token2wav_threads == 4);
    assert(!legacy.config_locked);
}

static void test_primary_follows_ggml_device_order() {
    temporary_model_tree tree;
    create_complete_model_tree(tree);
    auto config = complete_config();
    const auto cuda1 = std::string("CUDA1");
    const auto cuda0 = std::string("CUDA0");
    for (std::string::size_type pos = 0; (pos = config.find(cuda1, pos)) != std::string::npos; ) {
        config.replace(pos, cuda1.size(), "secondary");
        pos += std::string("secondary").size();
    }
    for (std::string::size_type pos = 0; (pos = config.find(cuda0, pos)) != std::string::npos; ) {
        config.replace(pos, cuda0.size(), "primary");
        pos += std::string("primary").size();
    }
    tree.write("omni-config.json", config);

    omni::hardware_snapshot hardware;
    hardware.devices.push_back({ "CUDA0", "NVIDIA H20", true, 8 * GiB, 96 * GiB, "GPU" });
    hardware.devices.push_back({ "CUDA1", "NVIDIA H20", true, 80 * GiB, 96 * GiB, "GPU" });

    const auto result = load_from_tree(tree, hardware);

    assert(result.ok);
    assert(result.loaded.llm_device == "CUDA0");
    assert(result.loaded.vision_device == "CUDA1");
    assert(result.loaded.primary_device_name == "CUDA0");
}

static bool parse_omni_server_arguments(std::vector<std::string> arguments) {
    common_params        params;
    std::vector<char *> argv;
    for (auto & argument : arguments) {
        argv.push_back(argument.data());
    }
    return common_params_parse(static_cast<int>(argv.size()), argv.data(), params, LLAMA_EXAMPLE_OMNI_SERVER);
}

static void test_omni_config_arguments_are_parsed() {
    common_params            params;
    std::vector<std::string> arguments = {
        "llama-omni-server", "--config", "/models/omni-config.json", "--model-dir", "/models/MiniCPM-o-4_5-gguf",
        "--token2wav-threads", "32", "--print-effective-config",
    };
    std::vector<char *> argv;
    for (auto & argument : arguments) {
        argv.push_back(argument.data());
    }

    const bool parsed =
        common_params_parse(static_cast<int>(argv.size()), argv.data(), params, LLAMA_EXAMPLE_OMNI_SERVER);

    assert(parsed);
    assert(params.omni_config.path == "/models/omni-config.json");
    assert(params.omni_config.model_dir == "/models/MiniCPM-o-4_5-gguf");
    assert(params.omni_config.token2wav_threads == 32);
    assert(params.omni_config.print_effective_config);
}

static void test_omni_config_argument_ranges_are_validated() {
    assert(!parse_omni_server_arguments({ "llama-omni-server", "--token2wav-threads", "0" }));
}

static void test_bundled_config_name_resolves_to_json_file() {
    const auto cuda = omni::resolve_config_path("cuda");
    assert(cuda.find("cuda.json") != std::string::npos);
    assert(std::filesystem::is_regular_file(cuda));

    const auto two_gpu = omni::resolve_config_path("cuda-2gpu");
    assert(two_gpu.find("cuda-2gpu.json") != std::string::npos);
    assert(std::filesystem::is_regular_file(two_gpu));

    temporary_model_tree tree;
    tree.write("custom.json", complete_config());
    const auto custom = omni::resolve_config_path((tree.root / "custom.json").string());
    assert(custom == (tree.root / "custom.json").string());
}

static void test_select_config_name_from_hardware() {
    omni::hardware_snapshot cpu_only;
    cpu_only.devices.push_back({ "CPU", "host", false, 0, 0, "CPU" });
    assert(omni::select_config_name(cpu_only) == "cpu");

    omni::hardware_snapshot one_cuda;
    one_cuda.devices.push_back({ "CPU", "host", false, 0, 0, "CPU" });
    one_cuda.devices.push_back({ "CUDA0", "NVIDIA", true, 8 * GiB, 24 * GiB, "GPU" });
    assert(omni::select_config_name(one_cuda) == "cuda");
    assert(omni::resolve_config_request("auto", one_cuda).find("cuda.json") != std::string::npos);

    omni::hardware_snapshot two_cuda;
    two_cuda.devices.push_back({ "CUDA0", "NVIDIA", true, 8 * GiB, 24 * GiB, "GPU" });
    two_cuda.devices.push_back({ "CUDA1", "NVIDIA", true, 8 * GiB, 24 * GiB, "GPU" });
    assert(omni::select_config_name(two_cuda) == "cuda-2gpu");

    omni::hardware_snapshot metal;
    metal.devices.push_back({ "Metal", "Apple M2", true, 8 * GiB, 16 * GiB, "GPU" });
    assert(omni::select_config_name(metal) == "metal");

    omni::hardware_snapshot vulkan;
    vulkan.devices.push_back({ "CPU", "host", false, 0, 0, "CPU" });
    vulkan.devices.push_back({ "Vulkan0", "AMD", true, 8 * GiB, 16 * GiB, "GPU" });
    std::string vulkan_error;
    assert(omni::select_config_name(vulkan, &vulkan_error).empty());
    assert(vulkan_error.find("Vulkan0") != std::string::npos);

    omni::hardware_snapshot two_vulkan;
    two_vulkan.devices.push_back({ "Vulkan0", "AMD", true, 8 * GiB, 16 * GiB, "GPU" });
    two_vulkan.devices.push_back({ "Vulkan1", "AMD", true, 8 * GiB, 16 * GiB, "GPU" });
    assert(omni::select_config_name(two_vulkan).empty());

    omni::hardware_snapshot cuda_and_vulkan;
    cuda_and_vulkan.devices.push_back({ "CUDA0", "NVIDIA", true, 8 * GiB, 24 * GiB, "GPU" });
    cuda_and_vulkan.devices.push_back({ "Vulkan0", "AMD", true, 8 * GiB, 16 * GiB, "GPU" });
    assert(omni::select_config_name(cuda_and_vulkan) == "cuda");
}

static void test_omni_config_auto_argument_is_parsed() {
    common_params            params;
    std::vector<std::string> arguments = { "llama-omni-server", "--config", "--model-dir", "/models/MiniCPM-o-4_5-gguf" };
    std::vector<char *> argv;
    for (auto & argument : arguments) {
        argv.push_back(argument.data());
    }
    const bool parsed =
        common_params_parse(static_cast<int>(argv.size()), argv.data(), params, LLAMA_EXAMPLE_OMNI_SERVER);
    assert(parsed);
    assert(params.omni_config.path == "auto");
    assert(params.omni_config.model_dir == "/models/MiniCPM-o-4_5-gguf");
}

static void test_omni_config_rejects_autotune_arguments() {
    assert(!parse_omni_server_arguments({ "llama-omni-server", "--config", "/tmp/omni-config.json", "--autotune", "refresh" }));
    assert(!parse_omni_server_arguments({ "llama-omni-server", "--config", "/tmp/omni-config.json", "--autotune-cache",
                                          "/tmp/omni-autotune.json" }));
    assert(!parse_omni_server_arguments({ "llama-omni-server", "--config", "/tmp/omni-config.json", "--autotune-rounds", "3" }));
}

int main() {
    test_config_loads_values_and_maps_logical_devices();
    test_config_explicit_overrides_win_over_json();
    test_config_reports_missing_file();
    test_config_reports_invalid_json();
    test_config_omits_optional_module();
    test_config_requires_llm_object();
    test_config_skips_missing_optional_model_file();
    test_config_reports_missing_model_file();
    test_config_rejects_quantization_filename_mismatch();
    test_config_reports_unavailable_logical_device();
    test_config_rejects_empty_device_value();
    test_effective_config_reports_source_path();
    test_config_controls_session_options_when_present();
    test_primary_follows_ggml_device_order();
    test_omni_config_arguments_are_parsed();
    test_omni_config_argument_ranges_are_validated();
    test_bundled_config_name_resolves_to_json_file();
    test_select_config_name_from_hardware();
    test_omni_config_auto_argument_is_parsed();
    test_omni_config_rejects_autotune_arguments();
    return 0;
}
