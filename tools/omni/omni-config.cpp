#include "omni-config.h"

#include "common.h"
#include "ggml-backend.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <vector>

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#elif defined(__linux__) || defined(__FreeBSD__)
#include <unistd.h>
#elif defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace omni {

static std::string normalize_dir(const std::filesystem::path & path) {
    std::error_code error;
    const auto canonical = std::filesystem::weakly_canonical(path, error);
    return error ? path.lexically_normal().string() : canonical.string();
}

static std::string executable_dir() {
#if defined(__APPLE__)
    std::vector<char> path;
    uint32_t size = 0;
    while (true) {
        size = static_cast<uint32_t>(path.size());
        if (_NSGetExecutablePath(path.data(), &size) == 0) {
            break;
        }
        path.resize(size);
    }
    return normalize_dir(std::filesystem::path(path.data()).parent_path());
#elif defined(__linux__) || defined(__FreeBSD__)
    std::vector<char> path(1024);
    while (true) {
#    if defined(__linux__)
        const ssize_t len = readlink("/proc/self/exe", path.data(), path.size());
#    else
        const ssize_t len = readlink("/proc/curproc/file", path.data(), path.size());
#    endif
        if (len < 0) {
            return {};
        }
        if (static_cast<size_t>(len) < path.size()) {
            return normalize_dir(std::filesystem::path(std::string(path.data(), static_cast<size_t>(len))).parent_path());
        }
        path.resize(path.size() * 2);
    }
#elif defined(_WIN32)
    std::vector<wchar_t> path(MAX_PATH);
    const DWORD len = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
    if (len == 0 || len >= path.size()) {
        return {};
    }
    return normalize_dir(std::filesystem::path(std::wstring(path.data(), len)).parent_path());
#else
    return {};
#endif
}

static void add_existing_dir(std::vector<std::string> & dirs, const std::filesystem::path & path) {
    std::error_code error;
    if (!std::filesystem::is_directory(path, error)) {
        return;
    }
    const std::string normalized = normalize_dir(path);
    if (std::find(dirs.begin(), dirs.end(), normalized) == dirs.end()) {
        dirs.push_back(normalized);
    }
}

static std::vector<std::string> config_search_dirs() {
    std::vector<std::string> dirs;
#ifdef OMNI_DEFAULT_CONFIG_DIR
    add_existing_dir(dirs, OMNI_DEFAULT_CONFIG_DIR);
#endif
#ifdef OMNI_INSTALL_CONFIG_DIR
    add_existing_dir(dirs, OMNI_INSTALL_CONFIG_DIR);
#endif
    const std::string exe = executable_dir();
    if (!exe.empty()) {
        add_existing_dir(dirs, std::filesystem::path(exe) / "config");
        add_existing_dir(dirs, std::filesystem::path(exe) / ".." / "share" / "llama.cpp-omni" / "config");
    }
    return dirs;
}

static bool is_bundled_config_dir(const std::string & dir) {
    if (dir.empty()) {
        return false;
    }
    const std::string normalized = normalize_dir(dir);
    for (const auto & search : config_search_dirs()) {
        if (search == normalized) {
            return true;
        }
    }
    return false;
}

std::string default_config_dir() {
    const auto dirs = config_search_dirs();
    return dirs.empty() ? std::string() : dirs.front();
}

std::string resolve_config_path(const std::string & name_or_path) {
    if (name_or_path.empty()) {
        return {};
    }

    std::error_code error;
    const std::filesystem::path requested(name_or_path);
    if (std::filesystem::is_regular_file(requested, error)) {
        return requested.string();
    }

    for (const auto & dir : config_search_dirs()) {
        const std::filesystem::path bundled = std::filesystem::path(dir) / name_or_path;
        if (std::filesystem::is_regular_file(bundled, error)) {
            return bundled.string();
        }
        if (requested.extension().empty()) {
            const std::filesystem::path named = std::filesystem::path(dir) / (name_or_path + ".json");
            if (std::filesystem::is_regular_file(named, error)) {
                return named.string();
            }
        }
    }
    return name_or_path;
}

static bool name_has_token(const std::string & name, const char * token) {
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return lower.find(token) != std::string::npos;
}

std::string select_config_name(const hardware_snapshot & hardware, std::string * error) {
    if (error != nullptr) {
        error->clear();
    }

    int n_cuda  = 0;
    int n_metal = 0;
    std::vector<std::string> unknown;
    for (const auto & device : hardware.devices) {
        if (!device.is_accelerator) {
            continue;
        }
        if (name_has_token(device.name, "cuda") || name_has_token(device.description, "cuda")) {
            n_cuda++;
        } else if (name_has_token(device.name, "metal") || name_has_token(device.description, "metal")) {
            n_metal++;
        } else {
            unknown.push_back(device.name.empty() ? device.description : device.name);
        }
    }
    if (n_metal > 0) {
        return "metal";
    }
    if (n_cuda >= 2) {
        return "cuda-2gpu";
    }
    if (n_cuda >= 1) {
        return "cuda";
    }
    if (unknown.empty()) {
        return "cpu";
    }
    if (error != nullptr) {
        *error = "no bundled config for accelerator(s): ";
        for (size_t i = 0; i < unknown.size(); ++i) {
            if (i > 0) {
                *error += ", ";
            }
            *error += unknown[i];
        }
        *error += "; pass --config cpu or a JSON path";
    }
    return {};
}

std::string resolve_config_request(const std::string & name_or_path,
                                   const hardware_snapshot & hardware,
                                   std::string * error) {
    if (name_or_path.empty() || name_or_path == "auto") {
        const std::string selected = select_config_name(hardware, error);
        if (selected.empty()) {
            return {};
        }
        return resolve_config_path(selected);
    }
    return resolve_config_path(name_or_path);
}

static std::string parent_dir(const std::string & path) {
    const auto pos = path.find_last_of("/\\");
    return pos == std::string::npos ? std::string() : path.substr(0, pos);
}

bool prepare_config(common_params & params, std::optional<config> & out, std::string & error) {
    out.reset();
    error.clear();
    if (params.omni_config.path.empty()) {
        return true;
    }

    const auto hardware  = detect_hardware_snapshot();
    const bool auto_pick = params.omni_config.path == "auto";
    std::string selected = params.omni_config.path;
    if (auto_pick) {
        selected = select_config_name(hardware, &error);
        if (selected.empty()) {
            if (error.empty()) {
                error = "failed to auto-select a bundled config";
            }
            return false;
        }
    }
    const std::string config_path = resolve_config_path(selected);

    std::string model_root = params.omni_config.model_dir;
    if (model_root.empty() && !params.model.path.empty()) {
        model_root = parent_dir(params.model.path);
    }
    if (model_root.empty()) {
        const std::string cfg_dir = parent_dir(config_path);
        if (!is_bundled_config_dir(cfg_dir)) {
            model_root = cfg_dir;
        }
    }
    if (model_root.empty()) {
        error = "--config requires --model-dir or --model to resolve relative model paths";
        return false;
    }

    if (params.omni_config.model_explicit) {
        std::ifstream explicit_model(params.model.path, std::ios::binary);
        if (!explicit_model.good()) {
            error = "missing explicit LLM model: " + params.model.path;
            return false;
        }
    }

    config_overrides overrides;
    if (params.omni_config.model_explicit) {
        overrides.llm_model = params.model.path;
    }
    if (params.omni_config.vision_explicit) {
        overrides.vision_model = params.vpm_model;
    }
    if (params.omni_config.audio_explicit) {
        overrides.audio_model = params.apm_model;
    }
    if (params.omni_config.tts_explicit) {
        overrides.tts_model = params.tts_model;
    }
    if (params.omni_config.projector_explicit) {
        overrides.projector_model = params.projector_model;
    }
    if (params.omni_config.n_gpu_layers_explicit) {
        overrides.n_gpu_layers = params.n_gpu_layers;
    }
    if (params.omni_config.n_ctx_explicit) {
        overrides.n_ctx = params.n_ctx;
    }
    if (params.omni_config.token2wav_threads_explicit) {
        overrides.token2wav_threads = params.omni_config.token2wav_threads;
    }
    if (params.omni_config.vpm_batch_encode_explicit) {
        overrides.vpm_batch_encode = params.vpm_batch_encode;
    }

    auto resolved = load_config(config_path, hardware, model_root, overrides);
    if (!resolved.ok) {
        error = resolved.error;
        return false;
    }
    if (auto_pick) {
        resolved.loaded.reason = "auto selected " + selected + "; " + resolved.loaded.reason;
    }
    if (!apply_config(params, resolved.loaded, error)) {
        return false;
    }
    out = std::move(resolved.loaded);
    return true;
}

hardware_snapshot detect_hardware_snapshot() {
    hardware_snapshot snapshot;
    ggml_backend_load_all();
    for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        if (device == nullptr) {
            continue;
        }

        size_t free_memory  = 0;
        size_t total_memory = 0;
        ggml_backend_dev_memory(device, &free_memory, &total_memory);
        const auto device_type = ggml_backend_dev_type(device);
        const char * type_name = device_type == GGML_BACKEND_DEVICE_TYPE_CPU ? "CPU" :
                                 device_type == GGML_BACKEND_DEVICE_TYPE_GPU ? "GPU" : "ACCEL";
        snapshot.devices.push_back({
            ggml_backend_dev_name(device),
            ggml_backend_dev_description(device),
            device_type != GGML_BACKEND_DEVICE_TYPE_CPU,
            static_cast<uint64_t>(free_memory),
            static_cast<uint64_t>(total_memory),
            type_name,
        });
    }
    return snapshot;
}

static std::vector<const device *> accelerator_devices(const hardware_snapshot & hardware) {
    std::vector<const device *> devices;
    for (const auto & device : hardware.devices) {
        if (device.is_accelerator) {
            devices.push_back(&device);
        }
    }
    return devices;
}

static std::string token2wav_device_for(const device & device) {
    std::string digits;
    for (auto it = device.name.rbegin(); it != device.name.rend() && std::isdigit(static_cast<unsigned char>(*it)); ++it) {
        digits.push_back(*it);
    }
    if (!digits.empty()) {
        std::reverse(digits.begin(), digits.end());
        return "gpu:" + digits;
    }
    return "gpu:0";
}

static const device * find_device(const hardware_snapshot & hardware, const std::string & name) {
    for (const auto & device : hardware.devices) {
        if (device.name == name) {
            return &device;
        }
    }
    return nullptr;
}

static std::string normalize_device(const std::string & requested,
                                    const hardware_snapshot & hardware,
                                    const std::string & module,
                                    std::string & error) {
    const auto accelerators = accelerator_devices(hardware);
    const device * primary = accelerators.empty() ? nullptr : accelerators.front();
    if (requested.empty() || requested == "cpu") {
        return requested.empty() ? "cpu" : requested;
    }
    if (requested == "primary" || requested == "secondary") {
        const size_t index = requested == "primary" ? 0 : 1;
        if (index < accelerators.size()) {
            return accelerators[index]->name;
        }
        error = "unsupported " + module + " placement device: " + requested +
                " (the requested accelerator is not available)";
        return {};
    }
    if (requested == "gpu") {
        if (primary != nullptr) {
            return primary->name;
        }
        error = "unsupported " + module + " placement device: gpu (no accelerator is available)";
        return {};
    }
    if (requested.rfind("gpu:", 0) == 0) {
        try {
            const size_t index = static_cast<size_t>(std::stoul(requested.substr(4)));
            const auto accelerators = accelerator_devices(hardware);
            if (index < accelerators.size()) {
                return accelerators[index]->name;
            }
        } catch (...) {
        }
    }
    if (find_device(hardware, requested) != nullptr) {
        return requested;
    }
    error = "unsupported " + module + " placement device: " + requested;
    return {};
}

static void add_module_placement(config & config,
                                 const std::string & module,
                                 const std::string & model,
                                 const std::string & precision,
                                 const std::string & device,
                                 const std::string & execution = "independent",
                                 const std::string & status = "active") {
    const std::string backend = status == "active" ? (device == "cpu" ? "cpu" : "gpu") : "-";
    config.placements.push_back({ module, model, precision, device, backend, execution, status });
}

static std::string quantization_from_path(const std::string & path) {
    std::string upper = path;
    std::transform(upper.begin(), upper.end(), upper.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
    if (upper.find("Q4_K_M") != std::string::npos) {
        return "Q4_K_M";
    }
    if (upper.find("Q8_0") != std::string::npos) {
        return "Q8_0";
    }
    if (upper.find("F16") != std::string::npos) {
        return "F16";
    }
    return "UNKNOWN";
}

static void apply_overrides(config & config, const config_overrides & overrides) {
    if (overrides.llm_model) {
        config.llm_model        = *overrides.llm_model;
        config.llm_quantization = quantization_from_path(*overrides.llm_model);
    }
    if (overrides.vision_model) {
        config.vision_model = *overrides.vision_model;
    }
    if (overrides.audio_model) {
        config.audio_model = *overrides.audio_model;
    }
    if (overrides.tts_model) {
        config.tts_model = *overrides.tts_model;
    }
    if (overrides.projector_model) {
        config.projector_model = *overrides.projector_model;
    }
    if (overrides.n_gpu_layers) {
        config.n_gpu_layers = *overrides.n_gpu_layers;
    }
    if (overrides.n_ctx) {
        config.n_ctx = *overrides.n_ctx;
    }
    if (overrides.token2wav_threads) {
        config.token2wav_threads = *overrides.token2wav_threads;
    }
    if (overrides.vpm_batch_encode) {
        config.vpm_batch_encode = *overrides.vpm_batch_encode;
    }
}

using config_json = nlohmann::ordered_json;

template <typename T>
static bool read_required_config_value(const config_json & object,
                                        const char *         section,
                                        const char *         key,
                                        T &                  value,
                                        std::string &        error) {
    if (!object.is_object() || !object.contains(key)) {
        error = std::string("config missing required field: ") + section + "." + key;
        return false;
    }
    try {
        value = object.at(key).get<T>();
    } catch (const std::exception & exception) {
        error = std::string("config field ") + section + "." + key + " has an invalid type: " +
                exception.what();
        return false;
    }
    return true;
}

static bool read_optional_config_object(const config_json & root,
                                         const char *         key,
                                         config_json &       value,
                                         bool &               present,
                                         std::string &        error) {
    present = false;
    if (!root.contains(key)) {
        return true;
    }
    if (!root.at(key).is_object()) {
        error = std::string("config field ") + key + " must be an object";
        return false;
    }
    present = true;
    value = root.at(key);
    return true;
}

static bool config_is_regular_file(const std::string & path) {
    std::error_code error;
    return std::filesystem::is_regular_file(path, error);
}

static bool config_is_directory(const std::string & path) {
    std::error_code error;
    return std::filesystem::is_directory(path, error);
}

static const std::vector<std::pair<const char *, const char *>> & token2wav_required_files() {
    static const std::vector<std::pair<const char *, const char *>> files = {
        { "encoder", "encoder.gguf" },
        { "flow_matching", "flow_matching.gguf" },
        { "flow_extra", "flow_extra.gguf" },
        { "hifigan2", "hifigan2.gguf" },
        { "prompt_cache", "prompt_cache.gguf" },
    };
    return files;
}

static bool token2wav_tree_complete(const std::string & dir) {
    if (!config_is_directory(dir)) {
        return false;
    }
    for (const auto & file_entry : token2wav_required_files()) {
        if (!config_is_regular_file((std::filesystem::path(dir) / file_entry.second).string())) {
            return false;
        }
    }
    return true;
}

static std::string resolve_config_model_path(const std::string & model_root, const std::string & configured_path) {
    std::filesystem::path path(configured_path);
    if (path.is_relative()) {
        path = std::filesystem::path(model_root) / path;
    }
    return path.lexically_normal().string();
}

static bool require_config_file(const std::string & module,
                                 const std::string & path,
                                 std::string &       error) {
    std::error_code filesystem_error;
    if (!std::filesystem::is_regular_file(path, filesystem_error)) {
        error = "config " + module + " model file does not exist: " + path;
        return false;
    }
    return true;
}

static bool require_non_empty_config_value(const char * section,
                                            const char * key,
                                            const std::string & value,
                                            std::string & error) {
    if (value.empty()) {
        error = std::string("config field ") + section + "." + key + " must not be empty";
        return false;
    }
    return true;
}

config_result load_config(const std::string &       config_path,
                          const hardware_snapshot & hardware,
                          const std::string &       model_root,
                          const config_overrides &  overrides) {
    config_result result;
    result.loaded.path = config_path;

    if (config_path.empty()) {
        result.error = "config file path is empty";
        return result;
    }
    if (model_root.empty()) {
        result.error = "config requires a model root; pass --model-dir or --model";
        return result;
    }

    std::ifstream file(config_path);
    if (!file.good()) {
        result.error = "config file does not exist or cannot be read: " + config_path;
        return result;
    }

    config_json document;
    try {
        file >> document;
    } catch (const std::exception & exception) {
        result.error = "invalid config JSON in " + config_path + ": " + exception.what();
        return result;
    }
    if (!document.is_object()) {
        result.error = "invalid config JSON: root must be an object";
        return result;
    }

    int schema_version = 0;
    if (!read_required_config_value(document, "root", "schema_version", schema_version, result.error)) {
        return result;
    }
    if (schema_version != 1) {
        result.error = "unsupported config schema_version: " + std::to_string(schema_version);
        return result;
    }

    config_json llm;
    bool        has_llm = false;
    if (!read_optional_config_object(document, "llm", llm, has_llm, result.error)) {
        return result;
    }
    if (!has_llm) {
        result.error = "config missing required object: llm";
        return result;
    }

    std::string llm_configured_model;
    std::string llm_quantization;
    std::string llm_configured_device;
    int32_t     n_gpu_layers = 0;
    if (!read_required_config_value(llm, "llm", "model", llm_configured_model, result.error) ||
        !read_required_config_value(llm, "llm", "quantization", llm_quantization, result.error) ||
        !read_required_config_value(llm, "llm", "device", llm_configured_device, result.error) ||
        !read_required_config_value(llm, "llm", "n_gpu_layers", n_gpu_layers, result.error) ||
        !require_non_empty_config_value("llm", "model", llm_configured_model, result.error) ||
        !require_non_empty_config_value("llm", "quantization", llm_quantization, result.error) ||
        !require_non_empty_config_value("llm", "device", llm_configured_device, result.error)) {
        return result;
    }

    struct optional_module {
        bool        present = false;
        bool        active  = false;
        std::string configured_model;
        std::string configured_device;
        int32_t     extra = 0;
    };

    auto read_file_module = [&](const char * key, optional_module & module, bool read_gpu_layers) {
        config_json object;
        if (!read_optional_config_object(document, key, object, module.present, result.error)) {
            return false;
        }
        if (!module.present) {
            return true;
        }
        if (!read_required_config_value(object, key, "model", module.configured_model, result.error) ||
            !read_required_config_value(object, key, "device", module.configured_device, result.error)) {
            return false;
        }
        if (read_gpu_layers &&
            !read_required_config_value(object, key, "gpu_layers", module.extra, result.error)) {
            return false;
        }
        return require_non_empty_config_value(key, "model", module.configured_model, result.error) &&
               require_non_empty_config_value(key, "device", module.configured_device, result.error);
    };

    optional_module vision_mod;
    optional_module audio_mod;
    optional_module tts_mod;
    optional_module projector_mod;
    if (!read_file_module("vision", vision_mod, false) ||
        !read_file_module("audio", audio_mod, false) ||
        !read_file_module("tts", tts_mod, true) ||
        !read_file_module("projector", projector_mod, false)) {
        return result;
    }

    optional_module token2wav_mod;
    config_json     token2wav;
    int32_t         token2wav_threads = 8;
    if (!read_optional_config_object(document, "token2wav", token2wav, token2wav_mod.present, result.error)) {
        return result;
    }
    if (token2wav_mod.present) {
        if (!read_required_config_value(token2wav, "token2wav", "model_dir", token2wav_mod.configured_model,
                                         result.error) ||
            !read_required_config_value(token2wav, "token2wav", "device", token2wav_mod.configured_device,
                                         result.error) ||
            !read_required_config_value(token2wav, "token2wav", "threads", token2wav_threads, result.error) ||
            !require_non_empty_config_value("token2wav", "model_dir", token2wav_mod.configured_model, result.error) ||
            !require_non_empty_config_value("token2wav", "device", token2wav_mod.configured_device, result.error)) {
            return result;
        }
        if (token2wav_threads <= 0) {
            result.error = "config token2wav.threads must be positive";
            return result;
        }
    }

    if (!read_required_config_value(document, "root", "n_ctx", result.loaded.n_ctx, result.error) ||
        !read_required_config_value(document, "root", "duplex", result.loaded.duplex_mode, result.error) ||
        !read_required_config_value(document, "root", "async", result.loaded.async_mode, result.error)) {
        return result;
    }
    if (document.contains("vpm_batch_encode") &&
        !read_required_config_value(document, "root", "vpm_batch_encode", result.loaded.vpm_batch_encode,
                                     result.error)) {
        return result;
    }
    if (result.loaded.n_ctx <= 0) {
        result.error = "config n_ctx must be positive";
        return result;
    }

    result.loaded.llm_model        = resolve_config_model_path(model_root, llm_configured_model);
    result.loaded.llm_quantization = llm_quantization;
    result.loaded.n_gpu_layers     = n_gpu_layers;
    result.loaded.tts_gpu_layers   = tts_mod.extra;
    result.loaded.token2wav_threads = token2wav_threads;
    if (vision_mod.present) {
        result.loaded.vision_model = resolve_config_model_path(model_root, vision_mod.configured_model);
    }
    if (audio_mod.present) {
        result.loaded.audio_model = resolve_config_model_path(model_root, audio_mod.configured_model);
    }
    if (tts_mod.present) {
        result.loaded.tts_model = resolve_config_model_path(model_root, tts_mod.configured_model);
    }
    if (projector_mod.present) {
        result.loaded.projector_model = resolve_config_model_path(model_root, projector_mod.configured_model);
    }
    if (token2wav_mod.present) {
        result.loaded.token2wav_model_dir = resolve_config_model_path(model_root, token2wav_mod.configured_model);
    }
    apply_overrides(result.loaded, overrides);

    const std::string detected_quantization = quantization_from_path(result.loaded.llm_model);
    if (detected_quantization != "UNKNOWN" && detected_quantization != result.loaded.llm_quantization) {
        result.error = "config llm.quantization " + result.loaded.llm_quantization +
                       " does not match the LLM model filename (detected " + detected_quantization + ")";
        return result;
    }
    if (!require_config_file("llm", result.loaded.llm_model, result.error)) {
        return result;
    }

    std::vector<std::string> skipped;
    auto activate_file_module = [&](const char * key, optional_module & module, std::string & path, bool explicit_path) {
        if (explicit_path) {
            if (!require_config_file(key, path, result.error)) {
                return false;
            }
            module.active = true;
            return true;
        }
        if (path.empty()) {
            return true;
        }
        if (!config_is_regular_file(path)) {
            skipped.push_back(key);
            path.clear();
            module.active = false;
            return true;
        }
        module.active = true;
        return true;
    };
    if (!activate_file_module("vision", vision_mod, result.loaded.vision_model, overrides.vision_model.has_value()) ||
        !activate_file_module("audio", audio_mod, result.loaded.audio_model, overrides.audio_model.has_value()) ||
        !activate_file_module("tts", tts_mod, result.loaded.tts_model, overrides.tts_model.has_value()) ||
        !activate_file_module("projector", projector_mod, result.loaded.projector_model,
                              overrides.projector_model.has_value())) {
        return result;
    }
    auto fill_device_from_llm = [&](optional_module & module) {
        if (module.active && module.configured_device.empty()) {
            module.configured_device = llm_configured_device;
        }
    };
    fill_device_from_llm(vision_mod);
    fill_device_from_llm(audio_mod);
    fill_device_from_llm(tts_mod);
    fill_device_from_llm(projector_mod);
    fill_device_from_llm(token2wav_mod);
    if (!result.loaded.token2wav_model_dir.empty()) {
        if (token2wav_tree_complete(result.loaded.token2wav_model_dir)) {
            token2wav_mod.active = true;
        } else {
            skipped.push_back("token2wav");
            result.loaded.token2wav_model_dir.clear();
            token2wav_mod.active = false;
        }
    }

    const auto accelerators = accelerator_devices(hardware);
    if (!accelerators.empty()) {
        const auto * primary = accelerators.front();
        result.loaded.primary_device_name         = primary->name;
        result.loaded.primary_device_description  = primary->description;
        result.loaded.primary_device_free_memory  = primary->free_memory;
        result.loaded.primary_device_total_memory = primary->total_memory;
    }

    auto resolve_configured_device = [&](const std::string & requested,
                                         const std::string & module,
                                         std::string &       resolved) {
        resolved = normalize_device(requested, hardware, module, result.error);
        return result.error.empty();
    };
    std::string llm_device;
    std::string vision_device;
    std::string audio_device;
    std::string tts_device;
    std::string projector_device;
    std::string token2wav_backend_device;
    if (!resolve_configured_device(llm_configured_device, "llm", llm_device)) {
        return result;
    }
    if (vision_mod.active &&
        !resolve_configured_device(vision_mod.configured_device, "vision", vision_device)) {
        return result;
    }
    if (audio_mod.active &&
        !resolve_configured_device(audio_mod.configured_device, "audio", audio_device)) {
        return result;
    }
    if (tts_mod.active &&
        !resolve_configured_device(tts_mod.configured_device, "tts", tts_device)) {
        return result;
    }
    if (projector_mod.active &&
        !resolve_configured_device(projector_mod.configured_device, "projector", projector_device)) {
        return result;
    }
    if (token2wav_mod.active &&
        !resolve_configured_device(token2wav_mod.configured_device, "token2wav", token2wav_backend_device)) {
        return result;
    }

    result.loaded.llm_device    = llm_device;
    result.loaded.vision_device = vision_device;
    result.loaded.audio_device  = audio_device;
    result.loaded.tts_device    = tts_device;
    if (token2wav_mod.active) {
        const device * token2wav_dev = find_device(hardware, token2wav_backend_device);
        result.loaded.token2wav_device = token2wav_backend_device == "cpu" || token2wav_dev == nullptr
                                             ? "cpu"
                                             : token2wav_device_for(*token2wav_dev);
    } else {
        result.loaded.token2wav_device = "cpu";
    }
    result.loaded.reason = "loaded config: " + config_path;
    if (overrides.llm_model) {
        result.loaded.reason += "; explicit LLM model override applied";
    }
    if (!skipped.empty()) {
        result.loaded.reason += "; skipped missing modules:";
        for (size_t i = 0; i < skipped.size(); ++i) {
            result.loaded.reason += (i == 0 ? " " : ", ");
            result.loaded.reason += skipped[i];
        }
    }

    result.loaded.placements.clear();
    add_module_placement(result.loaded, "llm", result.loaded.llm_model, result.loaded.llm_quantization,
                         result.loaded.llm_device);
    auto place_optional = [&](const char * key, const optional_module & module, const std::string & model,
                              const std::string & device) {
        if (!module.present && model.empty()) {
            return;
        }
        add_module_placement(result.loaded, key, model, "F16", device, "independent",
                             module.active ? "active" : "skipped");
    };
    place_optional("vision", vision_mod, result.loaded.vision_model, result.loaded.vision_device);
    place_optional("audio", audio_mod, result.loaded.audio_model, result.loaded.audio_device);
    place_optional("tts", tts_mod, result.loaded.tts_model, result.loaded.tts_device);
    place_optional("projector", projector_mod, result.loaded.projector_model, projector_device);
    place_optional("token2wav", token2wav_mod, result.loaded.token2wav_model_dir, token2wav_backend_device);

    result.ok = true;
    return result;
}

std::string format_config(const config & config) {
    std::ostringstream output;
    output << "config_path=" << config.path << '\n';
    output << "reason=" << config.reason << '\n';
    if (!config.primary_device_name.empty()) {
        output << "primary_device=" << config.primary_device_name;
        if (!config.primary_device_description.empty()) {
            output << " (" << config.primary_device_description << ')';
        }
        output << '\n';
        output << "primary_device_free_memory_mib=" << config.primary_device_free_memory / (1024ULL * 1024ULL) << '\n';
        output << "primary_device_total_memory_mib=" << config.primary_device_total_memory / (1024ULL * 1024ULL)
               << '\n';
    }
    output << "llm_model=" << config.llm_model << '\n';
    output << "llm_quantization=" << config.llm_quantization << '\n';
    output << "llm_device=" << config.llm_device << '\n';
    output << "llm_n_gpu_layers=" << config.n_gpu_layers << '\n';
    output << "vision_model=" << config.vision_model << '\n';
    output << "vision_device=" << config.vision_device << '\n';
    output << "audio_model=" << config.audio_model << '\n';
    output << "audio_device=" << config.audio_device << '\n';
    output << "tts_model=" << config.tts_model << '\n';
    output << "tts_device=" << config.tts_device << '\n';
    output << "tts_gpu_layers=" << config.tts_gpu_layers << '\n';
    output << "projector_model=" << config.projector_model << '\n';
    output << "token2wav_model_dir=" << config.token2wav_model_dir << '\n';
    output << "token2wav_device=" << config.token2wav_device << '\n';
    output << "token2wav_threads=" << config.token2wav_threads << '\n';
    output << "execution=" << (config.async_mode ? "async" : "sync") << ','
           << (config.duplex_mode ? "duplex" : "simplex");
    if (config.vpm_batch_encode) {
        output << ",vpm_batch_encode";
    }
    output << ",ctx_size=" << config.n_ctx << '\n';
    output << "placement_plan_count=" << config.placements.size() << '\n';
    for (const auto & placement : config.placements) {
        output << "placement=" << placement.module
               << ",model=" << placement.model
               << ",precision=" << placement.precision
               << ",device=" << placement.device
               << ",backend=" << placement.backend
               << ",execution=" << placement.execution
               << ",status=" << placement.status << '\n';
    }
    return output.str();
}

session_options resolve_session_options(const config * config,
                                        bool                   requested_duplex_mode,
                                        int32_t                requested_tts_gpu_layers,
                                        const std::string &    requested_token2wav_device,
                                        int32_t                requested_token2wav_threads) {
    if (config != nullptr) {
        return { config->duplex_mode, config->async_mode, config->tts_gpu_layers, config->token2wav_device,
                 config->token2wav_threads, true };
    }
    return { requested_duplex_mode, true, requested_tts_gpu_layers, requested_token2wav_device,
             requested_token2wav_threads, false };
}

bool config_accepts_session_mode(const config * config, bool requested_duplex) {
    return config == nullptr || config->duplex_mode == requested_duplex;
}

bool apply_config(common_params & params, const config & config, std::string & error) {
    error.clear();
    params.model.path       = config.llm_model;
    params.n_gpu_layers     = config.n_gpu_layers;
    params.n_ctx            = config.n_ctx;
    params.vpm_model        = config.vision_model;
    params.apm_model        = config.audio_model;
    params.tts_model        = config.tts_model;
    params.projector_model  = config.projector_model;
    params.vpm_batch_encode = config.vpm_batch_encode;

    const auto separator = config.tts_model.find_last_of("/\\");
    params.tts_bin_dir   = separator == std::string::npos ? "." : config.tts_model.substr(0, separator);

    ggml_backend_dev_t llm_device = config.llm_device == "cpu" ?
                                        ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU) :
                                        ggml_backend_dev_by_name(config.llm_device.c_str());
    if (llm_device == nullptr) {
        error = "resolved LLM placement device is unavailable: " + config.llm_device;
        return false;
    }
    params.devices    = { llm_device, nullptr };
    params.split_mode = LLAMA_SPLIT_MODE_NONE;
    params.main_gpu   = 0;
    if (config.llm_device == "cpu") {
        params.n_gpu_layers = 0;
    }
    return true;
}

}  // namespace omni
