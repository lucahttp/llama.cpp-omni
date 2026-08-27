#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct common_params;

namespace omni {

struct device {
    std::string name;
    std::string description;
    bool        is_accelerator = false;
    uint64_t    free_memory    = 0;
    uint64_t    total_memory   = 0;
    std::string type;
};

struct hardware_snapshot {
    std::vector<device> devices;
};

struct module_placement {
    std::string module;
    std::string model;
    std::string precision;
    std::string device;
    std::string backend;
    std::string execution;
    std::string status;
};

struct config {
    std::string              path;
    std::string              reason;
    std::string              primary_device_name;
    std::string              primary_device_description;
    uint64_t                 primary_device_free_memory  = 0;
    uint64_t                 primary_device_total_memory = 0;
    std::string              llm_model;
    std::string              llm_quantization;
    std::string              llm_device;
    std::string              vision_model;
    std::string              vision_device;
    std::string              audio_model;
    std::string              audio_device;
    std::string              tts_model;
    std::string              tts_device;
    std::string              projector_model;
    std::string              token2wav_model_dir;
    int32_t                  n_gpu_layers      = 0;
    int32_t                  tts_gpu_layers    = 0;
    std::string              token2wav_device  = "cpu";
    int32_t                  token2wav_threads = 8;
    int32_t                  n_ctx             = 0;
    bool                     duplex_mode       = false;
    bool                     async_mode        = false;
    bool                     vpm_batch_encode  = false;
    std::vector<module_placement> placements;
};

struct config_overrides {
    std::optional<std::string> llm_model;
    std::optional<std::string> vision_model;
    std::optional<std::string> audio_model;
    std::optional<std::string> tts_model;
    std::optional<std::string> projector_model;
    std::optional<std::string> llm_device;
    std::optional<std::string> vision_device;
    std::optional<std::string> audio_device;
    std::optional<std::string> tts_device;
    std::optional<std::string> projector_device;
    std::optional<std::string> token2wav_device;
    std::optional<int32_t>     token2wav_threads;
    std::optional<int32_t>     n_gpu_layers;
    std::optional<int32_t>     n_ctx;
    std::optional<bool>        vpm_batch_encode;
};

struct config_result {
    bool        ok = false;
    config      loaded;
    std::string error;
};

struct session_options {
    bool        duplex_mode       = false;
    bool        async_mode        = true;
    int32_t     tts_gpu_layers    = 100;
    std::string token2wav_device  = "gpu:0";
    int32_t     token2wav_threads = 8;
    bool        config_locked     = false;
};

hardware_snapshot detect_hardware_snapshot();

std::string default_config_dir();
std::string resolve_config_path(const std::string & name_or_path);
std::string select_config_name(const hardware_snapshot & hardware, std::string * error = nullptr);
std::string resolve_config_request(const std::string & name_or_path,
                                   const hardware_snapshot & hardware,
                                   std::string * error = nullptr);

config_result load_config(const std::string &       config_path,
                          const hardware_snapshot & hardware,
                          const std::string &       model_root,
                          const config_overrides &  overrides = {});

std::string format_config(const config & config);

bool prepare_config(common_params & params, std::optional<config> & out, std::string & error);

session_options resolve_session_options(const config *      config,
                                        bool                requested_duplex_mode,
                                        int32_t             requested_tts_gpu_layers,
                                        const std::string & requested_token2wav_device,
                                        int32_t             requested_token2wav_threads = 8);

bool config_accepts_session_mode(const config * config, bool requested_duplex);

bool apply_config(common_params & params, const config & config, std::string & error);

}  // namespace omni
