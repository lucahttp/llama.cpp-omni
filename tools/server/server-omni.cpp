// Omni streaming HTTP server — standalone omni API endpoints
// Based on the old server.cpp omni handlers, adapted for the new llama.cpp APIs

#include "omni.h"
#include "llama.h"
#include "common.h"
#include "log.h"
#include "arg.h"
#include "sampling.h"
#include "session.h"
#include "ws_handler.h"
#include "delegation-handler.h"

#include <mutex>
#include <thread>
#include <queue>
#include <condition_variable>
#include <fstream>
#include <string>

#include "httplib.h"
// nlohmann/json.hpp and 'using json = nlohmann::json' are already included
// by protocol.h (pulled in transitively via session.h / ws_handler.h).

static json format_error_response(const std::string & message, const std::string & type = "invalid_request_error") {
    return json{{"error", {{"message", message}, {"type", type}}}};
}

template<typename T>
static T json_value(const json & body, const std::string & key, const T & default_value) {
    if (body.contains(key)) {
        try {
            return body.at(key).get<T>();
        } catch (...) {
            return default_value;
        }
    }
    return default_value;
}

static void res_ok(httplib::Response & res, const json & data) {
    res.set_content(data.dump(), "application/json");
}

static void res_error(httplib::Response & res, const json & err) {
    res.status = json_value(err, "code", 500);
    res.set_content(err.dump(), "application/json");
}

static bool server_sent_event(httplib::DataSink & sink, const json & ev) {
    std::string str = "data: " + ev.dump() + "\n\n";
    return sink.write(str.data(), str.size());
}

static std::string parent_dir(const std::string & path) {
    const size_t pos = path.find_last_of("/\\");
    return pos == std::string::npos ? "." : path.substr(0, pos);
}

static bool ensure_omni_model_paths_from_llm(common_params & params) {
    if (params.model.path.empty()) {
        return false;
    }
    const std::string root = parent_dir(params.model.path);
    if (root.empty()) {
        return false;
    }
    if (params.vpm_model.empty()) {
        params.vpm_model = root + "/vision/MiniCPM-o-4_5-vision-F16.gguf";
    }
    if (params.apm_model.empty()) {
        params.apm_model = root + "/audio/MiniCPM-o-4_5-audio-F16.gguf";
    }
    if (params.tts_model.empty()) {
        params.tts_model = root + "/tts/MiniCPM-o-4_5-tts-F16.gguf";
    }
    if (params.tts_bin_dir.empty()) {
        params.tts_bin_dir = root + "/tts";
    }
    return true;
}

struct omni_server_state {
    omni_context * octx = nullptr;    // WS backend uses this as shared_octx
    std::mutex octx_mutex;            // protects omni_context lifecycle + prefill/decode entry
    SessionManager session_mgr;       // WS backend session management

    // Async delegation: a worker thread pool that calls external AI APIs or
    // CLI tools (Claude / OpenAI / gh copilot / agy) without blocking the
    // audio loop. DelegationHandler itself is internally thread-safe; the
    // surrounding state is split so we can lazily init() once and refuse
    // re-init after the worker thread is running.
    //
    // Lifecycle:
    //   1. Server boots with delegation_cfg_initialised = false.
    //   2. First POST /v1/delegation/config sets the config and calls init().
    //      The flag flips to true; the worker thread starts.
    //   3. Subsequent POST /v1/delegation/config returns 409 (config frozen
    //      for this process; restart the server to change).
    //
    // SSE loop integration is intentionally OUT OF SCOPE for this iteration:
    // we expose submit/pop/status over HTTP so callers (and tests) can drive
    // delegation explicitly. The next iteration will wire `pop_result()` into
    // the text_queue polling loop in /v1/stream/decode.
    DelegationConfig delegation_cfg;
    DelegationHandler delegation_handler;
    bool delegation_cfg_initialised = false;
    std::mutex delegation_init_mtx;   // guards the bool + the init() call
};

int main(int argc, char ** argv) {
#ifdef _WIN32
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif
    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    // omni HTTP server is single-session (1:1 duplex), so 1 sequence is enough.
    // common_params defaults n_parallel to -1 ("auto"); each example resolves it
    // itself (see tools/server/server.cpp). Without this, n_seq_max overflows
    // uint32 and trips LLAMA_MAX_SEQ(256) inside llama_context.
    if (params.n_parallel < 0) {
        params.n_parallel = 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    LOG_INF("Omni HTTP server starting...\n");

    // auto-detect omni model paths
    if (!params.vpm_model.empty() || !params.apm_model.empty() || !params.tts_model.empty()) {
        LOG_INF("Using explicit omni model paths from args\n");
    }

    // HTTP server setup
    httplib::Server svr;

    omni_server_state state;

    // GET /health
    svr.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        json health = {{"status", "ok"}, {"engine", "comni"}};
        res.set_header("X-Engine", "comni");
        res_ok(res, health);
    });

    svr.Get("/v1/health", [&](const httplib::Request &, httplib::Response & res) {
        json health = {{"status", "ok"}, {"engine", "comni"}};
        res.set_header("X-Engine", "comni");
        res_ok(res, health);
    });

    // POST /v1/stream/omni_init
    svr.Post("/v1/stream/omni_init", [&](const httplib::Request & req, httplib::Response & res) {
        json data = json::parse(req.body);

        if (!data.contains("msg_type") && !data.contains("media_type")) {
            res_error(res, format_error_response("\"msg_type\" or \"media_type\" must be provided"));
            return;
        }

        int media_type = data.value("msg_type", data.value("media_type", 2));
        bool use_tts   = data.value("use_tts", true);
        bool duplex_mode = data.value("duplex_mode", false);
        int tts_gpu_layers = data.value("tts_gpu_layers", 100);
        std::string token2wav_device = data.value("token2wav_device", "gpu:0");
        std::string output_dir = data.value("output_dir", "./tools/omni/output");
        std::string voice_audio = data.value("voice_audio", "");

        // validate key files
        auto check_file = [&](const std::string & role, const std::string & path) -> bool {
            if (path.empty()) return true;
            std::ifstream f(path);
            if (!f.good()) {
                res_error(res, format_error_response(
                    "omni_init missing required model file (" + role + "): " + path));
                return false;
            }
            return true;
        };

        // Keep legacy HTTP aligned with /backend: the LLM path (-m) anchors the
        // fixed MiniCPM-o sub-model layout; request model_dir is ignored.
        if (!ensure_omni_model_paths_from_llm(params)) {
            res_error(res, format_error_response("LLM model path (-m) is required to derive omni model paths"));
            return;
        }

        if (!check_file("LLM",    params.model.path) ||
            !check_file("vision", params.vpm_model)  ||
            !check_file("audio",  params.apm_model)  ||
            (use_tts && !check_file("tts", params.tts_model))) {
            return;
        }

        {
            std::lock_guard<std::mutex> lock(state.octx_mutex);
            if (state.octx) {
                omni_free(state.octx);
                state.octx = nullptr;
            }
        }

        omni_context * octx = omni_init(&params, media_type, use_tts, params.tts_bin_dir, tts_gpu_layers,
                                         token2wav_device, duplex_mode,
                                         /*existing_model=*/nullptr, /*existing_ctx=*/nullptr, output_dir);
        if (!octx) {
            res_error(res, format_error_response("omni_init failed"));
            return;
        }

        // voice clone / assistant prompt
        if (data.contains("voice_clone_prompt")) octx->omni_voice_clone_prompt = data["voice_clone_prompt"];
        if (data.contains("assistant_prompt")) octx->omni_assistant_prompt = data["assistant_prompt"];

        {
            std::lock_guard<std::mutex> lock(state.octx_mutex);
            state.octx = octx;
        }

        res_ok(res, {{"success", true}});
    });

    // POST /v1/stream/prefill
    svr.Post("/v1/stream/prefill", [&](const httplib::Request & req, httplib::Response & res) {
        json data = json::parse(req.body);

        if (!data.contains("audio_path_prefix") || !data.at("audio_path_prefix").is_string()) {
            res_error(res, format_error_response("\"audio_path_prefix\" must be provided as string"));
            return;
        }
        if (!data.contains("cnt") || !data.at("cnt").is_number_integer()) {
            res_error(res, format_error_response("\"cnt\" must be provided as integer"));
            return;
        }

        {
            std::lock_guard<std::mutex> lock(state.octx_mutex);
            if (state.octx == nullptr) {
                res_error(res, format_error_response("omni context not initialized. call /v1/stream/omni_init first"));
                return;
            }
        }

        std::string audio_path = data.at("audio_path_prefix");
        std::string img_path   = data.value("img_path_prefix", "");
        std::string text       = data.value("text", "");
        int cnt                = data.at("cnt");
        int max_slice_nums     = data.value("max_slice_nums", -1);

        bool ok = false;
        if (state.octx != nullptr) {
            ok = stream_prefill(state.octx, audio_path, img_path, cnt, max_slice_nums, text);
        }

        if (!ok) {
            res_error(res, format_error_response("stream_prefill failed"));
            return;
        }

        res_ok(res, {{"success", true}, {"audio_path_prefix", audio_path}, {"cnt", cnt}});
    });

    // POST /v1/stream/decode (SSE)
    svr.Post("/v1/stream/decode", [&](const httplib::Request & req, httplib::Response & res) {
        json data = json::parse(req.body);

        {
            std::lock_guard<std::mutex> lock(state.octx_mutex);
            if (state.octx == nullptr) {
                res_error(res, format_error_response("omni context not initialized. call /v1/stream/omni_init first"));
                return;
            }
        }

        std::string debug_dir = data.value("debug_dir", "./");
        bool stream = data.value("stream", true);
        int round_idx = data.value("round_idx", -1);

        // length_penalty
        if (data.contains("length_penalty") && data.at("length_penalty").is_number()) {
            float lp = data.at("length_penalty").get<float>();
            if (state.octx != nullptr) {
                state.octx->length_penalty = lp;
            }
        }

        // listen_prob_scale
        if (data.contains("listen_prob_scale") && data.at("listen_prob_scale").is_number()) {
            float lps = data.at("listen_prob_scale").get<float>();
            if (state.octx != nullptr) {
                state.octx->listen_prob_scale = lps;
            }
        }

        // speak_prob_scale
        if (data.contains("speak_prob_scale") && data.at("speak_prob_scale").is_number()) {
            float sps = data.at("speak_prob_scale").get<float>();
            if (state.octx != nullptr) {
                state.octx->speak_prob_scale = sps;
            }
        }

        if (!stream) {
            bool ok = false;
            if (state.octx != nullptr) {
                ok = stream_decode(state.octx, debug_dir, round_idx);
            }
            if (!ok) {
                res_error(res, format_error_response("stream_decode failed"));
                return;
            }
            res_ok(res, {{"success", true}});
            return;
        }

        // SSE streaming
        res.set_chunked_content_provider("text/event-stream",
            [&](size_t, httplib::DataSink & sink) -> bool {
                // reset state
                {
                    std::lock_guard<std::mutex> lock(state.octx->text_mtx);
                    state.octx->text_queue.clear();
                    state.octx->text_done_flag = false;
                    state.octx->text_streaming = true;
                }

                // start decode in background thread
                std::atomic<bool> worker_finished{false};
                std::thread worker([&](std::string dd, int ri) {
                    if (state.octx != nullptr) {
                        (void) stream_decode(state.octx, dd, ri);
                    }
                    worker_finished.store(true);
                }, debug_dir, round_idx);

                // poll text queue
                while (true) {
                    std::unique_lock<std::mutex> lk(state.octx->text_mtx);
                    state.octx->text_cv.wait_for(lk, std::chrono::milliseconds(50), [&]{
                        return !state.octx->text_queue.empty() || state.octx->text_done_flag || worker_finished.load();
                    });

                    while (!state.octx->text_queue.empty()) {
                        std::string frag = std::move(state.octx->text_queue.front());
                        state.octx->text_queue.pop_front();
                        lk.unlock();

                        json ev;
                        if (frag == "__IS_LISTEN__") {
                            ev = {{"content", ""}, {"stop", false}, {"is_listen", true}, {"end_of_turn", true}};
                        } else if (frag == "__END_OF_TURN__") {
                            ev = {{"content", ""}, {"stop", true}, {"is_listen", false}, {"end_of_turn", true}};
                        } else {
                            ev = {{"content", frag}, {"stop", false}, {"is_listen", false}, {"end_of_turn", false}};
                        }

                        if (!server_sent_event(sink, ev)) {
                            if (worker.joinable()) worker.join();
                            return false;
                        }
                        lk.lock();
                    }

                    if (state.octx->text_done_flag || (worker_finished.load() && state.octx->text_queue.empty())) break;
                }

                if (worker.joinable()) worker.join();

                // send done
                static const std::string ev_done = "data: [DONE]\n\n";
                sink.write(ev_done.data(), ev_done.size());
                sink.done();
                return false;
            });
    });

    // POST /v1/stream/update_session_config
    svr.Post("/v1/stream/update_session_config", [&](const httplib::Request & req, httplib::Response & res) {
        json data = json::parse(req.body);
        int media_type = data.value("media_type", -1);

        {
            std::lock_guard<std::mutex> lock(state.octx_mutex);
            if (state.octx == nullptr) {
                res_error(res, format_error_response("omni context not initialized"));
                return;
            }
            if (media_type > 0) {
                state.octx->media_type = media_type;
            }
            if (data.contains("listen_prob_scale") && data.at("listen_prob_scale").is_number()) {
                state.octx->listen_prob_scale = data.at("listen_prob_scale").get<float>();
            }
            if (data.contains("speak_prob_scale") && data.at("speak_prob_scale").is_number()) {
                state.octx->speak_prob_scale = data.at("speak_prob_scale").get<float>();
            }
            if (data.contains("force_listen_count") && data.at("force_listen_count").is_number_integer()) {
                state.octx->force_listen_count = data.at("force_listen_count").get<int>();
                state.octx->force_listen_used = 0;
            }
        }

        res_ok(res, {{"success", true}});
    });

    // POST /v1/stream/break
    svr.Post("/v1/stream/break", [&](const httplib::Request &, httplib::Response & res) {
        {
            std::lock_guard<std::mutex> lock(state.octx_mutex);
            if (state.octx != nullptr) {
                state.octx->break_event.store(true);
            }
        }
        res_ok(res, {{"success", true}});
    });

    //
    // Delegation API
    //
    // These endpoints drive tools/server/delegation-handler.{h,cpp}, an async
    // worker pool that calls Claude / OpenAI APIs and CLI tools (Claude Code,
    // gh copilot, agy) without blocking the audio pipeline.
    //
    // See docs/delegation-architecture.md for the design and the next iteration
    // (SSE-loop integration) that will inject results into /v1/stream/decode.
    //

    // POST /v1/delegation/config
    // Body: JSON matching DelegationConfig fields. Missing fields fall back to
    // the defaults declared in tools/server/delegation-handler.h.
    // Returns 409 if delegation was already initialised in this process;
    // restart the server to change config (worker thread cannot be restarted
    // safely mid-run).
    svr.Post("/v1/delegation/config", [&](const httplib::Request & req, httplib::Response & res) {
        json data;
        try {
            data = json::parse(req.body);
        } catch (const std::exception & e) {
            res_error(res, format_error_response(std::string("invalid JSON: ") + e.what()));
            return;
        }

        std::lock_guard<std::mutex> lock(state.delegation_init_mtx);
        if (state.delegation_cfg_initialised) {
            // Returning 409 instead of silently re-applying the config: the
            // worker thread has already started and there is no clean way to
            // swap its config without leaking threads or losing in-flight
            // requests. Documenting this in the response so callers know.
            res.status = 409;
            res_error(res, format_error_response(
                "delegation already initialised in this process; restart the server to change config"));
            return;
        }

        // Build DelegationConfig from the request, defaulting to whatever
        // DelegationConfig's own defaults provide. This keeps the wire format
        // forgiving: callers only need to send the fields they want to change.
        DelegationConfig cfg = state.delegation_cfg; // copy with defaults

        auto set_str = [&](const char * key, std::string & field) {
            if (data.contains(key) && data.at(key).is_string()) {
                field = data.at(key).get<std::string>();
            }
        };
        auto set_int = [&](const char * key, int & field) {
            if (data.contains(key) && data.at(key).is_number_integer()) {
                field = data.at(key).get<int>();
            }
        };
        auto set_bool = [&](const char * key, bool & field) {
            if (data.contains(key) && data.at(key).is_boolean()) {
                field = data.at(key).get<bool>();
            }
        };

        set_bool("enabled",                              cfg.enabled);
        set_str ("default_provider",                     cfg.default_provider);
        set_str ("claude_api_key_env",                   cfg.claude_api_key_env);
        set_str ("claude_model",                         cfg.claude_model);
        set_int ("claude_max_tokens",                    cfg.claude_max_tokens);
        set_str ("claude_api_base",                      cfg.claude_api_base);
        set_str ("openai_api_key_env",                   cfg.openai_api_key_env);
        set_str ("openai_model",                         cfg.openai_model);
        set_int ("openai_max_tokens",                    cfg.openai_max_tokens);
        set_str ("openai_api_base",                      cfg.openai_api_base);
        set_str ("claude_path",                          cfg.claude_path);
        set_str ("copilot_path",                         cfg.copilot_path);
        set_str ("agy_path",                             cfg.agy_path);
        set_int ("default_timeout_ms",                   cfg.default_timeout_ms);

        if (data.contains("filler_responses") && data.at("filler_responses").is_array()) {
            cfg.filler_responses.clear();
            for (const auto & item : data.at("filler_responses")) {
                if (item.is_string()) {
                    cfg.filler_responses.push_back(item.get<std::string>());
                }
            }
        }

        // init() starts the worker thread. We only do this when enabled=true
        // because the worker has nothing to do otherwise and would just spin
        // on its condition variable.
        if (cfg.enabled) {
            state.delegation_handler.init(cfg);
        }
        state.delegation_cfg = cfg;
        state.delegation_cfg_initialised = true;

        LOG_INF("Delegation configured: enabled=%d default_provider=%s\n",
                (int) cfg.enabled, cfg.default_provider.c_str());

        res_ok(res, {
            {"success", true},
            {"enabled", cfg.enabled},
            {"default_provider", cfg.default_provider}
        });
    });

    // POST /v1/delegation/execute
    // Body: { "type": "API_CLAUDE"|"API_OPENAI"|..., "messages": [...], "user_query": "...", "cli_command": "...", "timeout_ms": 30000 }
    // Returns: { "success": true, "request_id": "...", "pending": N }
    svr.Post("/v1/delegation/execute", [&](const httplib::Request & req, httplib::Response & res) {
        json data;
        try {
            data = json::parse(req.body);
        } catch (const std::exception & e) {
            res_error(res, format_error_response(std::string("invalid JSON: ") + e.what()));
            return;
        }

        if (!state.delegation_handler.is_enabled()) {
            // 412 Precondition Failed: delegation was not configured (or was
            // configured with enabled=false). Caller must POST /v1/delegation/config first.
            res.status = 412;
            res_error(res, format_error_response(
                "delegation is not enabled; POST /v1/delegation/config first"));
            return;
        }

        if (!data.contains("type") || !data.at("type").is_string()) {
            res_error(res, format_error_response("\"type\" must be a string"));
            return;
        }

        const std::string type_str = data.at("type").get<std::string>();
        DelegationType type = DelegationType::NONE;
        if      (type_str == "API_CLAUDE")       type = DelegationType::API_CLAUDE;
        else if (type_str == "API_OPENAI")       type = DelegationType::API_OPENAI;
        else if (type_str == "CLI_CLAUDE_CODE")  type = DelegationType::CLI_CLAUDE_CODE;
        else if (type_str == "CLI_COPILOT")      type = DelegationType::CLI_COPILOT;
        else if (type_str == "CLI_AGY")          type = DelegationType::CLI_AGY;
        else {
            res_error(res, format_error_response(
                "unknown delegation type: " + type_str));
            return;
        }

        DelegationRequest dreq;
        dreq.type        = type;
        dreq.user_query  = data.value("user_query",  std::string());
        dreq.cli_command = data.value("cli_command", std::string());
        dreq.timeout_ms  = data.value("timeout_ms",  state.delegation_cfg.default_timeout_ms);
        dreq.start_time  = std::chrono::steady_clock::now();

        if (data.contains("messages") && data.at("messages").is_array()) {
            for (const auto & m : data.at("messages")) {
                if (!m.is_object()) continue;
                DelegationMessage dm;
                dm.role    = m.value("role",    std::string());
                dm.content = m.value("content", std::string());
                if (!dm.role.empty() && !dm.content.empty()) {
                    dreq.messages.push_back(std::move(dm));
                }
            }
        }

        const std::string request_id = state.delegation_handler.delegate_async(dreq);
        if (request_id.empty()) {
            // delegate_async() returns "" when !is_enabled(). We already checked
            // that above, but be defensive: another thread could race-disable.
            res.status = 412;
            res_error(res, format_error_response("delegation is not enabled"));
            return;
        }

        LOG_INF("Delegation submitted: id=%s type=%s\n",
                request_id.c_str(), type_str.c_str());

        res_ok(res, {
            {"success",    true},
            {"request_id", request_id},
            {"pending",    state.delegation_handler.pending_count()},
            {"filler",     state.delegation_handler.get_filler_response()}
        });
    });

    // GET /v1/delegation/status
    // Returns the current state of the delegation subsystem and (by default)
    // drains any finished results in the same call. The objective calls for
    // exactly three endpoints, so we collapse peek+drain into one route via
    // a query param.
    //
    // Default (drain):  GET /v1/delegation/status
    //   { "success": true, "enabled": bool, "pending": N, "default_provider": "...",
    //     "count": M, "results": [ { "request_id", "success", "text", "error", "elapsed_ms" }, ... ] }
    //
    // Peek-only:        GET /v1/delegation/status?peek=1
    //   Same shape but `results` is always [] and the internal queue is not
    //   touched. Use from monitoring/health checks that must not consume work.
    svr.Get("/v1/delegation/status", [&](const httplib::Request & req, httplib::Response & res) {
        const bool enabled = state.delegation_handler.is_enabled();

        // Any truthy value of `peek` (other than "0") disables draining.
        const bool peek = req.has_param("peek") && req.get_param_value("peek") != "0";

        size_t pending = 0;
        json results_arr = json::array();

        if (enabled) {
            pending = state.delegation_handler.pending_count();

            if (!peek) {
                // Drain all currently-available results. Each result carries
                // its own elapsed_ms (the time the worker spent on it) — this
                // is what the objective meant by "devolver ... elapsed_ms".
                while (true) {
                    DelegationResult r;
                    if (!state.delegation_handler.pop_result(r)) break;
                    results_arr.push_back({
                        {"request_id", r.request_id},
                        {"success",    r.success},
                        {"text",       r.text},
                        {"error",      r.error},
                        {"elapsed_ms", r.elapsed_ms}
                    });
                }
            }
        }

        res_ok(res, {
            {"success",          true},
            {"enabled",          enabled},
            {"pending",          pending},
            {"default_provider", state.delegation_cfg.default_provider},
            {"count",            results_arr.size()},
            {"results",          results_arr}
        });
    });

    //
    // Backend Protocol (WebSocket + HTTP unary)
    //
    svr.WebSocket("/backend", [&](const httplib::Request &, httplib::ws::WebSocket & ws) {
        handle_ws_backend(ws, state.session_mgr, params,
                          /*model*/nullptr, /*ctx*/nullptr,
                          state.octx, state.octx_mutex);
    });

    svr.Post("/sessions/:session_id/close", [&](const httplib::Request & req, httplib::Response & res) {
        std::string session_id = req.path_params.at("session_id");
        LOG_INF("Close session requested: %s\n", session_id.c_str());

        auto * session = state.session_mgr.get(session_id);
        if (!session || session->state != SessionState::ACTIVE) {
            res_error(res, format_error_response("session not found", "not_found"));
            res.status = 404;
            return;
        }

        state.session_mgr.request_transport_close(session_id);

        // close is a completion primitive: do not return until inference
        // threads are stopped and the shared omni_context is safe to reuse.
        {
            std::lock_guard<std::mutex> octx_lock(state.octx_mutex);
            auto * closing = state.session_mgr.get(session_id);
            if (closing && closing->octx) {
                closing->octx->break_event = true;
                {
                    std::lock_guard<std::mutex> lk(closing->octx->text_mtx);
                    closing->octx->text_queue.clear();
                    closing->octx->text_done_flag = true;
                }
                closing->octx->text_cv.notify_all();
                omni_prepare_for_reuse(closing->octx);
            }

            state.session_mgr.close(session_id);
        }

        json resp;
        resp["ok"] = true;
        resp["session_id"] = session_id;
        resp["closed"] = true;
        res_ok(res, resp);
    });

    // start server
    const char * host = params.hostname.empty() ? "0.0.0.0" : params.hostname.c_str();
    LOG_INF("HTTP server listening on %s:%d (port=%d)...\n", host, params.port, params.port);
    if (!svr.listen(host, params.port)) {
        LOG_ERR("svr.listen failed on %s:%d (is_valid=%d)\n", host, params.port, (int)svr.is_valid());
    }

    // cleanup
    {
        std::lock_guard<std::mutex> lock(state.octx_mutex);
        if (state.octx) {
            omni_free(state.octx);
            state.octx = nullptr;
        }
    }
    llama_backend_free();
#ifdef _WIN32
    WSACleanup();
#endif

    return 0;
}
