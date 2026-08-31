#include "delegation-handler.h"

#include "common.h"
#include "log.h"

#include <httplib.h>
#include <iostream>
#include <sstream>
#include <chrono>
#include <random>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#include <process.h>
#define popen _popen
#define pclose _pclose
#else
#include <sys/wait.h>
#endif

// ==================== Utility ====================

static std::string get_env(const std::string& name, const std::string& default_val = "") {
    const char* val = std::getenv(name.c_str());
    return val ? val : default_val;
}

// ==================== Delegation Handler ====================

DelegationHandler::DelegationHandler() {}

DelegationHandler::~DelegationHandler() {
    cancel_all();
    running_ = false;
    worker_cv_.notify_all();
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

void DelegationHandler::init(const DelegationConfig& config) {
    config_ = config;

    if (config.enabled) {
        running_ = true;
        worker_thread_ = std::thread(&DelegationHandler::worker_loop, this);
        LOG_INF("DelegationHandler initialized with %zu filler responses\n",
                config.filler_responses.size());
    }
}

void DelegationHandler::worker_loop() {
    while (running_) {
        std::pair<std::string, DelegationRequest> work;

        {
            // worker_cv_ is signalled under work_mtx_; the predicate inspects
            // work_queue_, so the wait must hold work_mtx_, not pending_mtx_.
            // Using pending_mtx_ here would force submit_work to serialise on
            // the (unrelated) pending_ map.
            std::unique_lock<std::mutex> lock(work_mtx_);
            worker_cv_.wait_for(lock, std::chrono::seconds(1), [this]() {
                return !work_queue_.empty() || !running_;
            });

            if (!running_) break;
            if (work_queue_.empty()) continue;

            work = std::move(work_queue_.front());
            work_queue_.pop();
        }

        DelegationResult result;
        result.request_id = work.first;

        auto start = std::chrono::steady_clock::now();

        switch (work.second.type) {
            case DelegationType::API_CLAUDE:
                result = execute_claude_api(work.second);
                break;
            case DelegationType::API_OPENAI:
                result = execute_openai_api(work.second);
                break;
            case DelegationType::CLI_CLAUDE_CODE:
                result = execute_cli_claude(work.second);
                break;
            case DelegationType::CLI_COPILOT:
                result = execute_cli_copilot(work.second);
                break;
            case DelegationType::CLI_AGY:
                result = execute_cli_agy(work.second);
                break;
            default:
                result.success = false;
                result.error = "Unknown delegation type";
                break;
        }

        auto end = std::chrono::steady_clock::now();
        result.elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        {
            std::lock_guard<std::mutex> lock(results_mtx_);
            results_.push(result);
        }

        {
            std::lock_guard<std::mutex> lock(pending_mtx_);
            pending_.erase(work.first);
        }
    }
}

void DelegationHandler::submit_work(const std::string& id, const DelegationRequest& req) {
    {
        std::lock_guard<std::mutex> lock(pending_mtx_);
        pending_[id] = req;
    }
    {
        // work_queue_ is independent of pending_; serialising it on the same
        // mutex is a code smell (and historically caused submit_work to grab
        // pending_mtx_ twice in a row). Lock work_mtx_ here and notify.
        std::lock_guard<std::mutex> lock(work_mtx_);
        work_queue_.push({id, req});
    }
    worker_cv_.notify_one();
}

std::string DelegationHandler::delegate_async(const DelegationRequest& req) {
    if (!config_.enabled) {
        return "";
    }

    std::string id = "delegation_" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()
    );

    submit_work(id, req);

    LOG_INF("DelegationHandler: submitted request %s of type %d\n",
            id.c_str(), (int)req.type);

    return id;
}

bool DelegationHandler::has_result() const {
    std::lock_guard<std::mutex> lock(results_mtx_);
    return !results_.empty();
}

bool DelegationHandler::pop_result(DelegationResult& out) {
    std::lock_guard<std::mutex> lock(results_mtx_);
    if (results_.empty()) {
        return false;
    }
    out = results_.front();
    results_.pop();
    return true;
}

size_t DelegationHandler::pending_count() const {
    std::lock_guard<std::mutex> lock(pending_mtx_);
    return pending_.size();
}

void DelegationHandler::cancel(const std::string& request_id) {
    // Cancellation is not yet implemented: DelegationRequest is no longer
    // copyable (std::atomic<bool> removed), so we cannot store cancellation
    // state in the request struct. A separate std::unordered_map or similar
    // would be needed. Log and do nothing for now.
    LOG_INF("DelegationHandler: cancel(%s) called but not implemented\n", request_id.c_str());
}

void DelegationHandler::cancel_all() {
    // See cancel() above. Not implemented.
    LOG_INF("DelegationHandler: cancel_all() called but not implemented\n");
}

std::string DelegationHandler::get_filler_response() const {
    if (config_.filler_responses.empty()) {
        return "Un momento...";
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, config_.filler_responses.size() - 1);
    return config_.filler_responses[dis(gen)];
}

DelegationType DelegationHandler::token_to_delegation_type(
        int token_id,
        int delegate_tok,
        int claude_code_tok,
        int copilot_tok,
        int agy_tok) {
    if (token_id == delegate_tok) return DelegationType::API_CLAUDE;
    if (token_id == claude_code_tok) return DelegationType::CLI_CLAUDE_CODE;
    if (token_id == copilot_tok) return DelegationType::CLI_COPILOT;
    if (token_id == agy_tok) return DelegationType::CLI_AGY;
    return DelegationType::NONE;
}

// ==================== API Implementations ====================

DelegationResult DelegationHandler::execute_claude_api(const DelegationRequest& req) {
    DelegationResult result;
    result.request_id = req.id;

    std::string api_key = get_env(config_.claude_api_key_env);
    if (api_key.empty()) {
        result.success = false;
        result.error = "Claude API key not found";
        return result;
    }

    httplib::Client cli(config_.claude_api_base);
    // cpp-httplib 0.6.0 uses separate sec/usec setters rather than set_timeout_ms.
    const auto timeout_sec = req.timeout_ms / 1000;
    cli.set_connection_timeout(timeout_sec, 0);
    cli.set_read_timeout(timeout_sec);
    cli.set_write_timeout(timeout_sec);

    // Build request JSON
    json req_json;
    req_json["model"] = config_.claude_model;
    req_json["max_tokens"] = config_.claude_max_tokens;

    json messages_arr;
    for (const auto& msg : req.messages) {
        json m;
        m["role"] = msg.role;
        m["content"] = msg.content;
        messages_arr.push_back(m);
    }
    if (!req.user_query.empty()) {
        json q;
        q["role"] = "user";
        q["content"] = req.user_query;
        messages_arr.push_back(q);
    }
    req_json["messages"] = messages_arr;

    // Anthropic Messages API expects the API key in `x-api-key`, not in
    // `Authorization: Bearer`. Both forms are accepted, but `x-api-key` is
    // the canonical header in the public docs and the one we want to send.
    //
    // `anthropic-dangerous-direct-browser-access` is only for browser-direct
    // calls (no backend proxy); we are a server-to-server caller, so we omit it.
    auto res = cli.Post("/v1/messages",
        {
            {"x-api-key", api_key},
            {"Content-Type", "application/json"},
            {"anthropic-version", "2023-06-01"}
        },
        req_json.dump(), "application/json");

    if (res && res->status == 200) {
        try {
            auto resp_json = json::parse(res->body);
            if (resp_json.contains("content")) {
                for (const auto& block : resp_json["content"]) {
                    if (block.contains("text")) {
                        result.text += block["text"].get<std::string>();
                    }
                }
            }
            result.success = true;
        } catch (const std::exception& e) {
            result.success = false;
            result.error = std::string("JSON parse error: ") + e.what();
        }
    } else {
        result.success = false;
        result.error = res ? ("HTTP " + std::to_string(res->status)) : "Connection failed";
    }

    return result;
}

DelegationResult DelegationHandler::execute_openai_api(const DelegationRequest& req) {
    DelegationResult result;
    result.request_id = req.id;

    std::string api_key = get_env(config_.openai_api_key_env);
    if (api_key.empty()) {
        result.success = false;
        result.error = "OpenAI API key not found";
        return result;
    }

    httplib::Client cli(config_.openai_api_base);
    // cpp-httplib 0.6.0 uses separate sec/usec setters rather than set_timeout_ms.
    const auto timeout_sec = req.timeout_ms / 1000;
    cli.set_connection_timeout(timeout_sec, 0);
    cli.set_read_timeout(timeout_sec);
    cli.set_write_timeout(timeout_sec);

    json req_json;
    req_json["model"] = config_.openai_model;
    req_json["max_tokens"] = config_.openai_max_tokens;
    req_json["stream"] = false;

    json messages_arr;
    for (const auto& msg : req.messages) {
        json m;
        m["role"] = msg.role;
        m["content"] = msg.content;
        messages_arr.push_back(m);
    }
    if (!req.user_query.empty()) {
        json q;
        q["role"] = "user";
        q["content"] = req.user_query;
        messages_arr.push_back(q);
    }
    req_json["messages"] = messages_arr;

    auto res = cli.Post("/v1/chat/completions",
        {
            {"Authorization", "Bearer " + api_key},
            {"Content-Type", "application/json"}
        },
        req_json.dump(), "application/json");

    if (res && res->status == 200) {
        try {
            auto resp_json = json::parse(res->body);
            if (resp_json.contains("choices") && !resp_json["choices"].empty()) {
                result.text = resp_json["choices"][0]["message"]["content"].get<std::string>();
            }
            result.success = true;
        } catch (const std::exception& e) {
            result.success = false;
            result.error = std::string("JSON parse error: ") + e.what();
        }
    } else {
        result.success = false;
        result.error = res ? ("HTTP " + std::to_string(res->status)) : "Connection failed";
    }

    return result;
}

// ==================== CLI Implementations ====================

static std::string exec_sync(const std::string& cmd) {
    std::string output;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        return "";
    }

    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }

    pclose(pipe);
    return output;
}

DelegationResult DelegationHandler::execute_cli_claude(const DelegationRequest& req) {
    DelegationResult result;
    result.request_id = req.id;

    // Build prompt from messages
    std::string prompt;
    for (const auto& msg : req.messages) {
        prompt += msg.role + ": " + msg.content + "\n";
    }
    if (!req.user_query.empty()) {
        prompt += "user: " + req.user_query + "\n";
    }

    std::string cmd = config_.claude_path + " --print";
    result.text = exec_sync(cmd);
    result.success = !result.text.empty();
    if (result.text.empty()) {
        result.error = "Claude CLI returned empty output or not found";
    }

    return result;
}

DelegationResult DelegationHandler::execute_cli_copilot(const DelegationRequest& req) {
    DelegationResult result;
    result.request_id = req.id;

    std::string cmd = config_.copilot_path + " copilot suggest \"" + req.user_query + "\"";
    result.text = exec_sync(cmd);
    result.success = !result.text.empty();
    if (result.text.empty()) {
        result.error = "GitHub Copilot CLI returned empty output or not found";
    }

    return result;
}

DelegationResult DelegationHandler::execute_cli_agy(const DelegationRequest& req) {
    DelegationResult result;
    result.request_id = req.id;

    std::string cmd = config_.agy_path + " " + req.cli_command;
    result.text = exec_sync(cmd);
    result.success = !result.text.empty();
    if (result.text.empty()) {
        result.error = "Antigravity CLI returned empty output or not found";
    }

    return result;
}
