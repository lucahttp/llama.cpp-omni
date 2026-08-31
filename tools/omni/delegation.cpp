#include "delegation.h"

#include "common.h"
#include "log.h"

#include <curl/curl.h>
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

static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* output) {
    size_t totalSize = size * nmemb;
    output->append((char*)contents, totalSize);
    return totalSize;
}

static std::string get_env(const std::string& name, const std::string& default_val = "") {
    const char* val = std::getenv(name.c_str());
    return val ? val : default_val;
}

// ==================== Claude API Client ====================

ClaudeAPIClient::ClaudeAPIClient(const DelegationConfig& config) : config_(config), initialized_(false) {
    api_key_ = get_env(config.claude_api_key_env);
    initialized_ = !api_key_.empty();
    if (!initialized_) {
        LOG_WRN("Claude API key not found in environment variable %s\n", config.claude_api_key_env.c_str());
    }
}

DelegationResult ClaudeAPIClient::execute(const DelegationRequest& req) {
    DelegationResult result;
    result.request_id = req.id;

    if (!initialized_) {
        result.success = false;
        result.error = "Claude API client not initialized (missing API key)";
        return result;
    }

    CURL* curl = curl_easy_init();
    if (!curl) {
        result.success = false;
        result.error = "Failed to initialize CURL";
        return result;
    }

    auto start = std::chrono::steady_clock::now();

    // Build request JSON
    nlohmann::json req_json;
    req_json["model"] = config_.claude_model;
    req_json["max_tokens"] = config_.claude_max_tokens;

    // Build messages array
    nlohmann::json messages;
    for (const auto& msg : req.messages) {
        nlohmann::json m;
        m["role"] = msg.role;
        m["content"] = msg.content;
        messages.push_back(m);
    }
    // Add current query if provided
    if (!req.user_query.empty()) {
        nlohmann::json q;
        q["role"] = "user";
        q["content"] = req.user_query;
        messages.push_back(q);
    }
    req_json["messages"] = messages;

    std::string json_str = req_json.dump();
    std::string response_data;

    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, ("Authorization: Bearer " + api_key_).c_str());
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, "anthropic-version: 2023-06-01");
    headers = curl_slist_append(headers, "anthropic-dangerous-direct-browser-access: true");

    curl_easy_setopt(curl, CURLOPT_URL, (config_.claude_api_base + "/v1/messages").c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_str.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, req.timeout_ms);

    CURLcode res = curl_easy_perform(curl);

    if (res != CURLE_OK) {
        result.success = false;
        result.error = std::string("CURL error: ") + curl_easy_strerror(res);
    } else {
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

        if (http_code == 200) {
            try {
                auto resp_json = nlohmann::json::parse(response_data);
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
            result.error = std::string("HTTP ") + std::to_string(http_code) + ": " + response_data;
        }
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    auto end = std::chrono::steady_clock::now();
    result.elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    return result;
}

// ==================== OpenAI API Client ====================

OpenAIClient::OpenAIClient(const DelegationConfig& config) : config_(config), initialized_(false) {
    api_key_ = get_env(config.openai_api_key_env);
    initialized_ = !api_key_.empty();
    if (!initialized_) {
        LOG_WRN("OpenAI API key not found in environment variable %s\n", config.openai_api_key_env.c_str());
    }
}

DelegationResult OpenAIClient::execute(const DelegationRequest& req) {
    DelegationResult result;
    result.request_id = req.id;

    if (!initialized_) {
        result.success = false;
        result.error = "OpenAI API client not initialized (missing API key)";
        return result;
    }

    CURL* curl = curl_easy_init();
    if (!curl) {
        result.success = false;
        result.error = "Failed to initialize CURL";
        return result;
    }

    auto start = std::chrono::steady_clock::now();

    // Build request JSON
    nlohmann::json req_json;
    req_json["model"] = config_.openai_model;
    req_json["max_tokens"] = config_.openai_max_tokens;
    req_json["stream"] = false;

    // Build messages array
    nlohmann::json messages;
    for (const auto& msg : req.messages) {
        nlohmann::json m;
        m["role"] = msg.role;
        m["content"] = msg.content;
        messages.push_back(m);
    }
    if (!req.user_query.empty()) {
        nlohmann::json q;
        q["role"] = "user";
        q["content"] = req.user_query;
        messages.push_back(q);
    }
    req_json["messages"] = messages;

    std::string json_str = req_json.dump();
    std::string response_data;

    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, ("Authorization: Bearer " + api_key_).c_str());
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, (config_.openai_api_base + "/v1/chat/completions").c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_str.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, req.timeout_ms);

    CURLcode res = curl_easy_perform(curl);

    if (res != CURLE_OK) {
        result.success = false;
        result.error = std::string("CURL error: ") + curl_easy_strerror(res);
    } else {
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

        if (http_code == 200) {
            try {
                auto resp_json = nlohmann::json::parse(response_data);
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
            result.error = std::string("HTTP ") + std::to_string(http_code) + ": " + response_data;
        }
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    auto end = std::chrono::steady_clock::now();
    result.elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    return result;
}

// ==================== CLI Executor ====================

CLIExecutor::CLIExecutor(const DelegationConfig& config) : config_(config) {}

std::future<DelegationResult> CLIExecutor::exec_claude(
        const std::vector<DelegationMessage>& messages,
        const std::string& task,
        output_cb_t on_output) {

    return std::async(std::launch::async, [this, messages, task, on_output]() {
        DelegationResult result;
        result.request_id = "claude-cli";

        // Build prompt from messages
        std::string prompt;
        for (const auto& msg : messages) {
            prompt += msg.role + ": " + msg.content + "\n";
        }
        prompt += "user: " + task + "\n";

        // Execute: claude --print <prompt>
        std::string cmd = config_.claude_path + " --print";
        result.text = exec_sync(cmd, on_output);
        result.success = !result.text.empty();
        if (result.text.empty()) {
            result.error = "Claude CLI returned empty output or not found";
        }

        return result;
    });
}

std::future<DelegationResult> CLIExecutor::exec_copilot(
        const std::string& prompt,
        output_cb_t on_output) {

    return std::async(std::launch::async, [this, prompt, on_output]() {
        DelegationResult result;
        result.request_id = "copilot-cli";

        // Execute: gh copilot suggest "<prompt>"
        std::string cmd = config_.copilot_path + " copilot suggest \"" + prompt + "\"";
        result.text = exec_sync(cmd, on_output);
        result.success = !result.text.empty();
        if (result.text.empty()) {
            result.error = "GitHub Copilot CLI returned empty output or not found";
        }

        return result;
    });
}

std::future<DelegationResult> CLIExecutor::exec_agy(
        const std::string& command,
        output_cb_t on_output) {

    return std::async(std::launch::async, [this, command, on_output]() {
        DelegationResult result;
        result.request_id = "agy-cli";

        // Execute: agy <command>
        std::string cmd = config_.agy_path + " " + command;
        result.text = exec_sync(cmd, on_output);
        result.success = !result.text.empty();
        if (result.text.empty()) {
            result.error = "Antigravity CLI returned empty output or not found";
        }

        return result;
    });
}

std::string CLIExecutor::exec_sync(const std::string& cmd, output_cb_t on_output) {
    std::string output;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        return "";
    }

    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        std::string line(buffer);
        output += line;
        if (on_output) {
            on_output(line, false);
        }
    }

    int status = pclose(pipe);
    if (status != 0) {
        if (on_output) {
            on_output("Command exited with status " + std::to_string(status), true);
        }
    }

    return output;
}

// ==================== Delegation Manager ====================

DelegationManager::DelegationManager() {}

DelegationManager::~DelegationManager() {
    cancel_all();
    running_ = false;
    worker_cv_.notify_all();
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

void DelegationManager::init(const DelegationConfig& config) {
    config_ = config;

    if (config.enabled) {
        // Initialize API clients
        claude_client_ = std::make_unique<ClaudeAPIClient>(config);
        openai_client_ = std::make_unique<OpenAIClient>(config);
        cli_executor_ = std::make_unique<CLIExecutor>(config);

        // Start worker thread
        running_ = true;
        worker_thread_ = std::thread(&DelegationManager::worker_loop, this);

        LOG_INF("DelegationManager initialized with %zu filler responses\n",
                config.filler_responses.size());
    }
}

void DelegationManager::worker_loop() {
    while (running_) {
        std::pair<std::string, DelegationRequest> work;

        {
            std::unique_lock<std::mutex> lock(pending_mtx_);
            worker_cv_.wait_for(lock, std::chrono::seconds(1), [this]() {
                return !work_queue_.empty() || !running_;
            });

            if (!running_) break;
            if (work_queue_.empty()) continue;

            work = std::move(work_queue_.front());
            work_queue_.pop();
        }

        // Execute the request
        DelegationResult result;
        result.request_id = work.first;

        auto start = std::chrono::steady_clock::now();

        switch (work.second.type) {
            case DelegationType::API_CLAUDE:
                if (claude_client_) {
                    result = claude_client_->execute(work.second);
                }
                break;

            case DelegationType::API_OPENAI:
                if (openai_client_) {
                    result = openai_client_->execute(work.second);
                }
                break;

            case DelegationType::CLI_CLAUDE_CODE: {
                if (cli_executor_) {
                    auto future = cli_executor_->exec_claude(
                        work.second.messages,
                        work.second.user_query,
                        [](const std::string&, bool) {}  // ignore output in async mode
                    );
                    result = future.get();
                }
                break;
            }

            case DelegationType::CLI_COPILOT: {
                if (cli_executor_) {
                    auto future = cli_executor_->exec_copilot(
                        work.second.user_query,
                        [](const std::string&, bool) {}
                    );
                    result = future.get();
                }
                break;
            }

            case DelegationType::CLI_AGY: {
                if (cli_executor_) {
                    auto future = cli_executor_->exec_agy(
                        work.second.cli_command,
                        [](const std::string&, bool) {}
                    );
                    result = future.get();
                }
                break;
            }

            default:
                result.success = false;
                result.error = "Unknown delegation type";
                break;
        }

        auto end = std::chrono::steady_clock::now();
        result.elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        // Store result
        {
            std::lock_guard<std::mutex> lock(results_mtx_);
            results_.push(result);
        }

        // Remove from pending
        {
            std::lock_guard<std::mutex> lock(pending_mtx_);
            pending_.erase(work.first);
        }
    }
}

void DelegationManager::submit_work(const std::string& id, const DelegationRequest& req) {
    {
        std::lock_guard<std::mutex> lock(pending_mtx_);
        pending_[id] = req;
    }
    {
        std::lock_guard<std::mutex> lock(pending_mtx_);
        work_queue_.push({id, req});
    }
    worker_cv_.notify_one();
}

std::string DelegationManager::delegate_async(const DelegationRequest& req) {
    if (!config_.enabled) {
        return "";
    }

    std::string id = "delegation_" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()
    );

    submit_work(id, req);

    LOG_INF("DelegationManager: submitted request %s of type %d\n",
            id.c_str(), (int)req.type);

    return id;
}

bool DelegationManager::has_result() const {
    std::lock_guard<std::mutex> lock(results_mtx_);
    return !results_.empty();
}

bool DelegationManager::pop_result(DelegationResult& out) {
    std::lock_guard<std::mutex> lock(results_mtx_);
    if (results_.empty()) {
        return false;
    }
    out = results_.front();
    results_.pop();
    return true;
}

size_t DelegationManager::pending_count() const {
    std::lock_guard<std::mutex> lock(pending_mtx_);
    return pending_.size();
}

void DelegationManager::cancel(const std::string& request_id) {
    std::lock_guard<std::mutex> lock(pending_mtx_);
    auto it = pending_.find(request_id);
    if (it != pending_.end()) {
        it->second.cancelled = true;
        LOG_INF("DelegationManager: cancelled request %s\n", request_id.c_str());
    }
}

void DelegationManager::cancel_all() {
    std::lock_guard<std::mutex> lock(pending_mtx_);
    for (auto& pair : pending_) {
        pair.second.cancelled = true;
    }
    LOG_INF("DelegationManager: cancelled all pending requests\n");
}

std::string DelegationManager::get_filler_response() const {
    if (config_.filler_responses.empty()) {
        return "Un momento...";
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, config_.filler_responses.size() - 1);
    return config_.filler_responses[dis(gen)];
}

DelegationType DelegationManager::token_to_delegation_type(
        llama_token token,
        llama_token delegate_tok,
        llama_token claude_code_tok,
        llama_token copilot_tok,
        llama_token agy_tok) {
    if (token == delegate_tok) return DelegationType::API_CLAUDE;  // Default to Claude
    if (token == claude_code_tok) return DelegationType::CLI_CLAUDE_CODE;
    if (token == copilot_tok) return DelegationType::CLI_COPILOT;
    if (token == agy_tok) return DelegationType::CLI_AGY;
    return DelegationType::NONE;
}

// ==================== Delegation Detection ====================

DelegationType detect_delegation_trigger(
        struct omni_context* ctx,
        llama_token token) {

    if (!ctx) return DelegationType::NONE;

    return DelegationManager::token_to_delegation_type(
        token,
        ctx->special_token_delegate,
        ctx->special_token_claude_code,
        ctx->special_token_copilot,
        ctx->special_token_agy
    );
}

bool handle_delegation_trigger(
        struct omni_context* ctx,
        DelegationType type,
        const std::string& user_query,
        DelegationManager* mgr) {

    if (!ctx || !mgr || !mgr->is_enabled()) {
        return false;
    }

    if (type == DelegationType::NONE) {
        return false;
    }

    DelegationRequest req;
    req.type = type;
    req.user_query = user_query;
    req.timeout_ms = 30000;
    req.start_time = std::chrono::steady_clock::now();

    std::string id = mgr->delegate_async(req);

    if (!id.empty()) {
        std::lock_guard<std::mutex> lock(ctx->delegation_mtx);
        ctx->delegation_pending[id] = user_query;
        ctx->delegation_in_progress = true;
        ctx->active_delegation_id = id;

        LOG_INF("Delegation triggered: id=%s, type=%d\n", id.c_str(), (int)type);
        return true;
    }

    return false;
}

bool inject_delegation_result(
        struct omni_context* ctx,
        DelegationManager* mgr) {

    if (!ctx || !mgr) return false;

    DelegationResult result;
    if (!mgr->pop_result(result)) {
        return false;
    }

    LOG_INF("Injecting delegation result: id=%s, success=%d, text_len=%zu\n",
            result.request_id.c_str(), result.success, result.text.size());

    // Remove from pending
    {
        std::lock_guard<std::mutex> lock(ctx->delegation_mtx);
        ctx->delegation_pending.erase(result.request_id);
        if (ctx->active_delegation_id == result.request_id) {
            ctx->delegation_in_progress = false;
            ctx->active_delegation_id.clear();
        }
    }

    // Inject result text into the voice stream
    // This would be done by pushing to text_queue and/or TTS queue
    // For now, just log - integration with stream_decode is done separately

    return true;
}
