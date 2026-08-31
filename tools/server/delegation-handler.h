#pragma once

#include "protocol.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <queue>
#include <future>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>

// ==================== Delegation Types ====================

enum class DelegationType {
    NONE,
    API_CLAUDE,
    API_OPENAI,
    CLI_CLAUDE_CODE,
    CLI_COPILOT,
    CLI_AGY
};

struct DelegationMessage {
    std::string role;
    std::string content;
};

struct DelegationRequest {
    std::string id;
    DelegationType type;
    std::vector<DelegationMessage> messages;
    std::string user_query;
    std::string cli_command;
    int timeout_ms;
    std::chrono::steady_clock::time_point start_time;
    // Note: cancellation is not implemented in this iteration. If needed, a
    // separate std::unordered_map<std::string, std::atomic<bool>> can track
    // cancelled state without making DelegationRequest non-copyable.
};

struct DelegationResult {
    std::string request_id;
    bool success;
    std::string text;
    std::string error;
    int elapsed_ms;
};

struct DelegationConfig {
    bool enabled = false;
    std::string default_provider = "claude";

    std::string claude_api_key_env = "ANTHROPIC_API_KEY";
    std::string claude_model = "claude-opus-4-5";
    int claude_max_tokens = 8192;
    std::string claude_api_base = "https://api.anthropic.com";

    std::string openai_api_key_env = "OPENAI_API_KEY";
    std::string openai_model = "gpt-4.5";
    int openai_max_tokens = 4096;
    std::string openai_api_base = "https://api.openai.com";

    std::string claude_path = "claude";
    std::string copilot_path = "gh";
    std::string agy_path = "agy";

    std::vector<std::string> filler_responses = {
        "Déjame pensar en eso...",
        "Buscando información...",
        "Un momento...",
        "Estoy consultando...",
        "Déjame ver..."
    };

    int default_timeout_ms = 30000;
};

// ==================== Delegation Handler ====================

class DelegationHandler {
public:
    DelegationHandler();
    ~DelegationHandler();

    void init(const DelegationConfig& config);
    bool is_enabled() const { return config_.enabled; }

    // Trigger delegation (non-blocking)
    std::string delegate_async(const DelegationRequest& req);

    // Check for results
    bool has_result() const;
    bool pop_result(DelegationResult& out);
    size_t pending_count() const;

    void cancel(const std::string& request_id);  // not implemented; cancellation needs a separate map
    void cancel_all();                          // not implemented; see above

    std::string get_filler_response() const;

    // Convert token type to delegation type
    static DelegationType token_to_delegation_type(
        int token_id,
        int delegate_tok,
        int claude_code_tok,
        int copilot_tok,
        int agy_tok
    );

private:
    void worker_loop();
    void submit_work(const std::string& id, const DelegationRequest& req);

    DelegationResult execute_claude_api(const DelegationRequest& req);
    DelegationResult execute_openai_api(const DelegationRequest& req);
    DelegationResult execute_cli_claude(const DelegationRequest& req);
    DelegationResult execute_cli_copilot(const DelegationRequest& req);
    DelegationResult execute_cli_agy(const DelegationRequest& req);

    DelegationConfig config_;
    std::unordered_map<std::string, DelegationRequest> pending_;
    std::queue<DelegationResult> results_;
    mutable std::mutex pending_mtx_;   // mutable so const methods can lock it
    mutable std::mutex results_mtx_;  // mutable so const methods can lock it
    // work_mtx_ guards work_queue_ only. pending_mtx_ guards pending_; the two
    // are independent so callers holding one do not block the other. worker_cv_
    // is signalled under work_mtx_ (the worker holds work_mtx_ while popping).
    mutable std::mutex work_mtx_;     // mutable so const methods can lock it

    std::thread worker_thread_;
    std::atomic<bool> running_{false};
    std::condition_variable worker_cv_;
    std::queue<std::pair<std::string, DelegationRequest>> work_queue_;
};
