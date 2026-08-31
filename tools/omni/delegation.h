#pragma once

#include "omni.h"

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
    API_CLAUDE,      // Delegate to Claude API
    API_OPENAI,      // Delegate to OpenAI API
    CLI_CLAUDE_CODE, // Execute Claude Code CLI
    CLI_COPILOT,     // Execute GitHub Copilot CLI
    CLI_AGY          // Execute Antigravity CLI
};

struct DelegationMessage {
    std::string role;     // "user" or "assistant"
    std::string content;
};

struct DelegationRequest {
    std::string id;
    DelegationType type;
    std::vector<DelegationMessage> messages;  // Conversation history
    std::string user_query;                   // Current user query
    std::string cli_command;                 // For CLI execution
    int timeout_ms;
    std::atomic<bool> cancelled{false};
    std::chrono::steady_clock::time_point start_time;
};

struct DelegationResult {
    std::string request_id;
    bool success;
    std::string text;                        // Response text
    std::string error;
    int elapsed_ms;
};

struct DelegationConfig {
    bool enabled = false;
    std::string default_provider = "claude";

    // Claude API config
    std::string claude_api_key_env = "ANTHROPIC_API_KEY";
    std::string claude_model = "claude-opus-4-5";
    int claude_max_tokens = 8192;
    std::string claude_api_base = "https://api.anthropic.com";

    // OpenAI API config
    std::string openai_api_key_env = "OPENAI_API_KEY";
    std::string openai_model = "gpt-4.5";
    int openai_max_tokens = 4096;
    std::string openai_api_base = "https://api.openai.com";

    // CLI paths
    std::string claude_path = "claude";
    std::string copilot_path = "gh";
    std::string agy_path = "agy";

    // Filler responses while waiting for delegation
    std::vector<std::string> filler_responses = {
        "Déjame pensar en eso...",
        "Buscando información...",
        "Un momento...",
        "Estoy consultando...",
        "Déjame ver..."
    };

    // Timeout
    int default_timeout_ms = 30000;
};

// ==================== API Client Interface ====================

class IAPIClient {
public:
    virtual ~IAPIClient() = default;
    virtual DelegationResult execute(const DelegationRequest& req) = 0;
    virtual std::string get_provider_name() const = 0;
};

// ==================== Claude API Client ====================

class ClaudeAPIClient : public IAPIClient {
public:
    ClaudeAPIClient(const DelegationConfig& config);

    DelegationResult execute(const DelegationRequest& req) override;
    std::string get_provider_name() const override { return "claude"; }

private:
    DelegationConfig config_;
    std::string api_key_;
    bool initialized_;
};

// ==================== OpenAI API Client ====================

class OpenAIClient : public IAPIClient {
public:
    OpenAIClient(const DelegationConfig& config);

    DelegationResult execute(const DelegationRequest& req) override;
    std::string get_provider_name() const override { return "openai"; }

private:
    DelegationConfig config_;
    std::string api_key_;
    bool initialized_;
};

// ==================== CLI Executor ====================

class CLIExecutor {
public:
    using output_cb_t = std::function<void(const std::string&, bool)>;  // text, is_error

    CLIExecutor(const DelegationConfig& config);

    // Execute Claude Code CLI
    std::future<DelegationResult> exec_claude(
        const std::vector<DelegationMessage>& messages,
        const std::string& task,
        output_cb_t on_output
    );

    // Execute GitHub Copilot CLI
    std::future<DelegationResult> exec_copilot(
        const std::string& prompt,
        output_cb_t on_output
    );

    // Execute Antigravity (agy) CLI
    std::future<DelegationResult> exec_agy(
        const std::string& command,
        output_cb_t on_output
    );

private:
    DelegationConfig config_;
    std::string exec_sync(const std::string& cmd, output_cb_t on_output);
};

// ==================== Delegation Manager ====================

class DelegationManager {
public:
    DelegationManager();
    ~DelegationManager();

    // Initialize with config
    void init(const DelegationConfig& config);

    // Check if delegation is enabled
    bool is_enabled() const { return config_.enabled; }

    // Trigger async delegation (non-blocking)
    // Returns request ID for tracking
    std::string delegate_async(const DelegationRequest& req);

    // Check if a delegation result is available
    bool has_result() const;

    // Pop next available result (thread-safe)
    bool pop_result(DelegationResult& out);

    // Get pending delegation count
    size_t pending_count() const;

    // Cancel a pending delegation
    void cancel(const std::string& request_id);

    // Cancel all pending delegations
    void cancel_all();

    // Get random filler response
    std::string get_filler_response() const;

    // Convert token type to delegation type
    static DelegationType token_to_delegation_type(
        llama_token token,
        llama_token delegate_tok,
        llama_token claude_code_tok,
        llama_token copilot_tok,
        llama_token agy_tok
    );

private:
    DelegationConfig config_;
    std::unordered_map<std::string, DelegationRequest> pending_;
    std::queue<DelegationResult> results_;
    std::mutex pending_mtx_;
    std::mutex results_mtx_;

    // API clients
    std::unique_ptr<ClaudeAPIClient> claude_client_;
    std::unique_ptr<OpenAIClient> openai_client_;

    // CLI executor
    std::unique_ptr<CLIExecutor> cli_executor_;

    // Worker thread for async execution
    std::thread worker_thread_;
    std::atomic<bool> running_{false};
    std::condition_variable worker_cv_;
    std::queue<std::pair<std::string, DelegationRequest>> work_queue_;

    void worker_loop();
    void submit_work(const std::string& id, const DelegationRequest& req);
};

// ==================== Delegation Detection ====================

// Detect if a token triggers delegation and return the type
DelegationType detect_delegation_trigger(
    struct omni_context * ctx,
    llama_token token
);

// Handle delegation trigger - starts async delegation
// Returns true if delegation was triggered
bool handle_delegation_trigger(
    struct omni_context * ctx,
    DelegationType type,
    const std::string& user_query,
    DelegationManager* mgr
);

// Check if delegation result is ready and inject it
// Returns true if result was injected
bool inject_delegation_result(
    struct omni_context * ctx,
    DelegationManager* mgr
);
