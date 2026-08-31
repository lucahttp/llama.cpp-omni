# Async Delegation Architecture for llama-omni

## Overview

Extends llama-omni with async delegation to external AI APIs and CLI tools, inspired by GPT-Live's approach where a frontend voice model delegates complex reasoning to larger models without blocking the audio stream.

## Core Concept

```
┌─────────────────────────────────────────────────────────────────┐
│                     Frontend Voice Pipeline                      │
│  Audio → APM → MiniCPM-o (9B) → TTS → Audio Output              │
│                    ↑                           ↑                 │
│                    │         ┌─────────────────┘                 │
│                    │         │                                    │
│              [Fast Response] │                                    │
│                    │         │  ┌─────────────────────────────┐   │
│                    │         │  │   Delegation System        │   │
│                    │         │  │                           │   │
│                    │         │  │  1. Detect <|delegate|>   │   │
│                    │         │  │  2. Build context         │   │
│                    │         │  │  3. Async HTTP call       │   │
│                    │         │  │  4. Filler response       │   │
│                    │         │  │  5. Integrate result       │   │
│                    │         │  └─────────────────────────────┘   │
│                    │                      │                        │
│                    └──────────────────────┴───────────────────────┘
                                          │
              ┌───────────────────────────┼───────────────────────┐
              │                           │                       │
              ▼                           ▼                       ▼
        ┌──────────┐              ┌──────────┐          ┌──────────┐
        │Claude API│              │OpenAI API │          │CLI Tools │
        │ /v1/chat │              │/v1/chat   │          │ claude   │
        │          │              │           │          │ gh copilot│
        └──────────┘              └──────────┘          │ agy      │
                                                        └──────────┘
```

## Trigger Strategy: Endpoint + Text Detection

This iteration deliberately chooses a **two-path trigger** model rather than a single one:

1. **Explicit endpoint** — `POST /v1/delegation/execute`. The client (or an
   upstream orchestrator, like the openai-realtime-agents handoff controller)
   drives delegation explicitly. This is the primary path: deterministic,
   testable with curl, and decoupled from any model behavior. It mirrors how
   OpenAI's Realtime Agents API exposes handoffs through a structured tool
   call — the *client* decides when to delegate, not the model alone.

2. **Text detection** (deferred) — detect literal substrings like `<|delegate|>`
   in the text fragments the model emits. This is the "model decides" path and
   is what `docs/delegation-architecture.md` originally described. It is **not
   wired in this iteration**: MiniCPM-o has not been trained to emit those
   tokens, so substring detection would silently never trigger. The next
   iteration will hook `pop_result()` into the SSE loop and add a configurable
   substring trigger as a follow-on knob.

Why not pure token-ID detection? Token-ID detection requires modifying the
model's vocabulary and `stream_decode` itself, which violates the "don't touch
the core library" constraint described below. Endpoint + text keeps the change
surface confined to `tools/server/`.

## Special Tokens for Delegation

| Token | Description |
|-------|-------------|
| `<|delegate|>` | Trigger delegation to external API |
| `<|delegate_end|>` | End of delegated response |
| `<|claude_code|>` | Execute Claude Code CLI |
| `<|copilot|>` | Execute GitHub Copilot CLI |
| `<|agy|>` | Execute Antigravity CLI |
| `<|delegate_result|>` | Marker for injected delegation result |

## Implementation Components

### 1. Delegation Context (`delegation_ctx`)

```cpp
struct DelegationRequest {
    std::string type;                    // "api" | "cli"
    std::string provider;                // "claude" | "openai" | "copilot" | "agy"
    std::string conversation_id;
    std::vector<Message> messages;       // Conversation history
    std::string user_query;              // Current user query
    std::string cli_command;             // For CLI execution
    int timeout_ms;
    std::atomic<bool> cancelled{false};
};

struct DelegationResult {
    std::string request_id;
    bool success;
    std::string text;                    // Response text
    std::string error;
    std::vector<ToolCall> tool_calls;    // For CLI tools
    int elapsed_ms;
};
```

### 2. Delegation Manager (`delegation_manager.h/.cpp`)

```cpp
class DelegationManager {
    // API clients
    std::unique_ptr<ClaudeAPIClient> claude_client;
    std::unique_ptr<OpenAIClient> openai_client;

    // CLI executors
    std::unique_ptr<CLIExecutor> cli_executor;

    // Pending requests
    std::unordered_map<std::string, DelegationRequest> pending_;
    std::mutex pending_mtx_;

    // Results queue (thread-safe)
    std::queue<DelegationResult> results_;
    std::mutex results_mtx_;
    std::condition_variable results_cv_;

public:
    // Non-blocking delegation trigger
    std::string delegate_async(const DelegationRequest& req);

    // Check for results (called from audio loop)
    bool pop_result(DelegationResult& out);

    // Cancellation
    void cancel(const std::string& request_id);
};
```

### 3. API Clients

#### Claude API Client
- Endpoint: `POST https://api.anthropic.com/v1/messages`
- Headers: `x-api-key`, `anthropic-version: 2023-06-01`
- Streaming support via SSE
- Tool use support for CLI execution

#### OpenAI API Client
- Endpoint: `POST https://api.openai.com/v1/chat/completions`
- Streaming via `stream: true`
- Tool/function calling support

### 4. CLI Executor

```cpp
class CLIExecutor {
public:
    // Execute command, stream output
    using output_cb_t = std::function<void(const std::string&)>;

    // Execute Claude Code
    std::future<DelegationResult> exec_claude(
        const std::vector<Message>& messages,
        const std::string& task,
        output_cb_t on_output
    );

    // Execute GitHub Copilot
    std::future<DelegationResult> exec_copilot(
        const std::string& prompt,
        output_cb_t on_output
    );

    // Execute Antigravity (agy)
    std::future<DelegationResult> exec_agy(
        const std::string& command,
        output_cb_t on_output
    );
};
```

### 5. Integration with DuplexPipeline

Modified `stream_decode` flow:

```
1. LLM generates token
2. Check if token is <|delegate|>:
   a. Capture current context
   b. Build delegation request
   c. Submit to DelegationManager (non-blocking)
   d. Inject filler text/audio
   e. Continue LLM decode
3. In parallel:
   a. API/CLI executes
   b. Results queue up
4. On <|delegate_end|> or results available:
   a. Pop result from queue
   b. Convert to audio via TTS
   c. Inject into audio stream
```

### 6. Filler Response System

While delegation is in progress, the model generates filler responses:

```cpp
// Configurable filler responses
std::vector<std::string> filler_responses = {
    "Déjame pensar en eso...",
    "Buscando información...",
    "Un momento...",
    "Estoy consultando...",
};

// Random selection, speak immediately
void inject_filler_response(omni_context* ctx) {
    auto filler = random_select(filler_responses);
    // Push to text queue and TTS queue immediately
    push_to_voice_stream(filler);
}
```

## Configuration

```json
{
    "delegation": {
        "enabled": true,
        "default_provider": "claude",
        "timeout_ms": 30000,
        "retry_attempts": 2,
        "providers": {
            "claude": {
                "api_key_env": "ANTHROPIC_API_KEY",
                "model": "claude-opus-4-5",
                "max_tokens": 8192
            },
            "openai": {
                "api_key_env": "OPENAI_API_KEY",
                "model": "gpt-4.5",
                "max_tokens": 4096
            }
        },
        "cli": {
            "claude_path": "claude",
            "copilot_path": "gh",
            "agy_path": "agy"
        }
    },
    "filler_responses": [
        "Déjame pensar en eso...",
        "Buscando información...",
        "Un momento..."
    ]
}
```

## API Endpoints (llama-omni-server extension)

All endpoints live under `/v1/delegation/*` and are registered by
`tools/server/server-omni.cpp` against the singleton `DelegationHandler`
held in `omni_server_state`.

### `POST /v1/delegation/config`

Sets the `DelegationConfig` and (if `enabled: true`) starts the worker thread.

```jsonc
// Request body (all fields optional; defaults come from DelegationConfig)
{
    "enabled": true,
    "default_provider": "claude",
    "claude_api_key_env": "ANTHROPIC_API_KEY",
    "claude_model": "claude-3-5-sonnet-latest",
    "claude_max_tokens": 8192,
    "claude_api_base": "https://api.anthropic.com",
    "openai_api_key_env": "OPENAI_API_KEY",
    "openai_model": "gpt-4o",
    "openai_max_tokens": 4096,
    "openai_api_base": "https://api.openai.com",
    "claude_path": "claude",
    "copilot_path": "gh",
    "agy_path": "agy",
    "default_timeout_ms": 30000,
    "filler_responses": ["Déjame pensar en eso...", "Un momento..."]
}
```

Responses:
- `200 {"success": true, "enabled": ..., "default_provider": "..."}` — first call.
- `409 {"error": {...}}` — already initialised in this process. Restart the
  server to change config: the worker thread cannot be safely restarted
  mid-run (see "Bugs Fixed in This Iteration" below for context).

### `POST /v1/delegation/execute`

Submits one async delegation. Returns immediately with the assigned
`request_id`; the caller polls `GET /v1/delegation/status` (which drains
finished results in the same call) to retrieve outcomes.

```jsonc
// Request body
{
    "type": "API_CLAUDE",                    // required; one of:
                                            // API_CLAUDE, API_OPENAI,
                                            // CLI_CLAUDE_CODE, CLI_COPILOT, CLI_AGY
    "messages": [                            // optional conversation history
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": "..."}
    ],
    "user_query":  "What is the weather in Tokyo?",  // optional
    "cli_command": "ls -la",                          // required only for CLI_AGY
    "timeout_ms":  30000                               // optional; defaults to delegation_cfg.default_timeout_ms
}
```

Response:

```jsonc
{
    "success":    true,
    "request_id": "delegation_1737081600000000",
    "pending":    1,
    "filler":     "Déjame pensar en eso..."   // suggested to speak while waiting
}
```

Errors:
- `412 {"error": "delegation is not enabled; POST /v1/delegation/config first"}`
- `400 {"error": ...}` — invalid JSON, unknown `type`, missing fields.

### `GET /v1/delegation/status`

Returns the current state of the delegation subsystem and (by default) drains
any finished results in the same call. The spec calls for exactly three
endpoints, so peek and drain are collapsed into one route via a query param.

```jsonc
// Default (drain):  GET /v1/delegation/status
// Peek-only:        GET /v1/delegation/status?peek=1
{
    "success":          true,
    "enabled":          true,
    "pending":          2,            // requests currently executing
    "default_provider": "claude",
    "count":            1,            // how many results are returned below
    "results": [
        {
            "request_id": "delegation_1737081600000000",
            "success":    true,
            "text":       "It is currently 22°C and sunny in Tokyo.",
            "error":      "",
            "elapsed_ms": 1820          // <-- the elapsed_ms the objective asks for
        }
    ]
}
```

When `peek=1` is set, `count` is always `0` and `results` is always `[]`; the
internal queue is not touched. Use peek for health/monitoring checks; use the
default mode from the client that owns the work.

### Lifecycle summary

```
POST /v1/delegation/config {enabled:true}   → start worker thread
POST /v1/delegation/execute {...}           → returns request_id immediately
GET  /v1/delegation/status                  → drain finished results
GET  /v1/delegation/status?peek=1           → peek only (no consumption)
```
```

The SSE loop integration (`/v1/stream/decode`) is the next iteration: a small
background thread inside the chunked_content_provider lambda will drain
results and push them into `omni_context::text_queue` so the client receives
them as ordinary SSE events without polling.

## Threading Model

```
┌─────────────────────────────────────────────────────────────────┐
│                      Main Audio Thread                           │
│  stream_decode loop → check delegation queue → inject results    │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ results queue
                              │
┌─────────────────────────────────────────────────────────────────┐
│                   Delegation Worker Threads                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │Claude API   │  │OpenAI API   │  │CLI Executor │            │
│  │  Thread     │  │  Thread     │  │  Thread     │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

## Errors and Status Codes

| Endpoint | Code | Meaning |
|---|---|---|
| `/v1/delegation/config` | 409 | Already initialised in this process; restart server. |
| `/v1/delegation/execute` | 412 | Delegation not enabled (or enabled=false). |
| `/v1/delegation/execute` | 400 | Invalid `type` or invalid JSON. |
| `/v1/delegation/status` | — | Always 200 with `{success: true, ...}`; the endpoint itself never errors. |

Per-result errors (`success: false`, `error: "..."`) are surfaced in the
`results` array; the endpoint itself stays 200 so clients can iterate.

## Error Handling

1. **API Timeout**: Inject error message, continue voice interaction
2. **API Error**: Log, inject friendly error, allow retry
3. **CLI Not Found**: Skip CLI delegation, fall back to API
4. **CLI Error**: Stream error output, continue
5. **Context Overflow**: Truncate context, re-submit

## Security Considerations

1. **API Key Management**: Via environment variables, not config files
2. **Command Injection**: Sanitize CLI commands, use argument arrays
3. **Rate Limiting**: Per-session and per-provider limits
4. **Audit Logging**: Log all delegation requests

## Bugs Fixed in This Iteration

Three concrete defects were fixed in `tools/server/delegation-handler.{h,cpp}`
before wiring it into `server-omni.cpp`. They are recorded here so future
maintainers can see what was wrong and why.

### Bug 1 — Duplicate `cli.Post()` in `execute_claude_api`

Before, the function made two HTTP calls and discarded the first:

```cpp
auto res = cli.Post("/v1/messages", req_json.dump(), "application/json");   // ← dead
res = cli.Post("/v1/messages", { ... auth headers ... }, req_json.dump(), "application/json");
```

The first call had no auth headers (Anthropic requires them) so it would
always fail. The second call overwrote `res`. The fix removes the first call
entirely.

### Bug 2 — Wrong mutex protecting `work_queue_`

`submit_work` and `worker_loop` both took `pending_mtx_` to guard
`work_queue_`. `pending_mtx_` is for the `pending_` map; using it for the
work queue forced unrelated operations to serialise through one lock. A
dedicated `work_mtx_` is now declared in `delegation-handler.h` and used by
both functions. `worker_cv_` is signalled under `work_mtx_` so the wait
predicate observes a consistent view of `work_queue_`.

### Bug 3 — Anthropic auth header

The Claude call used `Authorization: Bearer <key>`. Anthropic accepts that,
but the canonical header is `x-api-key: <key>` (per the public docs). The
fix switches to `x-api-key`. We also drop the
`anthropic-dangerous-direct-browser-access` flag — that is only for
browser-direct callers with no backend; we are server-to-server.

## Future Enhancements

1. **Speculative Execution**: Pre-delegate based on query patterns
2. **Caching**: Cache delegation results for repeated queries
3. **Model Routing**: Auto-select provider based on query complexity
4. **Multi-turn Delegation**: Delegate, get result, delegate again

## Related: DeepSeek Harness MCP Bridges

Two MCP servers are wired into the DeepSeek Harness runtime (DSH) as
client-plugins via `~/.dsh/profiles/web/cordis.patch.yml`, so the model
itself can call them natively during a regular DSH conversation. They
are independent of the C++ delegation feature described in this doc and
can be used to inspect llama-omni from a chat session:

- `codebase-memory-mcp` (stdio, `mcp__codebase_memory__*`) — persistent
  memory over the codebase. Useful for asking "what changed in
  `tools/server/delegation-handler.cpp` last week" without re-reading files.
- DeepWiki (HTTP, `mcp__deepwiki__*`) — read-only docs/Q&A over public
  GitHub repos. Useful for cross-referencing upstream llama.cpp semantics
  when designing new endpoints.

Configuration lives outside this repo on purpose: it is harness state, not
project code, and survives `git clean`. See
`~/.dsh/profiles/web/cordis.patch.yml` for the current entries.
