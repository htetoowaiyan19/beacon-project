# BEACON API reference

API version 3.0.0 is model-only; the product release is BEACON v1.0.0. Start the verified release with `python scripts/run_release.py` or `run_trained_chat.bat`. Interactive OpenAPI documentation: `/docs`; schema: `/openapi.json`.

## Endpoints

| Method | Path | Result |
| --- | --- | --- |
| GET | `/api/config` | Mode and streaming protocol |
| GET | `/api/health` | Server, device and model status |
| POST | `/api/chat/stream` | Stream a model reply |
| GET | `/` | Browser chat interface |

Retrieval parameters and document routes have been removed. Unavailable document routes may return 404 or 405 because of the static frontend mount. The API currently has no authentication or rate limiting and allows all CORS origins. The launcher defaults to `0.0.0.0`; the command above uses localhost.

## GET /api/config

```json
{"mode":"model-only","stream_protocol":"ui-message-stream-v1"}
```

## GET /api/health

Both config and health responses include `release` with `name`, `version`, `edition`, `released_on`, `model`, and `limitations`. The minimal config example above omits this additional metadata.

Health returns `status: "online"`, `mode: "model-only"`, and `gpu`. GPU information includes `device`, `vram_allocated_gb`, `vram_reserved_gb`, `is_loaded`, `has_lora`, `adapter_available`, and `adapter_name`, including on CPU. Adapter availability checks for its weights and configuration; the release launcher separately verifies the weights' checksum. The endpoint creates the lazy model service but does not load weights or guarantee a chat will succeed. VRAM measures allocations in this server process. Frontend mock responses are explicitly labeled by their device field.

## POST /api/chat/stream

Send `Content-Type: application/json`.

| Field | Default | Constraints / behavior |
| --- | --- | --- |
| `message` | Required | String, 1-32,000 characters |
| `history` | `[]` | Ordered `{role, content}` messages; role is user or assistant |
| `use_base_model` | `false` | Disable the trained adapter for this request |
| `think` | `false` | Enable Qwen thinking mode |
| `temperature` | `0.7` | 0-2 |
| `top_p` | `0.8` | 0-1 |
| `max_new_tokens` | `512` | Integer, 16-2,048 |
| `system_prompt` | `null` | Optional persona override; otherwise the shared companion prompt |

The server is stateless for chat history; include prior exchanges on every request. The runtime persona follows the user's language and adapts tone, with additional instructions for concise answers, uncertainty and avoiding invented personal experiences. First chat lazily loads the model and adapter. Generation is serialized within the process. If the configured adapter is missing, a trained-model request emits a stream error instead of silently answering as the base model. Explicit `use_base_model: true` requests remain available. Set `BEACON_LORA_PATH` before startup with `scripts/run_server.py` to select another evaluated adapter; the release launcher always selects the frozen v1 snapshot.

```powershell
$body = @{ message = 'Explain TCP and UDP'; history = @(); max_new_tokens = 256 } | ConvertTo-Json
Invoke-WebRequest -Uri http://localhost:8000/api/chat/stream -Method Post -ContentType application/json -Body $body
```

This PowerShell example collects the response. For incremental display use a streaming HTTP client or the browser parser in `frontend/stream.js`.

## Streaming contract

Response: HTTP 200, `Content-Type: text/event-stream`, `x-vercel-ai-ui-message-stream: v1`, `Cache-Control: no-cache`, `X-Accel-Buffering: no`. This is the [Vercel AI SDK UI Message Stream protocol](https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol).

Each SSE frame contains `data: <JSON>` and a blank line. A successful stream is:

```text
data: {"type":"start","messageId":"m"}

data: {"type":"start-step"}

data: {"type":"text-start","id":"t"}

data: {"type":"text-delta","id":"t","delta":"Hello"}

data: {"type":"text-end","id":"t"}

data: {"type":"data-metrics","data":{"total_tokens":1,"elapsed_seconds":0.2,"tokens_per_second":5.0,"prompt_tokens":50,"finish_reason":"stop"}}

data: {"type":"finish-step"}

data: {"type":"finish","finishReason":"stop"}

data: [DONE]
```

IDs and metrics above are illustrative. There can be many deltas; preserve order and join them. Metrics count actual generated token IDs, not text chunks. `finishReason` is `stop` or `length`; an exhausted output limit is not a transport error. Decode UTF-8 incrementally and buffer incomplete SSE frames. Do not record a reply as complete until both the finish event and `[DONE]` arrive.

Invalid JSON or field validation returns HTTP 422 before streaming. Generation errors after the stream starts retain HTTP 200 and emit a text-end event if necessary, an `error` event with `errorText`, then `finish` with `finishReason: "error"`, and `[DONE]`. Inspect the server log for details. A disconnected client cancels generation; its partial response should not become a completed history exchange.

Training controls are local desktop controls, not HTTP endpoints; see [TRAINING_UI.md](TRAINING_UI.md).
