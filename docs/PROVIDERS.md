# Provider Compatibility

EviBind requires one non-streaming OpenAI Chat Completions endpoint:

```text
POST <base_url>/v1/chat/completions
```

The upstream must accept a forced OpenAI function tool whose JSON Schema uses a
nested `oneOf`, `const`, closed objects, and candidate-ID enums, then return one
assistant `message.tool_calls` choice with JSON object arguments encoded as a
string. EviBind sends only the internal `evibind_action` tool—not the executable
application tools—sets `n=1` and `parallel_tool_calls=false`, prepends the
candidate catalog as a system instruction, and leaves unrelated provider request
fields untouched. Providers that ignore forced tool choice or cannot serialize
this schema are not compatible with the v2 full-guarantee path.

The branch union is nested under a single required `action` property rather than
sitting at the top level of `parameters`, because OpenAI rejects
`oneOf`/`anyOf`/`allOf`/`enum`/`const`/`not` there. Read the branches with
`tapbench.one_call_gateway.action_branches()` rather than indexing `oneOf`
directly; it accepts both the nested and the historical flat form.

## Compatibility Status

| Upstream | Transport | Validation status |
|---|---|---|
| llama.cpp | `/v1/chat/completions` | Live local conformance and benchmark queue |
| OpenAI | Chat Completions | Live credentials, 2026-08-18: 150/150 requests served end to end |
| vLLM | OpenAI-compatible server | Protocol fixture; live server not installed in this environment |
| SGLang | OpenAI-compatible server | Protocol fixture; live server not installed in this environment |

Protocol compatibility is not an accuracy result. Model weights, chat templates,
reasoning settings, and tool parsers can materially change which handles are
proposed before EviBind certifies them.

### OpenAI: how the action schema was fixed

The row above was recorded as a protocol fixture for a long time because live
credentials were never exercised. When they were, OpenAI rejected the forced
action tool before the model was ever consulted:

```text
HTTP 400  invalid_function_parameters
Invalid schema for function 'evibind_action': schema must have type 'object'
and not have 'oneOf'/'anyOf'/'allOf'/'enum'/'const'/'not' at the top level.
```

`_indexed_action_schema()` returned `{"type": "object", "oneOf": [...]}`, so
every request through `evibind serve` against `https://api.openai.com/v1`
failed this way. The union now sits under one required `action` property, which
keeps each branch's constraints — one mode, one required field set — while
satisfying the rule. Both proposal parsers accept either shape, so certificates
recorded against the flat form still replay.
`tests/test_openai_schema_compat.py` keeps it from regressing.

Verified after the change: 150/150 InjectBench cases served end to end against
`gpt-5.4-nano`, and `examples/live_gateway_demo.py` binds the authorised
account against a live upstream.

One thing made this hard to diagnose and is still worth fixing: the gateway
returns `{"message": "upstream returned HTTP 400"}` with the upstream's
explanation discarded, and logs nothing about failed upstream calls. The error
above was only recoverable by running a logging proxy in between. Surfacing
upstream error bodies under `allow_diagnostics` would turn that into one line
of output.

Note that reaching the upstream is not the same as completing a call. The
serving path currently withholds far more than the offline binding path does,
for a separate reason — see
[`FINDINGS.md`](FINDINGS.md) §10.

## Envelope Adapters

The dependency-free `evibind.adapters` module maps the single canonical action
tool and tool-call response envelopes for OpenAI Chat Completions and Responses,
Anthropic Messages, and Google Gemini Interactions and legacy `generateContent`.
The mappings follow the providers' documented function fields:

- <https://platform.openai.com/docs/api-reference/responses>
- <https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools>
- <https://ai.google.dev/gemini-api/docs/function-calling>

These are unit-tested envelope mappings, not native HTTP transports or live
credentialed compatibility claims. The gateway still requires the Chat
Completions endpoint described above.

## llama.cpp

Start a tool-capable model with its compatible Jinja chat template and point
EviBind to the server:

```bash
llama-server -m model.gguf --jinja --port 8080
export EVIBIND_UPSTREAM_BASE_URL=http://127.0.0.1:8080/v1
evibind serve
```

The llama.cpp documentation describes OpenAI-style function calling through
`llama-server` and notes that tool support depends on the selected template:
<https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md>.

## vLLM

Enable the model-appropriate tool parser and automatic tool choice when starting
vLLM. Keep `parallel_tool_calls=false`; vLLM documents that this constrains a
response to zero or one tool call:
<https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/>.

```bash
export EVIBIND_UPSTREAM_BASE_URL=http://127.0.0.1:8000/v1
evibind serve
```

## SGLang

Start SGLang with the parser that matches the model. Its tool-parser guide warns
that an incompatible parser or chat template can produce inconsistent tool
calls:
<https://docs.sglang.io/docs/advanced_features/tool_parser>.

```bash
export EVIBIND_UPSTREAM_BASE_URL=http://127.0.0.1:30000/v1
evibind serve
```

## Hosted APIs

Keep provider credentials only in the EviBind process environment:

```bash
export EVIBIND_UPSTREAM_BASE_URL=https://api.openai.com/v1
export EVIBIND_UPSTREAM_API_KEY=...
export EVIBIND_GATEWAY_API_KEY=...
evibind serve
```

Before using another OpenAI-compatible provider, replay
`examples/openai_request.json`, verify that the provider accepts the stripped
schema, and confirm that a tool-call response uses the expected
`choices[].message.tool_calls[]` shape. Do not infer accuracy support from a
successful transport check.
