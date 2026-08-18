from __future__ import annotations

import gc
import json
import math
import time
import urllib.error
import urllib.request
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_experiment_config
from .families import get_family
from .io import read_jsonl, write_jsonl, write_yaml
from .models import backend_defaults, model_by_key, model_group_keys
from .retrieval import rank_tools
from .thinking import prediction_has_thinking_marker, thinking_metadata


def _grid_by_id(subgrids_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(grid["id"]): grid for grid in subgrids_cfg.get("subgrids", [])}


def _csv_filter(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def _selected_methods(grid: dict[str, Any], methods_filter: set[str] | None) -> list[str]:
    methods = [str(method) for method in grid.get("methods", [])]
    if methods_filter is not None:
        methods = [method for method in methods if method in methods_filter]
    if not methods:
        raise ValueError(f"no methods selected for grid {grid.get('id')}")
    return methods


def _selected_model_keys(models_cfg: dict[str, Any], grid: dict[str, Any], model_filter: set[str] | None) -> list[str]:
    keys = model_group_keys(models_cfg, str(grid["model_group"]))
    if model_filter is not None:
        keys = [key for key in keys if key in model_filter or model_by_key(models_cfg, key).get("id") in model_filter]
        for entry in models_cfg.get("evaluated_models", []):
            key = str(entry.get("key"))
            model_id = str(entry.get("id"))
            if (key in model_filter or model_id in model_filter) and key not in keys:
                keys.append(key)
    if not keys:
        raise ValueError(f"no models selected for grid {grid.get('id')}")
    return keys


def _method_allowed_for_model(method: str, model_entry: dict[str, Any], *, backend: str) -> bool:
    if backend not in {"hf-native", "hf-xgrammar"}:
        return True
    is_specialist = bool(model_entry.get("specialist_native_format"))
    if method == "native_specialist":
        return backend == "hf-native" and is_specialist
    if is_specialist:
        return False
    return backend == "hf-xgrammar"


def _identity_for(model_entry: dict[str, Any], *, backend: str) -> dict[str, Any]:
    defaults = backend_defaults(model_entry)
    if backend == "oracle":
        return {
            "model_id": str(model_entry["id"]),
            "backend": "oracle_dry_run",
            "quantization": "none",
            "chat_template": str(defaults.get("chat_template", "none")),
            "grammar_engine": "oracle_action_ir",
            "model_artifact": str(defaults.get("model_artifact", model_entry["id"])),
        }
    if backend == "llama-server":
        return {
            "model_id": str(model_entry["id"]),
            "backend": "llama.cpp",
            "quantization": str(defaults.get("quantization", "unknown")),
            "chat_template": str(defaults.get("chat_template", "unknown")),
            "grammar_engine": str(defaults.get("grammar_engine", "gbnf")),
            "model_artifact": str(defaults.get("model_artifact", model_entry["id"])),
        }
    if backend == "hf-xgrammar":
        return {
            "model_id": str(model_entry["id"]),
            "backend": "hf/xgrammar",
            "quantization": "bf16",
            "chat_template": str(defaults.get("chat_template", "unknown")),
            "grammar_engine": "xgrammar_json_schema",
            "model_artifact": str(model_entry["id"]),
        }
    if backend == "hf-native":
        return {
            "model_id": str(model_entry["id"]),
            "backend": "hf/native",
            "quantization": "bf16",
            "chat_template": str(defaults.get("chat_template", "unknown")),
            "grammar_engine": str(defaults.get("grammar_engine", "native_parser")),
            "model_artifact": str(model_entry["id"]),
        }
    raise ValueError(f"unknown runner backend: {backend}")


def _apply_identity_overrides(
    identity: dict[str, Any],
    *,
    model_artifact: str | None = None,
    quantization: str | None = None,
    chat_template: str | None = None,
    grammar_engine: str | None = None,
) -> dict[str, Any]:
    out = dict(identity)
    if model_artifact:
        out["model_artifact"] = model_artifact
    if quantization:
        out["quantization"] = quantization
    if chat_template:
        out["chat_template"] = chat_template
    if grammar_engine:
        out["grammar_engine"] = grammar_engine
    return out


def _call_only_action(case: dict[str, Any]) -> dict[str, Any]:
    family = get_family(str(case["family"]))
    tool = case["tools"][0]["canonical_name"]
    if case["task_kind"] == "missing_info":
        arguments = {family.missing_slot: case.get("derivable_values", {}).get(family.missing_slot, "guessed-value")}
    else:
        arguments = {}
    return {"mode": "call", "tool": tool, "arguments": arguments, "payload": {}}


def oracle_action(case: dict[str, Any], method: str) -> dict[str, Any]:
    if "call_only" in method and case.get("gold_action", {}).get("mode") != "call":
        return _call_only_action(case)
    return deepcopy(case["gold_action"])


RETRIEVER_TOP_K = 8


def _retrieval_arm_for_method(method: str) -> str | None:
    if method == "tap_tfidf_char":
        return "tfidf_char"
    if method == "tap_embedding":
        return "cheap_embedding"
    return None


def tools_for_method(case: dict[str, Any], method: str) -> list[dict[str, Any]]:
    arm = _retrieval_arm_for_method(method)
    if arm is None:
        return list(case.get("tools", []))
    return rank_tools(case, arm=arm)[:RETRIEVER_TOP_K]


def render_prompt(case: dict[str, Any], method: str) -> str:
    system = "\n".join(message["content"] for message in case.get("messages", []) if message.get("role") == "system")
    user = "\n".join(message["content"] for message in case.get("messages", []) if message.get("role") == "user")
    tools = json.dumps(tools_for_method(case, method), sort_keys=True)
    return (
        f"{system}\n\n"
        f"Method: {method}\n"
        "Available tools are JSON Schema tool descriptors. Return exactly one JSON object with keys "
        "mode, tool, arguments, and payload.\n\n"
        f"TOOLS:\n{tools}\n\n"
        f"USER:\n{user}\n\n"
        "ACTION IR:"
    )


def method_instruction(method: str) -> str:
    if "call_only" in method:
        return (
            "This is the call-only baseline. You must return mode='call' and choose the best available tool. "
            "If a required argument is missing, do not ask a clarification; leave the argument absent rather than inventing it."
        )
    if "abstain" in method or "full_tap" in method:
        return (
            "This method is abstention-aware. Use mode='clarify' with payload.missing_slots as a JSON list only when "
            "the user asks for an available tool action but a required argument is absent. Use mode='no_tool' when the "
            "request is general knowledge, explanation, comparison, summarization, or otherwise does not ask to operate "
            "one of the available tools; do not ask clarification for those requests. Use mode='direct_answer' when the "
            "request explicitly asks to answer directly without tools. Use mode='call' only when all required arguments "
            "are present in the request."
        )
    if method == "json_mode":
        return (
            "Return valid JSON Action IR. Use call only when the request fully specifies an available tool action; "
            "use clarify for missing required tool arguments, no_tool for unrelated/general requests, and direct_answer "
            "when the user asks to answer directly."
        )
    if method == "prompt_strict_json":
        return (
            "Return strict Action IR JSON. Use call only for fully specified tool actions, clarify for missing required "
            "tool arguments, no_tool for unrelated/general requests, and direct_answer for explicit direct-answer requests."
        )
    if method == "prompt_verbose":
        return (
            "Carefully inspect the request, compare the available tools, and return the single Action IR object that best matches. "
            "Do not add explanation outside JSON."
        )
    if method == "prompt_minimal":
        return "Return the best Action IR JSON object."
    if method == "prompt_few_shot":
        return (
            "Example clarify output: {'mode':'clarify','tool':null,'arguments':{},'payload':{'missing_slots':['date']}}. "
            "Example call output: {'mode':'call','tool':'tool_name','arguments':{'arg':'value'},'payload':{}}."
        )
    if method == "prompt_self_check":
        return "Before finalizing, internally check mode, tool name, required arguments, and whether any value was invented."
    if method == "best_of_n_budget_matched":
        return "Return the best Action IR JSON object. This baseline is matched by the run manifest wall-clock accounting."
    return "Return the best Action IR JSON object for this method."


def render_chat_messages(case: dict[str, Any], method: str, *, thinking_mode: str = "off") -> list[dict[str, str]]:
    request = "\n".join(message["content"] for message in case.get("messages", []) if message.get("role") == "user")
    compact_tools = []
    for tool in tools_for_method(case, method):
        compact_tools.append(
            {
                "name": tool.get("name"),
                "description": tool.get("description"),
                "parameters": tool.get("parameters"),
            }
        )
    thinking_instruction = (
        "Do not output private reasoning or analysis. Return only the final JSON object."
        if thinking_mode == "off"
        else "If you reason internally, keep it brief and do not include reasoning text in the final JSON object."
    )
    return [
        {
            "role": "system",
            "content": (
                "Return only one JSON object in Action IR. Valid modes are call, clarify, no_tool, "
                "and direct_answer. For call mode, use keys mode, tool, arguments, payload. "
                "Use clarify only for missing required tool arguments, no_tool for requests that do not need an available "
                "tool, and direct_answer when the user explicitly asks to answer directly. Do not add prose or markdown. "
                "Do not invent missing values. " + thinking_instruction
            ),
        },
        {
            "role": "user",
            "content": (
                f"Method: {method}\n"
                f"Method policy: {method_instruction(method)}\n"
                f"Request: {request}\n"
                f"Available tools: {json.dumps(compact_tools, sort_keys=True)}\n"
                "Return the Action IR JSON now."
            ),
        },
    ]


def render_xlam_messages(case: dict[str, Any]) -> list[dict[str, str]]:
    request = "\n".join(message["content"] for message in case.get("messages", []) if message.get("role") == "user")
    return [{"role": "user", "content": request}]


def render_xlam_tools(case: dict[str, Any], method: str) -> list[dict[str, Any]]:
    tools = []
    for tool in tools_for_method(case, method):
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description"),
                    "parameters": tool.get("parameters"),
                },
            }
        )
    return tools


def _parse_llama_text(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return {"mode": "no_tool", "tool": None, "arguments": {}, "payload": {"raw_text": ""}}
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {"mode": "direct_answer", "tool": None, "arguments": {}, "payload": {"raw_text": stripped}}


def parse_xlam_native_text(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return {"mode": "no_tool", "tool": None, "arguments": {}, "payload": {"raw_text": ""}}
    candidate = stripped
    if not candidate.startswith("["):
        start = candidate.find("[")
        end = candidate.rfind("]")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
    try:
        calls = json.loads(candidate)
    except json.JSONDecodeError:
        return {"mode": "direct_answer", "tool": None, "arguments": {}, "payload": {"raw_text": stripped}}
    if not isinstance(calls, list) or not calls:
        return {"mode": "no_tool", "tool": None, "arguments": {}, "payload": {"raw_text": stripped}}
    first = calls[0]
    if not isinstance(first, dict):
        return {"mode": "direct_answer", "tool": None, "arguments": {}, "payload": {"raw_text": stripped}}
    arguments = first.get("arguments", {})
    return {
        "mode": "call",
        "tool": first.get("name"),
        "arguments": arguments if isinstance(arguments, dict) else {},
        "payload": {"native_tool_call_count": len(calls)},
    }


def _llama_response_metadata(data: dict[str, Any]) -> dict[str, Any]:
    usage = data.get("usage", {}) if isinstance(data.get("usage"), dict) else {}
    timings = data.get("timings", {}) if isinstance(data.get("timings"), dict) else {}
    choices = data.get("choices", [])
    choice = choices[0] if choices else {}
    return {
        "response_id": data.get("id"),
        "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "prompt_tokens_per_second": timings.get("prompt_per_second"),
        "generated_tokens_per_second": timings.get("predicted_per_second"),
        "prompt_ms": timings.get("prompt_ms"),
        "generation_ms": timings.get("predicted_ms"),
    }


def _request_llama_json(
    request: urllib.request.Request,
    endpoint: str,
    *,
    max_attempts: int = 3,
) -> tuple[dict[str, Any], int]:
    retryable_statuses = {500, 502, 503, 504}
    for attempt in range(max_attempts):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.loads(response.read().decode("utf-8")), attempt
        except urllib.error.HTTPError as exc:
            if exc.code not in retryable_statuses or attempt + 1 >= max_attempts:
                raise RuntimeError(f"llama-server request failed for {endpoint}: {exc}") from exc
        except urllib.error.URLError as exc:
            if attempt + 1 >= max_attempts:
                raise RuntimeError(f"llama-server request failed for {endpoint}: {exc}") from exc
        time.sleep(0.25 * (2**attempt))
    raise RuntimeError(f"llama-server request failed for {endpoint}: retry budget exhausted")


def llama_server_action(
    case: dict[str, Any],
    method: str,
    endpoint: str,
    *,
    max_tokens: int,
    temperature: float,
    thinking_mode: str,
    seed: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    payload = {
        "messages": render_chat_messages(case, method, thinking_mode=thinking_mode),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    if seed is not None:
        payload["seed"] = seed
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    data, retry_count = _request_llama_json(request, endpoint)
    choices = data.get("choices", [])
    message = choices[0].get("message", {}) if choices else {}
    content = message.get("content", data.get("content", data.get("response", "")))
    metadata = _llama_response_metadata(data)
    metadata["retry_count"] = retry_count
    metadata["raw_text"] = str(content)
    return _parse_llama_text(str(content)), metadata


def action_ir_json_schema(case: dict[str, Any]) -> dict[str, Any]:
    tool_names = sorted({str(tool.get("name")) for tool in case.get("tools", []) if tool.get("name") is not None})
    return {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": ["call", "clarify", "no_tool", "direct_answer"]},
            "tool": {"anyOf": [{"type": "string", "enum": tool_names}, {"type": "null"}]},
            "arguments": {"type": "object", "additionalProperties": True},
            "payload": {"type": "object", "additionalProperties": True},
        },
        "required": ["mode", "tool", "arguments", "payload"],
        "additionalProperties": False,
    }


class _HFBackendState:
    def __init__(self) -> None:
        self.model_key: str | None = None
        self.model_id: str | None = None
        self.tokenizer: Any = None
        self.model: Any = None
        self.config: Any = None
        self.tokenizer_info: Any = None
        self.grammar_compiler: Any = None
        self.device: Any = None

    def close(self) -> None:
        self.model_key = None
        self.model_id = None
        self.tokenizer = None
        self.model = None
        self.config = None
        self.tokenizer_info = None
        self.grammar_compiler = None
        self.device = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def ensure_loaded(self, model_key: str, model_entry: dict[str, Any]) -> None:
        if self.model_key == model_key:
            return
        self.close()
        import torch
        import xgrammar as xgr
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        model_id = str(model_entry["id"])
        model_path = snapshot_download(model_id, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        self.config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if not torch.cuda.is_available():
            self.model.to("cpu")
        self.model.eval()
        self.device = getattr(self.model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        vocab_size = int(getattr(self.config, "vocab_size", self.tokenizer.vocab_size))
        self.tokenizer_info = xgr.TokenizerInfo.from_huggingface(self.tokenizer, vocab_size=vocab_size)
        self.grammar_compiler = xgr.GrammarCompiler(self.tokenizer_info)
        self.model_key = model_key
        self.model_id = model_id


def _hf_inputs_for_messages(
    state: _HFBackendState,
    messages: list[dict[str, str]],
    *,
    tools: list[dict[str, Any]] | None = None,
    thinking_mode: str = "off",
) -> Any:
    kwargs: dict[str, Any] = {
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    if tools is not None:
        kwargs["tools"] = tools
    if thinking_mode in {"off", "budget_128"}:
        kwargs["enable_thinking"] = thinking_mode != "off"
    try:
        inputs = state.tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        inputs = state.tokenizer.apply_chat_template(messages, **kwargs)
    except Exception:
        prompt = "\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\nassistant:"
        inputs = state.tokenizer(prompt, return_tensors="pt")
    return {key: value.to(state.device) for key, value in inputs.items()}


def _decode_new_tokens(state: _HFBackendState, outputs: Any, prompt_tokens: int) -> str:
    generated = outputs[0][prompt_tokens:]
    return state.tokenizer.decode(generated, skip_special_tokens=True)


def _safe_eos_ids(tokenizer: Any) -> list[int]:
    eos = tokenizer.eos_token_id
    if eos is None:
        return []
    if isinstance(eos, list):
        return [int(item) for item in eos]
    return [int(eos)]


def _xgrammar_prompt_diagnostics(state: _HFBackendState, compiled_grammar: Any, inputs: Any) -> dict[str, Any]:
    import torch
    import xgrammar as xgr

    with torch.no_grad():
        logits = state.model(**inputs).logits[:, -1, :].float()
    vocab_size = int(logits.shape[-1])
    matcher = xgr.GrammarMatcher(compiled_grammar)
    bitmask = xgr.allocate_token_bitmask(1, state.tokenizer_info.vocab_size)
    matcher.fill_next_token_bitmask(bitmask, 0)
    masked = logits.clone()
    xgr.apply_token_bitmask_inplace(masked, bitmask.to(masked.device), vocab_size=vocab_size)
    legal_mask = torch.isfinite(masked[0])
    probs = torch.softmax(logits[0], dim=-1)
    legal_probs = probs[legal_mask]
    legal_mass = float(legal_probs.sum().item())
    if legal_probs.numel() and legal_mass > 0:
        norm = legal_probs / legal_mass
        entropy = float(-(norm * torch.log(norm.clamp_min(1e-30))).sum().item())
    else:
        entropy = 0.0
    eos_masked = True
    for eos_id in _safe_eos_ids(state.tokenizer):
        if 0 <= eos_id < vocab_size and bool(legal_mask[eos_id].item()):
            eos_masked = False
    legal_count = int(legal_mask.sum().item())
    return {
        "legal_token_count": legal_count,
        "legal_token_mass": legal_mass,
        "rerouted_mass": float(max(0.0, 1.0 - legal_mass)),
        "mask_entropy": entropy,
        "mask_entropy_norm": float(entropy / math.log(max(2, legal_count))),
        "eos_masked_non_accepting": bool(eos_masked),
        "xgrammar_completed_at_prompt": bool(matcher.is_completed()),
    }


def hf_xgrammar_action(
    state: _HFBackendState,
    case: dict[str, Any],
    method: str,
    *,
    max_tokens: int,
    temperature: float,
    seed: int,
    thinking_mode: str = "off",
) -> tuple[Any, dict[str, Any]]:
    import torch
    import xgrammar as xgr

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    inputs = _hf_inputs_for_messages(state, render_chat_messages(case, method, thinking_mode=thinking_mode), thinking_mode=thinking_mode)
    compiled = state.grammar_compiler.compile_json_schema(action_ir_json_schema(case), strict_mode=True)
    diagnostics = _xgrammar_prompt_diagnostics(state, compiled, inputs)
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    processor = xgr.contrib.hf.LogitsProcessor(compiled)
    started = time.perf_counter()
    with torch.no_grad():
        outputs = state.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            logits_processor=[processor],
            pad_token_id=state.tokenizer.pad_token_id or state.tokenizer.eos_token_id,
            eos_token_id=state.tokenizer.eos_token_id,
        )
    generation_seconds = time.perf_counter() - started
    text = _decode_new_tokens(state, outputs, prompt_tokens)
    completion_tokens = int(outputs.shape[-1] - prompt_tokens)
    metadata = {
        "finish_reason": "stop",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": int(outputs.shape[-1]),
        "generated_tokens_per_second": completion_tokens / generation_seconds if generation_seconds > 0 else None,
        "generation_ms": generation_seconds * 1000.0,
        "raw_text": text,
        "diagnostics": diagnostics,
    }
    return _parse_llama_text(text), metadata


def hf_native_action(
    state: _HFBackendState,
    case: dict[str, Any],
    method: str,
    *,
    max_tokens: int,
    temperature: float,
    seed: int,
    thinking_mode: str = "off",
) -> tuple[Any, dict[str, Any]]:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    inputs = _hf_inputs_for_messages(state, render_xlam_messages(case), tools=render_xlam_tools(case, method), thinking_mode=thinking_mode)
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    started = time.perf_counter()
    with torch.no_grad():
        outputs = state.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=state.tokenizer.pad_token_id or state.tokenizer.eos_token_id,
            eos_token_id=state.tokenizer.eos_token_id,
        )
    generation_seconds = time.perf_counter() - started
    text = _decode_new_tokens(state, outputs, prompt_tokens)
    completion_tokens = int(outputs.shape[-1] - prompt_tokens)
    metadata = {
        "finish_reason": "stop",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": int(outputs.shape[-1]),
        "generated_tokens_per_second": completion_tokens / generation_seconds if generation_seconds > 0 else None,
        "generation_ms": generation_seconds * 1000.0,
        "raw_text": text,
        "native_parser": "xlam_json_array",
    }
    return parse_xlam_native_text(text), metadata


def _generation_jobs(
    cases: list[dict[str, Any]],
    grids: dict[str, dict[str, Any]],
    models_cfg: dict[str, Any],
    *,
    backend: str,
    method_filter: set[str] | None,
    model_filter: set[str] | None,
    seed_values: list[int],
    max_generations: int | None,
) -> list[tuple[dict[str, Any], str, str, int]]:
    jobs: list[tuple[dict[str, Any], str, str, int]] = []
    for case in cases:
        grid = grids[str(case["hypothesis_grid_id"])]
        for method in _selected_methods(grid, method_filter):
            for model_key in _selected_model_keys(models_cfg, grid, model_filter):
                model_entry = model_by_key(models_cfg, model_key)
                if not _method_allowed_for_model(method, model_entry, backend=backend):
                    continue
                for seed in seed_values:
                    jobs.append((case, method, model_key, seed))
                    if max_generations is not None and len(jobs) >= max_generations:
                        if backend.startswith("hf-"):
                            jobs.sort(key=lambda job: (job[2], job[1], job[0]["case_id"], job[3]))
                        return jobs
    if backend.startswith("hf-"):
        jobs.sort(key=lambda job: (job[2], job[1], job[0]["case_id"], job[3]))
    return jobs


def run_cases(
    *,
    cases_path: str | Path,
    output_path: str | Path,
    timings_path: str | Path,
    manifest_path: str | Path,
    backend: str = "oracle",
    endpoint: str = "http://127.0.0.1:8080",
    methods: str | None = None,
    models: str | None = None,
    seeds: str = "1",
    max_generations: int | None = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
    model_artifact: str | None = None,
    quantization: str | None = None,
    chat_template: str | None = None,
    grammar_engine: str | None = None,
    diagnostics_path: str | Path | None = None,
    thinking_mode: str | None = "off",
    reasoning_budget: int | None = None,
) -> dict[str, Any]:
    cfg = load_experiment_config()
    grids = _grid_by_id(cfg.subgrids)
    method_filter = _csv_filter(methods)
    model_filter = _csv_filter(models)
    seed_values = [int(seed.strip()) for seed in seeds.split(",") if seed.strip()]
    cases = read_jsonl(cases_path)
    thinking = thinking_metadata("not_applicable" if backend == "oracle" and thinking_mode is None else thinking_mode, reasoning_budget)
    started_at = datetime.now(timezone.utc).isoformat()
    predictions: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    hf_state = _HFBackendState() if backend.startswith("hf-") else None
    jobs = _generation_jobs(
        cases,
        grids,
        cfg.models,
        backend=backend,
        method_filter=method_filter,
        model_filter=model_filter,
        seed_values=seed_values,
        max_generations=max_generations,
    )

    try:
        for case, method, model_key, seed in jobs:
            model_entry = model_by_key(cfg.models, model_key)
            identity = _apply_identity_overrides(
                _identity_for(model_entry, backend=backend),
                model_artifact=model_artifact,
                quantization=quantization,
                chat_template=chat_template,
                grammar_engine=grammar_engine,
            )
            start = time.perf_counter()
            error: str | None = None
            if backend == "oracle":
                action = oracle_action(case, method)
                response_metadata: dict[str, Any] = {}
            else:
                try:
                    if backend == "llama-server":
                        action, response_metadata = llama_server_action(case, method, endpoint, max_tokens=max_tokens, temperature=temperature, thinking_mode=str(thinking["thinking_mode"]))
                    elif backend == "hf-xgrammar":
                        assert hf_state is not None
                        hf_state.ensure_loaded(model_key, model_entry)
                        action, response_metadata = hf_xgrammar_action(hf_state, case, method, max_tokens=max_tokens, temperature=temperature, seed=seed, thinking_mode=str(thinking["thinking_mode"]))
                    elif backend == "hf-native":
                        assert hf_state is not None
                        hf_state.ensure_loaded(model_key, model_entry)
                        action, response_metadata = hf_native_action(hf_state, case, method, max_tokens=max_tokens, temperature=temperature, seed=seed, thinking_mode=str(thinking["thinking_mode"]))
                    else:
                        raise ValueError(f"unknown runner backend: {backend}")
                except Exception as exc:  # keep long experiment batches alive when one request fails
                    action = {"runner_error": str(exc)}
                    error = str(exc)
                    response_metadata = {
                        "finish_reason": "runner_error",
                        "error_type": exc.__class__.__name__,
                        "error_message": str(exc),
                    }
            elapsed = time.perf_counter() - start
            prediction_row = {
                "case_id": case["case_id"],
                "method": method,
                "model_id": identity["model_id"],
                "seed": seed,
                "prediction": action,
                "response_metadata": response_metadata,
                "runner_error": error,
                **identity,
                **thinking,
            }
            prediction_row["thinking_marker_detected"] = prediction_has_thinking_marker(prediction_row)
            predictions.append(
                {
                    **prediction_row,
                }
            )
            timings.append(
                {
                    "case_id": case["case_id"],
                    "hypothesis_grid_id": case["hypothesis_grid_id"],
                    "method": method,
                    "model_key": model_key,
                    "model_id": identity["model_id"],
                    "backend": identity["backend"],
                    "elapsed_seconds": elapsed,
                    "runner_error": error,
                    **thinking,
                    "thinking_marker_detected": prediction_row["thinking_marker_detected"],
                    **response_metadata,
                }
            )
            diagnostics = response_metadata.get("diagnostics") if isinstance(response_metadata, dict) else None
            if isinstance(diagnostics, dict):
                factors = case.get("factors", {})
                diagnostic_rows.append(
                    {
                        "case_id": case["case_id"],
                        "hypothesis_grid_id": case["hypothesis_grid_id"],
                        "method": method,
                        "model_key": model_key,
                        "model_id": identity["model_id"],
                        "seed": seed,
                        **identity,
                        **thinking,
                        "thinking_marker_detected": prediction_row["thinking_marker_detected"],
                        **{key: factors[key] for key in ("N", "q", "d", "e", "sigma", "alpha") if key in factors},
                        **diagnostics,
                    }
                )
    finally:
        if hf_state is not None:
            hf_state.close()

    write_jsonl(output_path, predictions)
    write_jsonl(timings_path, timings)
    if diagnostics_path is not None:
        write_jsonl(diagnostics_path, diagnostic_rows)
    manifest = {
        "schema_version": "tapbench.run_manifest.v1",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "endpoint": endpoint if backend == "llama-server" else None,
        "dry_run": backend == "oracle",
        "cases_path": str(cases_path),
        "predictions_path": str(output_path),
        "timings_path": str(timings_path),
        "diagnostics_path": str(diagnostics_path) if diagnostics_path is not None else None,
        "generation_count": len(predictions),
        "diagnostic_count": len(diagnostic_rows),
        "seed_values": seed_values,
        "methods_filter": sorted(method_filter) if method_filter else None,
        "models_filter": sorted(model_filter) if model_filter else None,
        "identity_overrides": {
            "model_artifact": model_artifact,
            "quantization": quantization,
            "chat_template": chat_template,
            "grammar_engine": grammar_engine,
        },
        "thinking_mode": thinking["thinking_mode"],
        "reasoning_budget": thinking["reasoning_budget"],
        "note": "oracle backend validates pipeline shape only; do not treat as model evidence" if backend == "oracle" else None,
    }
    write_yaml(manifest_path, manifest)
    return manifest
