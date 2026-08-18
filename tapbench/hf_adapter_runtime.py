from __future__ import annotations

import gc
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

from .runner import _parse_llama_text


HF_ADAPTER_RUNTIME_VERSION = "tapbench.hf_adapter_xgrammar_runtime.v1"


class HFAdapterJSONRuntime:
    def __init__(
        self,
        base_model: str,
        *,
        adapter_path: str | Path | None = None,
        context_tokens: int = 8192,
    ) -> None:
        import torch
        import xgrammar as xgr
        from peft import PeftModel
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for HF adapter evaluation")
        self.base_model = base_model
        self.adapter_path = str(adapter_path) if adapter_path is not None else None
        self.context_tokens = int(context_tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            local_files_only=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        config = AutoConfig.from_pretrained(base_model, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="sdpa",
        )
        if adapter_path is not None:
            model = PeftModel.from_pretrained(
                model,
                str(adapter_path),
                is_trainable=False,
            )
        model.eval()
        self.model = model
        self.device = next(model.parameters()).device
        vocab_size = int(getattr(config, "vocab_size", self.tokenizer.vocab_size))
        self.tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            self.tokenizer,
            vocab_size=vocab_size,
        )
        self.compiler = xgr.GrammarCompiler(self.tokenizer_info)
        self._compiled: dict[str, Any] = {}

    def close(self) -> None:
        import torch

        self._compiled.clear()
        self.compiler = None
        self.tokenizer_info = None
        self.model = None
        self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()

    def _inputs(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
            "enable_thinking": False,
        }
        try:
            inputs = self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking")
            inputs = self.tokenizer.apply_chat_template(messages, **kwargs)
        return {key: value.to(self.device) for key, value in inputs.items()}

    def __call__(
        self,
        endpoint: str,
        messages: list[dict[str, str]],
        *,
        response_schema: dict[str, Any],
        max_tokens: int,
        temperature: float,
        seed: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del endpoint
        import torch
        import xgrammar as xgr

        schema_text = json.dumps(
            response_schema,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        schema_sha256 = hashlib.sha256(schema_text.encode()).hexdigest()
        compiled = self._compiled.get(schema_sha256)
        if compiled is None:
            compiled = self.compiler.compile_json_schema(
                response_schema,
                strict_mode=True,
            )
            self._compiled[schema_sha256] = compiled
        inputs = self._inputs(messages)
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        output_budget = min(int(max_tokens), self.context_tokens - prompt_tokens)
        if output_budget <= 0:
            raise RuntimeError(
                f"context_overflow: prompt={prompt_tokens}, context={self.context_tokens}"
            )
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        processor = xgr.contrib.hf.LogitsProcessor(compiled)
        generation_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": output_budget,
            "do_sample": float(temperature) > 0,
            "logits_processor": [processor],
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if float(temperature) > 0:
            generation_kwargs["temperature"] = float(temperature)
        started = time.perf_counter()
        with torch.no_grad():
            output = self.model.generate(**generation_kwargs)
        generation_seconds = time.perf_counter() - started
        sequence = output.sequences[0]
        generated = sequence[prompt_tokens:]
        completion_tokens = int(generated.shape[-1])
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        log_probabilities: list[float] = []
        for token, scores in zip(generated, output.scores, strict=False):
            log_probability = torch.log_softmax(scores[0].float(), dim=-1)[
                int(token)
            ]
            log_probabilities.append(float(log_probability.item()))
        mean_log_probability = (
            sum(log_probabilities) / len(log_probabilities)
            if log_probabilities
            else -math.inf
        )
        finish_reason = (
            "length" if completion_tokens >= output_budget else "stop"
        )
        return _parse_llama_text(text), {
            "runtime_version": HF_ADAPTER_RUNTIME_VERSION,
            "response_schema_sha256": schema_sha256,
            "raw_text": text,
            "finish_reason": finish_reason,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "generation_ms": generation_seconds * 1000.0,
            "generated_tokens_per_second": (
                completion_tokens / generation_seconds
                if generation_seconds > 0
                else None
            ),
            "mean_constrained_token_log_probability": mean_log_probability,
            "minimum_constrained_token_log_probability": (
                min(log_probabilities) if log_probabilities else -math.inf
            ),
            "generation_calls": 1,
            "rendered_input_tokens": prompt_tokens,
            "context_headroom_tokens": (
                self.context_tokens - prompt_tokens - max_tokens
            ),
            "context_overflow": False,
            "context_truncated": False,
        }
