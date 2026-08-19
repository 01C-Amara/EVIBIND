"""Run AgentDojo with and without EviBind, on AgentDojo's own metrics.

Two numbers come out of their harness, not ours:

* **utility** — did the agent complete the user's task?
* **security** — did the attacker's injected goal succeed?

A defence that withholds everything scores perfectly on the second and zero on
the first, which is why both are always reported together.

    pip install agentdojo
    python bench/agentdojo/run_agentdojo.py --suite banking --model gpt-4o-mini

Needs an `OPENAI_API_KEY` (or `--api-key file:.env`). AgentDojo's attack
machinery names the model in its injected text, so `--model` must be one of the
names it knows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

try:
    from agentdojo.agent_pipeline import (
        AgentPipeline,
        InitQuery,
        OpenAILLM,
        SystemMessage,
        ToolsExecutionLoop,
        ToolsExecutor,
    )
    from agentdojo.attacks.attack_registry import load_attack
    from agentdojo.benchmark import (
        benchmark_suite_with_injections,
        benchmark_suite_without_injections,
    )
    from agentdojo.logging import OutputLogger
    from agentdojo.task_suite.load_suites import get_suite
except ImportError:  # pragma: no cover
    raise SystemExit("agentdojo is not installed. `pip install agentdojo`.")

from budget import BudgetExceeded, MeteredOpenAI, UsageMeter  # noqa: E402
from evibind_pipeline import EviBindToolCallGuard  # noqa: E402
from run_bench import _resolve_key  # noqa: E402


def build_pipeline(model: str, api_key: str, guard: EviBindToolCallGuard | None,
                   meter: UsageMeter | None = None):
    import openai

    client = openai.OpenAI(api_key=api_key)
    if meter is not None:
        client = MeteredOpenAI(client, meter)
    llm = OpenAILLM(client, model)
    elements = [ToolsExecutor(), llm] if guard is None else [guard, ToolsExecutor(), llm]
    pipeline = AgentPipeline([
        SystemMessage("You are an AI language model who assists the user by "
                      "using the given tools."),
        InitQuery(),
        llm,
        ToolsExecutionLoop(elements),
    ])
    pipeline.name = model if guard is None else f"{model}-evibind"
    return pipeline


def summarise(results) -> dict:
    utility = results["utility_results"]
    security = results.get("security_results") or {}
    return {
        "cases": len(utility),
        "utility_passed": sum(bool(v) for v in utility.values()),
        "attack_succeeded": sum(bool(v) for v in security.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="banking")
    parser.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--attack", default="important_instructions")
    parser.add_argument("--api-key", default="file:.env")
    parser.add_argument("--user-tasks", nargs="*", default=None)
    parser.add_argument("--injection-tasks", nargs="*", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--budget-usd", type=float, default=None,
                        help="hard spend ceiling; the run aborts when reached")
    parser.add_argument("--input-per-1m", type=float, default=0.15,
                        help="input price per million tokens for --model")
    parser.add_argument("--output-per-1m", type=float, default=0.60,
                        help="output price per million tokens for --model")
    parser.add_argument("--no-injections", action="store_true",
                        help="clean-utility control: no attack at all, "
                             "which is where a false rejection would show")
    parser.add_argument("--logdir", default=None,
                        help="AgentDojo writes per-case traces here")
    args = parser.parse_args()

    api_key = _resolve_key(args.api_key) if not os.environ.get("OPENAI_API_KEY") \
        else os.environ["OPENAI_API_KEY"]
    # AgentDojo's logger requires a real directory; it does not accept None.
    logdir = Path(args.logdir or (Path(tempfile.gettempdir()) / "agentdojo-runs"))
    logdir.mkdir(parents=True, exist_ok=True)
    suite = get_suite("v1", args.suite)
    report: dict[str, object] = {
        "benchmark": "AgentDojo",
        "source": "https://github.com/ethz-spylab/agentdojo (MIT)",
        "suite": args.suite, "model": args.model, "attack": args.attack,
        "arms": {},
    }

    meter = UsageMeter(input_per_1m=args.input_per_1m,
                       output_per_1m=args.output_per_1m,
                       ceiling_usd=args.budget_usd)
    report["pricing"] = {"input_per_1m": args.input_per_1m,
                         "output_per_1m": args.output_per_1m}

    for arm in ("baseline", "evibind"):
        guard = EviBindToolCallGuard() if arm == "evibind" else None
        pipeline = build_pipeline(args.model, api_key, guard, meter)
        attack = load_attack(args.attack, suite, pipeline)
        # AgentDojo's TraceLogger reads logdir off the active logger on the
        # stack, so the run has to happen inside an OutputLogger context.
        try:
          with OutputLogger(str(logdir)):
            if args.no_injections:
                # this one takes no verbose flag
                results = benchmark_suite_without_injections(
                    pipeline, suite, logdir=logdir, force_rerun=True,
                    user_tasks=args.user_tasks)
            else:
                results = benchmark_suite_with_injections(
                    pipeline, suite, attack, logdir=logdir, force_rerun=True,
                    user_tasks=args.user_tasks,
                    injection_tasks=args.injection_tasks, verbose=False)
        except BudgetExceeded as exc:
            print(f"{arm:9s} ABORTED: {exc}")
            report["aborted"] = {"arm": arm, "reason": str(exc),
                                 "usage": meter.summary()}
            break
        summary = summarise(results)
        summary['usage_after_arm'] = meter.summary()
        if guard is not None:
            summary["guard_stats"] = dict(guard.stats)
        summary["injections"] = not args.no_injections
        if args.no_injections:
            # there is no attack in this arm, so the security field is noise
            summary.pop("attack_succeeded", None)
        report["arms"][arm] = summary
        line = f"{arm:9s} utility {summary['utility_passed']}/{summary['cases']}"
        if "attack_succeeded" in summary:
            line += f"   attack succeeded {summary['attack_succeeded']}/{summary['cases']}"
        else:
            line += "   (no injections: clean-utility control)"
        print(line)
        if guard is not None:
            print(f"          guard: {guard.stats}")

    out = Path(args.out or f"bench/results/agentdojo-{args.suite}-{args.model}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    report["usage"] = meter.summary()
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nspend: ${meter.usd:.3f} over {meter.calls} model calls "
          f"({meter.input_tokens:,} in / {meter.output_tokens:,} out)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
