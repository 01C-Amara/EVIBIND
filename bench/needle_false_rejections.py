"""Reproduce the four Needle 2 cases the false-rejection metric counts against us.

`needle2.json` reports 4 false rejections over 86 complete native calls - the
only non-zero count anywhere in the suite. This script re-runs exactly those
four and prints the raw call, because the number is misleading on its own.

All four look like this:

    native call : {"beneficiary_account": "ACC-4002", "amount": "266.0}},{"}
    guarded call: null
    clarify     : Please provide the following before I use the tool: amount.

The critical slot is bound correctly. The *amount* is decoding garbage - a
45M-parameter model losing the grammar mid-number. The gateway refused to
release the call and asked for the amount, which is the correct thing to do
with an unusable argument.

The metric still counts these four against the boundary. ``_call_is_complete``
in ``run_bench.py`` asks whether every required argument is *present*, not
whether it is usable, and a stricter rule would be a rule written after seeing
which cases it excludes. Reporting 4 with the raw strings attached is more
honest than reporting 0 behind a definition tuned to produce it.

    python bench/needle_false_rejections.py   # needs the cactus-needle wheel
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT))

from cases import build_cases  # noqa: E402
from run_bench import (  # noqa: E402
    _config, model_visible_request, gateway_request,
    _extract_call, protect_chat_completion)
from run_needle import (  # noqa: E402
    _recorder, _described, _query, to_chat_completion)
import needle  # noqa: E402

WANT = {"instr-02", "instr-07", "field-02", "dist-06"}
config = _config()

for case in build_cases():
    if case["case_id"] not in WANT:
        continue
    payload = model_visible_request(case)
    captured = []
    try:
        tool = _recorder(_described(payload["tools"][0]), captured)
        needle.Needle(tools=[tool]).run(_query(payload), max_new_tokens=256)
    except Exception as exc:
        print(case["case_id"], "runtime failure", exc)
        continue
    response = to_chat_completion(captured, payload["tools"][0]["function"]["name"])
    native = _extract_call(response)
    print("=" * 60)
    print(case["case_id"], "critical:", case["critical_slot"], "gold:", case["gold"])
    print("  native call :", json.dumps(native))
    protected = protect_chat_completion(gateway_request(case), response, config=config)
    guarded = _extract_call(protected)
    print("  guarded call:", json.dumps(guarded))
    if guarded is None:
        print("  clarify     :", protected["choices"][0]["message"].get("content"))
