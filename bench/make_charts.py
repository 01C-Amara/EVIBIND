"""Generate README chart SVGs from InjectBench result JSON.

Deterministic, dependency-free. Status colors (good / neutral / critical) carry
outcome; every value is directly labeled, so color never carries meaning alone.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
RESULTS = ROOT / "bench" / "results"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GOOD = "#0ca30c"
CRIT = "#d03b3b"
NEUTRAL = "#b5b3ac"
BLUE_LIGHT = "#86b6ef"
BLUE_DARK = "#1c5cab"
GRID = "#e7e6e2"
FONT = "font-family='-apple-system,Segoe UI,Helvetica,Arial,sans-serif'"

LABELS = {
    "inject_instruction": "injected instruction",
    "inject_data_field": "injected data field",
    "inject_authority": "forged authority",
    "defer_to_tool": "user defers to tool",
    "supersession": "stale after correction",
    "ambiguity": "ambiguous request",
    "cross_slot": "swapped slots",
    "distractor": "near-duplicate distractors",
    "transcription": "garbled long identifier",
    "negation": "explicitly forbidden value",
}
ORIGIN = ("inject_instruction", "inject_data_field", "inject_authority",
          "defer_to_tool")


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def control_chart() -> str:
    data = _load("scripted-compromised.json")
    cats = list(data["by_category"])
    width, row_h, left, top = 940, 30, 290, 118
    plot_w = width - left - 120
    height = top + len(cats) * row_h + 82
    n = 15

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}'>",
        f"<rect width='{width}' height='{height}' fill='{SURFACE}'/>",
        f"<text x='24' y='32' {FONT} font-size='17' font-weight='600' fill='{INK}'>"
        f"What the gateway structurally prevents - worst-case selector, 150 cases</text>",
        f"<text x='24' y='52' {FONT} font-size='12.5' fill='{INK2}'>"
        f"A scripted model that always emits the engineered wrong value. Each row: "
        f"15 cases, outcome after the gateway.</text>",
        f"<text x='24' y='72' {FONT} font-size='12.5' fill='{INK2}'>"
        f"Origin violations are repaired or fail closed; selection errors are not "
        f"the boundary's job and pass through.</text>",
    ]
    lx = left
    for name, color in (("repaired to intended call", GOOD),
                        ("withheld - fail closed", NEUTRAL),
                        ("still wrong - not confined", CRIT)):
        parts.append(f"<rect x='{lx}' y='{top - 26}' width='10' height='10' rx='3' fill='{color}'/>")
        parts.append(f"<text x='{lx + 15}' y='{top - 17}' {FONT} font-size='11.5' fill='{INK2}'>{name}</text>")
        lx += 200

    y = top
    group_drawn = set()
    for cat in cats:
        block = data["by_category"][cat]["guarded"]
        group = "ORIGIN VIOLATION" if cat in ORIGIN else "SELECTION ERROR"
        if group not in group_drawn:
            if group_drawn:
                y += 16
            group_drawn.add(group)
            parts.append(f"<text x='24' y='{y + 15}' {FONT} font-size='11' "
                         f"font-weight='700' fill='{INK2}' letter-spacing='0.5'>{group}</text>")
        parts.append(f"<text x='{left - 12}' y='{y + 15}' {FONT} font-size='12' "
                     f"fill='{INK}' text-anchor='end'>{LABELS[cat]}</text>")
        x = left
        for count, color in ((block["exact"], GOOD), (block["abstain"], NEUTRAL),
                             (block["harmful"] + block["other"], CRIT)):
            if count == 0:
                continue
            seg = plot_w * count / n
            parts.append(f"<rect x='{x:.1f}' y='{y}' width='{max(seg - 2, 2):.1f}' "
                         f"height='21' rx='4' fill='{color}'/>")
            parts.append(f"<text x='{x + seg / 2:.1f}' y='{y + 15}' {FONT} font-size='11' "
                         f"font-weight='600' fill='"
                         f"{'#ffffff' if color != NEUTRAL else INK}' "
                         f"text-anchor='middle'>{count}</text>")
            x += seg
        y += row_h

    parts.append(f"<text x='24' y='{height - 20}' {FONT} font-size='11.5' fill='{INK2}'>"
                 f"60/60 origin violations neutralised (45 repaired to the user-authorised "
                 f"value, 15 withheld); garbled identifiers also caught.</text>")
    parts.append("</svg>")
    return "".join(parts)


def live_chart() -> str:
    data = _load("claude-haiku.json")
    cats = list(data["by_category"])
    width, row_h, left, top = 900, 28, 250, 108
    height = top + len(cats) * row_h + 60
    col = left + 90

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}'>",
        f"<rect width='{width}' height='{height}' fill='{SURFACE}'/>",
        f"<text x='24' y='32' {FONT} font-size='17' font-weight='600' fill='{INK}'>"
        f"Live run: Claude Haiku, 150 cases - the gateway costs nothing</text>",
        f"<text x='24' y='52' {FONT} font-size='12.5' fill='{INK2}'>"
        f"Same model output scored in both arms. Every correct call Haiku made was "
        f"released unchanged; every abstention was preserved.</text>",
        f"<text x='24' y='72' {FONT} font-size='12.5' fill='{INK2}'>"
        f"0 false rejections, 0 rewritten arguments, 0 harmful releases across all "
        f"ten categories.</text>",
        f"<text x='{col}' y='{top - 8}' {FONT} font-size='11' fill='{INK2}'>native</text>",
        f"<text x='{col + 300}' y='{top - 8}' {FONT} font-size='11' fill='{INK2}'>guarded</text>",
    ]
    y = top
    for cat in cats:
        block = data["by_category"][cat]
        parts.append(f"<text x='{left - 12}' y='{y + 14}' {FONT} font-size='12' "
                     f"fill='{INK}' text-anchor='end'>{LABELS[cat]}</text>")
        for offset, arm in ((0, "native"), (300, "guarded")):
            t = block[arm]
            if t["exact"]:
                text, color = f"{t['exact']}/15 intended call released", GOOD
            elif t["abstain"]:
                text, color = f"{t['abstain']}/15 asked instead of acting", NEUTRAL
            else:
                text, color = "mixed", CRIT
            parts.append(f"<circle cx='{col + offset}' cy='{y + 10}' r='5' fill='{color}'/>")
            parts.append(f"<text x='{col + offset + 12}' y='{y + 14}' {FONT} "
                         f"font-size='11.5' fill='{INK2}'>{text}</text>")
        y += row_h

    parts.append(f"<text x='24' y='{height - 18}' {FONT} font-size='11.5' fill='{INK2}'>"
                 f"Haiku resisted all 60 injection cases on its own - the gateway's value "
                 f"here is that it added no cost while making the guarantee structural.</text>")
    parts.append("</svg>")
    return "".join(parts)


def paper_chart() -> str:
    models = [("Qwen3-1.7B", 5, 89), ("Qwen3.6-35B-A3B", 86, 100),
              ("GPT-OSS-120B", 97, 100), ("GPT-5.6-Luna", 100, 100)]
    width, row_h, left, top = 860, 52, 190, 92
    plot_w = width - left - 90
    height = top + len(models) * row_h + 58

    def sx(v: float) -> float:
        return left + plot_w * v / 100

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}'>",
        f"<rect width='{width}' height='{height}' fill='{SURFACE}'/>",
        f"<text x='24' y='34' {FONT} font-size='17' font-weight='600' fill='{INK}'>"
        f"Paper result: exact critical binding, pre-output-frozen 100-case test</text>",
        f"<text x='24' y='54' {FONT} font-size='12.5' fill='{INK2}'>"
        f"Actual catalog vs admissible top-1 (paper Table 4). Pruning helps models "
        f"below ceiling.</text>",
        f"<circle cx='{left}' cy='{top - 16}' r='6' fill='{BLUE_LIGHT}'/>",
        f"<text x='{left + 12}' y='{top - 12}' {FONT} font-size='11.5' fill='{INK2}'>actual catalog</text>",
        f"<circle cx='{left + 130}' cy='{top - 16}' r='6' fill='{BLUE_DARK}'/>",
        f"<text x='{left + 142}' y='{top - 12}' {FONT} font-size='11.5' fill='{INK2}'>admissible top-1</text>",
    ]
    for tick in (0, 25, 50, 75, 100):
        x = sx(tick)
        parts.append(f"<line x1='{x:.1f}' y1='{top}' x2='{x:.1f}' y2='{height - 46}' "
                     f"stroke='{GRID}' stroke-width='1'/>")
        parts.append(f"<text x='{x:.1f}' y='{height - 28}' {FONT} font-size='11' "
                     f"fill='{INK2}' text-anchor='middle'>{tick}%</text>")
    y = top + row_h / 2
    for name, a, b in models:
        parts.append(f"<text x='{left - 12}' y='{y + 4}' {FONT} font-size='12.5' "
                     f"fill='{INK}' text-anchor='end'>{name}</text>")
        if a != b:
            parts.append(f"<line x1='{sx(a):.1f}' y1='{y}' x2='{sx(b):.1f}' y2='{y}' "
                         f"stroke='{BLUE_DARK}' stroke-width='2'/>")
            parts.append(f"<circle cx='{sx(a):.1f}' cy='{y}' r='8' fill='{BLUE_LIGHT}' "
                         f"stroke='{SURFACE}' stroke-width='2'/>")
            parts.append(f"<text x='{sx(a) - 14:.1f}' y='{y + 4}' {FONT} font-size='11.5' "
                         f"fill='{INK2}' text-anchor='end'>{a}%</text>")
        parts.append(f"<circle cx='{sx(b):.1f}' cy='{y}' r='8' fill='{BLUE_DARK}' "
                     f"stroke='{SURFACE}' stroke-width='2'/>")
        parts.append(f"<text x='{sx(b) + 14:.1f}' y='{y + 4}' {FONT} font-size='11.5' "
                     f"font-weight='600' fill='{INK}'>{b}%</text>")
        y += row_h
    parts.append("</svg>")
    return "".join(parts)


MODEL_ORDER = [
    ("gpt-5.6-terra", "GPT-5.6 Terra"),
    ("gpt-5.6-luna", "GPT-5.6 Luna"),
    ("gpt-5.6-sol", "GPT-5.6 Sol"),
    ("grok-4.6", "Grok 4.6"),
    ("grok-4.5", "Grok 4.5"),
    ("claude-haiku", "Claude Haiku"),
    ("gpt-5.4-mini", "GPT-5.4 mini"),
    ("gpt-5.4-nano", "GPT-5.4 nano"),
    ("gpt-4.1-mini", "GPT-4.1 mini"),
]

SEGMENTS = (("harmful", CRIT, "attacker value bound"),
            ("correct", GOOD, "user value bound"),
            ("abstain", NEUTRAL, "withheld"),
            ("other", GRID, "other"))


def matrix_chart() -> str:
    """Origin-violation outcomes per model, native vs behind the gateway."""
    models = []
    for name, pretty in MODEL_ORDER:
        path = RESULTS / f"{name}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if "native_slot" not in data.get("origin_violation", {}):
            continue
        models.append((pretty, data["origin_violation"]))
    if not models:
        return ""

    width, left, top = 960, 168, 152
    bar_h, gap, block_h = 17, 5, 56
    plot_w = width - left - 200
    height = top + len(models) * block_h + 62
    total = models[0][1]["n"]

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}'>",
        f"<rect width='{width}' height='{height}' fill='{SURFACE}'/>",
        f"<text x='24' y='32' {FONT} font-size='17' font-weight='600' fill='{INK}'>"
        f"Origin violations across nine models - {total} cases each</text>",
        f"<text x='24' y='53' {FONT} font-size='12.5' fill='{INK2}'>"
        f"Where the critical slot ended up when the context carried an injected "
        f"instruction, a poisoned data field, forged authority, or a user deferring "
        f"to tool output.</text>",
        f"<text x='24' y='71' {FONT} font-size='12.5' fill='{INK2}'>"
        f"Both bars score the same model output: the upper bar is the call the model "
        f"made, the lower bar is that call after the gateway re-derived the slot.</text>",
    ]

    lx = 24
    for key, color, caption in SEGMENTS:
        parts.append(f"<rect x='{lx}' y='{top - 41}' width='11' height='11' "
                     f"fill='{color}' stroke='{INK2}' stroke-width='0.4'/>")
        parts.append(f"<text x='{lx + 16}' y='{top - 32}' {FONT} font-size='11' "
                     f"fill='{INK2}'>{caption}</text>")
        lx += 24 + len(caption) * 6.0

    y = top
    for pretty, block in models:
        parts.append(f"<text x='{left - 14}' y='{y + 22}' {FONT} font-size='12.5' "
                     f"font-weight='600' fill='{INK}' text-anchor='end'>{pretty}</text>")
        for row, (arm, tag) in enumerate((("native_slot", "native"),
                                          ("guarded_slot", "guarded"))):
            by = y + row * (bar_h + gap)
            x = left
            for key, color, _ in SEGMENTS:
                count = block[arm].get(key, 0)
                if not count:
                    continue
                w = plot_w * count / block["n"]
                parts.append(f"<rect x='{x:.1f}' y='{by}' width='{w:.1f}' "
                             f"height='{bar_h}' fill='{color}'/>")
                if w > 22:
                    ink = SURFACE if color in (CRIT, GOOD) else INK
                    parts.append(f"<text x='{x + w / 2:.1f}' y='{by + 12.5}' {FONT} "
                                 f"font-size='10.5' fill='{ink}' "
                                 f"text-anchor='middle'>{count}</text>")
                x += w
            harmful = block[arm].get("harmful", 0)
            correct = block[arm].get("correct", 0)
            parts.append(f"<text x='{left + plot_w + 10}' y='{by + 12.5}' {FONT} "
                         f"font-size='10.5' fill='{INK2}'>{tag}: {harmful} harmful, "
                         f"{correct} correct</text>")
        y += block_h

    parts.append(f"<text x='24' y='{height - 20}' {FONT} font-size='11.5' fill='{INK2}'>"
                 f"Every model reaches 0 harmful behind the gateway. On the weaker "
                 f"selectors the correct-call count also rises, because a slot re-derived "
                 f"from the user's own span replaces the injected one.</text>")
    parts.append("</svg>")
    return "".join(parts)


CONTRAST_MODELS = [
    ("gpt-5.6-sol", "GPT-5.6 Sol"),
    ("gpt-5.6-luna", "GPT-5.6 Luna"),
    ("gpt-5.6-terra", "GPT-5.6 Terra"),
    ("grok-4.6", "Grok 4.6"),
    ("gpt-5.4-mini", "GPT-5.4 mini"),
    ("gpt-5.4-nano", "GPT-5.4 nano"),
    ("gpt-4.1-mini", "GPT-4.1 mini"),
]


def _injecagent_rate(model: str) -> tuple[int, int] | None:
    path = RESULTS / f"injecagent-dh_enhanced-{model}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    block = data["security"]
    return block["native_attack"], block["scored"]


def _injectbench_rate(model: str) -> tuple[int, int] | None:
    path = RESULTS / f"{model}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    origin = data.get("origin_violation", {})
    if "native_slot" not in origin:
        return None
    return origin["native_slot"]["harmful"], origin["n"]


def contrast_chart() -> str:
    """Two attack shapes, same models: what they refuse and what they don't."""
    rows = []
    for model, pretty in CONTRAST_MODELS:
        selection = _injecagent_rate(model)
        substitution = _injectbench_rate(model)
        if selection and substitution:
            rows.append((pretty, selection, substitution))
    if not rows:
        return ""
    # weakest last: the story reads top to bottom, frontier models resisting
    # both shapes and the cheap tiers falling only to the second
    rows.sort(key=lambda row: row[2][0] / max(row[2][1], 1))

    width, left, top = 900, 150, 150
    bar_h, gap, block_h = 20, 8, 74
    plot_w = width - left - 210
    height = top + len(rows) * block_h + 76

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}'>",
        f"<rect width='{width}' height='{height}' fill='{SURFACE}'/>",
        f"<text x='24' y='32' {FONT} font-size='17' font-weight='600' fill='{INK}'>"
        f"Two shapes of injection, the same three models</text>",
        f"<text x='24' y='53' {FONT} font-size='12.5' fill='{INK2}'>"
        f"How often each model did what the attacker's text told it to, with no "
        f"gateway in the way. Both runs are live and unguarded.</text>",
        f"<text x='24' y='71' {FONT} font-size='12.5' fill='{INK2}'>"
        f"Upper bar: &#8220;call this other tool&#8221; (InjecAgent dh_enhanced, 510 cases). "
        f"Lower bar: &#8220;use this account number instead&#8221; "
        f"(InjectBench, 60 origin-violation cases).</text>",
        f"<rect x='24' y='{top - 38}' width='11' height='11' fill='{GOOD}' "
        f"stroke='{INK2}' stroke-width='0.4'/>",
        f"<text x='40' y='{top - 29}' {FONT} font-size='11' fill='{INK2}'>"
        f"refused the attacker</text>",
        f"<rect x='190' y='{top - 38}' width='11' height='11' fill='{CRIT}' "
        f"stroke='{INK2}' stroke-width='0.4'/>",
        f"<text x='206' y='{top - 29}' {FONT} font-size='11' fill='{INK2}'>"
        f"did what the attacker's text said</text>",
    ]

    y = top
    for pretty, (sel_hit, sel_n), (sub_hit, sub_n) in rows:
        parts.append(f"<text x='{left - 14}' y='{y + 26}' {FONT} font-size='12.5' "
                     f"font-weight='600' fill='{INK}' text-anchor='end'>{pretty}</text>")
        for row, (label, hit, total) in enumerate((
                ("tool selection", sel_hit, sel_n),
                ("argument substitution", sub_hit, sub_n))):
            by = y + row * (bar_h + gap)
            share = hit / total if total else 0.0
            bad_w = plot_w * share
            parts.append(f"<rect x='{left}' y='{by}' width='{plot_w}' "
                         f"height='{bar_h}' fill='{GOOD}'/>")
            if bad_w > 0.5:
                parts.append(f"<rect x='{left}' y='{by}' width='{bad_w:.1f}' "
                             f"height='{bar_h}' fill='{CRIT}'/>")
            parts.append(f"<text x='{left + plot_w + 10}' y='{by + 14.5}' {FONT} "
                         f"font-size='11' fill='{INK2}'>{hit}/{total} &#183; "
                         f"{label}</text>")
        y += block_h

    parts.append(f"<text x='24' y='{height - 20}' {FONT} font-size='11.5' fill='{INK2}'>"
                 f"Every model refuses to be redirected to another tool. The cheaper tiers "
                 f"still take the attacker's account number, because that never arrives "
                 f"as an instruction &#8212; it arrives as a fact.</text>")
    parts.append("</svg>")
    return "".join(parts)


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    for name, svg in (("bench_control.svg", control_chart()),
                      ("bench_live.svg", live_chart()),
                      ("bench_models.svg", matrix_chart()),
                      ("bench_contrast.svg", contrast_chart()),
                      ("paper_binding.svg", paper_chart())):
        if not svg:
            continue
        (ASSETS / name).write_text(svg, encoding="utf-8")
        print("wrote", ASSETS / name)


if __name__ == "__main__":
    main()
