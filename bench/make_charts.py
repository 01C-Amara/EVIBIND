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


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    for name, svg in (("bench_control.svg", control_chart()),
                      ("bench_live.svg", live_chart()),
                      ("paper_binding.svg", paper_chart())):
        (ASSETS / name).write_text(svg, encoding="utf-8")
        print("wrote", ASSETS / name)


if __name__ == "__main__":
    main()
