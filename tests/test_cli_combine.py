from __future__ import annotations

from tapbench.cli import main
from tapbench.io import read_jsonl, write_jsonl


def test_combine_jsonl_preserves_input_and_row_order(tmp_path) -> None:
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    output = tmp_path / "out.jsonl"
    write_jsonl(first, [{"id": 1}, {"id": 2}])
    write_jsonl(second, [{"id": 3}])
    assert main(["combine-jsonl", "--inputs", f"{first},{second}", "--output", str(output)]) == 0
    assert read_jsonl(output) == [{"id": 1}, {"id": 2}, {"id": 3}]
