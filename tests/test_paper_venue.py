"""Venue rules the paper audit applies to the canonical source.

The paper targets TMLR. ICLR 2027 was the first target and was never submitted
to; its profile stays so the frozen v8 evidence bundle, audited under ICLR
rules, still reproduces.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tapbench.paper_audit import (
    DEFAULT_VENUE,
    PaperAuditError,
    venue_failures,
)

TMLR_MAIN = r"""
\documentclass[10pt]{article}
\usepackage{tmlr}
\begin{document}
\subsubsection*{Broader Impact Statement}
\bibliographystyle{tmlr}
\bibliography{references}
\appendix
\end{document}
"""


def _paper(tmp_path: Path, *styles: str) -> Path:
    for name in styles:
        (tmp_path / name).write_text("% style\n", encoding="utf-8")
    return tmp_path


def test_default_venue_is_tmlr() -> None:
    assert DEFAULT_VENUE == "tmlr"


def test_complete_tmlr_source_passes(tmp_path: Path) -> None:
    paper = _paper(tmp_path, "tmlr.sty", "tmlr.bst")
    assert venue_failures(TMLR_MAIN, TMLR_MAIN, paper, "tmlr") == []


def test_tmlr_has_no_page_rule_to_fail(tmp_path: Path) -> None:
    # TMLR sets no page limit, so a long paper is not a venue failure; an
    # invented limit would push caveats out of the main text again.
    from tapbench.paper_audit import VENUE_PROFILES

    assert VENUE_PROFILES["tmlr"]["page_limit"] is None
    assert VENUE_PROFILES["iclr2027"]["page_limit"] == 9


def test_missing_broader_impact_statement_fails(tmp_path: Path) -> None:
    paper = _paper(tmp_path, "tmlr.sty", "tmlr.bst")
    main = TMLR_MAIN.replace(r"\subsubsection*{Broader Impact Statement}", "")
    assert "missing_section:Broader Impact Statement" in venue_failures(
        main, main, paper, "tmlr"
    )


def test_deanonymized_build_fails(tmp_path: Path) -> None:
    # `[preprint]` de-anonymizes; a blind submission must use the bare package.
    paper = _paper(tmp_path, "tmlr.sty", "tmlr.bst")
    main = TMLR_MAIN.replace(r"\usepackage{tmlr}", r"\usepackage[preprint]{tmlr}")
    assert any(f.startswith("missing_submission_element:") for f in
               venue_failures(main, main, paper, "tmlr"))


def test_link_to_the_public_repository_fails(tmp_path: Path) -> None:
    paper = _paper(tmp_path, "tmlr.sty", "tmlr.bst")
    main = TMLR_MAIN + r"\url{https://github.com/example-org/EVIBIND}"
    failures = venue_failures(main, main, paper, "tmlr",
                              ["github.com/example-org/EVIBIND"])
    assert failures == ["deanonymizing_link:github.com/example-org/EVIBIND"]


def test_missing_official_style_files_fail(tmp_path: Path) -> None:
    failures = venue_failures(TMLR_MAIN, TMLR_MAIN, tmp_path, "tmlr")
    assert "missing_official_style:tmlr.sty" in failures
    assert "missing_official_style:tmlr.bst" in failures


def test_iclr_profile_still_checks_the_legacy_rules(tmp_path: Path) -> None:
    failures = venue_failures(TMLR_MAIN, TMLR_MAIN, tmp_path, "iclr2027")
    assert r"missing_submission_element:\usepackage{iclr2027_conference,times}" in failures
    assert r"missing_submission_element:\author{Anonymous authors}" in failures


def test_unknown_venue_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(PaperAuditError):
        venue_failures(TMLR_MAIN, TMLR_MAIN, tmp_path, "neurips2027")


def test_repository_link_is_caught_in_any_case_and_without_host(tmp_path: Path) -> None:
    (tmp_path / "CITATION.cff").write_text(
        'repository-code: "https://github.com/Example-Org/EVIBIND"\n', encoding="utf-8")
    from tapbench.paper_audit import _identifying_urls

    urls = _identifying_urls(tmp_path)
    paper = _paper(tmp_path, "tmlr.sty", "tmlr.bst")
    for leak in ("https://github.com/example-org/evibind", "see Example-Org/EVIBIND"):
        main = TMLR_MAIN + leak
        assert any(f.startswith("deanonymizing_link:")
                   for f in venue_failures(main, main, paper, "tmlr", urls)), leak


def test_numbered_broader_impact_heading_is_accepted(tmp_path: Path) -> None:
    paper = _paper(tmp_path, "tmlr.sty", "tmlr.bst")
    main = TMLR_MAIN.replace(r"\subsubsection*{Broader Impact Statement}",
                             r"\section{Broader Impact Statement}")
    assert venue_failures(main, main, paper, "tmlr") == []
