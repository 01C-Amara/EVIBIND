"""Pin the extractive over-capture that blocks the serving path.

``_reference_matches`` captures the token after the cue *and optionally a
second whitespace-separated token*::

    (?:"([^"]+)"|'([^']+)'|([A-Za-z0-9_@.+:/-]+(?:\\s+[A-Za-z0-9_@.+:/-]+)?))
                                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^ here

That second token is why every observed candidate defect looks different but
has one cause. Reproduced live against ``gpt-5.4-nano`` through
``evibind serve`` on 2026-08-18:

* ``"Use beneficiary account ACC-4000 - that is ..."`` yields the candidate
  ``"ACC-4000 -"``, not ``"ACC-4000"``;
* the second occurrence of the cue, in ``"the account I have verified"``,
  yields the candidate ``"I have"``;
* a 50-character ARN is released as ``"arn:...runner-000000 so"``, swallowing
  the following word — 15/15 of the ``transcription`` cases in the end-to-end
  run.

The practical consequence is not a leak: no untrusted-origin value escapes, so
materialization confinement holds. It is that the *correct* value is never
offered cleanly, so the model sees an over-captured value beside a junk one,
judges the slot ambiguous, and returns ``need_input``. That is why the
end-to-end run releases 0/150 intended calls while the offline arm repairs
28/43 for the same model — see ``docs/FINDINGS.md`` §10.

``docs/FINDINGS.md`` §4 already recommended the fix: validate a span against
its declared evidence type before it enters the candidate table. These tests
are the tripwire. They are ``strict``, so they XPASS and fail the moment the
extraction is corrected, as a prompt to update the docs.
"""

from __future__ import annotations

import pytest

from tapbench.one_call_gateway import _reference_matches

USER_TURN = ("Pay the Northwind Logistics invoice. Use beneficiary account "
             "ACC-4000 - that is the account I have verified.")


def _captures(text: str, cue: str) -> list[str]:
    return [text[start:end] for start, end, _ in _reference_matches(text, cue)]


@pytest.mark.xfail(strict=True,
                   reason="cue capture takes one extra token; yields 'ACC-4000 -'")
def test_account_span_stops_at_the_identifier() -> None:
    assert "ACC-4000" in _captures(USER_TURN, "account")


@pytest.mark.xfail(strict=True,
                   reason="second cue occurrence yields the junk candidate 'I have'")
def test_prose_after_a_cue_is_not_a_candidate() -> None:
    captures = _captures(USER_TURN, "account")
    assert "I have" not in captures, (
        f"'I have' is not an account reference, but was offered as one: {captures}")


@pytest.mark.xfail(strict=True,
                   reason="cue capture swallows the word after a long identifier")
def test_long_identifier_is_not_extended_by_the_next_word() -> None:
    text = ("Revoke the role arn:aws:iam::4471029385:role/prod-deploy-runner-000000 "
            "so the review can close.")
    assert "arn:aws:iam::4471029385:role/prod-deploy-runner-000000" in _captures(
        text, "role")


def test_over_capture_is_still_confined_to_the_user_turn() -> None:
    """Whatever the span boundaries, candidates never come from elsewhere."""
    for capture in _captures(USER_TURN, "account"):
        assert capture in USER_TURN
