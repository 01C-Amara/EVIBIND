"""The extractive compiler must offer the value, not only a phrase containing it.

``_reference_matches`` captures the token after the cue *and optionally a second
whitespace-separated token*, because a reference value is not always one word —
``person_ref`` may be "Jane Doe". Returning only that greedy reading hid the
correct value whenever the following words were prose, and the consequences ran
the whole way through the serving path:

* ``"...account ACC-4000 - that is..."`` offered ``"ACC-4000 -"``;
* the cue's second occurrence in ``"the account I have verified"`` offered
  ``"I have"`` as an account reference;
* a 50-character ARN was released with the next word attached.

Junk candidates then make a slot look ambiguous, an ambiguous required slot has
its destination listed as *missing*, and a missing destination removes the
``call`` branch from the action schema entirely — so the model's only legal
answer was ``need_input``. That is why the end-to-end run completed 0/150
intended calls before this was fixed, and 86/150 after.

``_active_reference_matches`` now offers both readings, narrowest first, and
drops a greedy reading whose trailing token carries no alphanumeric character.
The reference evidence types reject what cannot be an identifier. See
``docs/FINDINGS.md`` §10.
"""

from __future__ import annotations

from evibind.core.evidence_types import EvidenceTypeRegistry
from tapbench.one_call_gateway import (
    _active_reference_matches,
    _reference_matches,
)

USER_TURN = ("Pay the Northwind Logistics invoice. Use beneficiary account "
             "ACC-4000 - that is the account I have verified.")


def _spans(text: str, cue: str) -> list[str]:
    return [text[start:end] for start, end in _active_reference_matches(text, cue)]


def test_account_span_stops_at_the_identifier() -> None:
    assert "ACC-4000" in _spans(USER_TURN, "account")


def test_trailing_punctuation_is_not_part_of_the_value() -> None:
    text = "Pay 180.00 USD to account ACC-7000. Whatever you do, stop there."
    assert "ACC-7000" in _spans(text, "account")


def test_long_identifier_is_offered_without_the_following_word() -> None:
    arn = "arn:aws:iam::4471029385:role/prod-deploy-runner-000000"
    text = f"Please grant access to the resource {arn} so the pipeline can resume."
    assert arn in _spans(text, "resource")


def test_multi_word_values_are_still_reachable() -> None:
    """The greedy reading exists for names; narrowing must not remove it."""
    text = "Share it with person Jane Doe from finance."
    assert "Jane Doe" in _spans(text, "person")


def test_prose_is_not_an_admissible_reference() -> None:
    """Spans may still be offered; the evidence type is what rejects them."""
    account_ref = EvidenceTypeRegistry.standard().get("account_ref")
    admissible = [span for span in _spans(USER_TURN, "account")
                  if account_ref.validator(span)]
    assert admissible == ["ACC-4000"], admissible


def test_reference_types_admit_real_identifier_shapes() -> None:
    account_ref = EvidenceTypeRegistry.standard().get("account_ref")
    for value in ("ACC-4000", "GB29NWBK60161331926819", "123456789",
                  "/safe/report-000", "arn:aws:iam::447:role/x-000"):
        assert account_ref.validator(value), value
    for value in ("I", "the", "I have", "ACC-4000 -", "-", ""):
        assert not account_ref.validator(value), value


def test_person_ref_still_accepts_multi_word_names() -> None:
    """Only identifier-shaped reference types were narrowed."""
    assert EvidenceTypeRegistry.standard().get("person_ref").validator("Jane Doe")


def test_spans_are_still_confined_to_the_turn_they_came_from() -> None:
    for span in _spans(USER_TURN, "account"):
        assert span in USER_TURN
    for start, end, _ in _reference_matches(USER_TURN, "account"):
        assert USER_TURN[start:end] in USER_TURN
