"""Trusted state is how a document-sourced value becomes protectable.

`bench/agentdojo/` measures the limit this addresses: only about a third of a
realistic agent's action-critical arguments are values the user actually wrote.
For the rest — an IBAN inside an invoice, a record ID inside a search result —
the authorised value and the attack arrive through the same channel, and a slot
bound to `user.current_turn` can only withhold.

The application already knows how to fetch that value properly. Passing it in
`evibind.dialogue_state` and marking the slot `source-policy: trusted_state_only`
gives the boundary a channel the attacker cannot write to, which is the whole
requirement. These tests pin both halves so the example cannot rot.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))

from trusted_state_binding import (  # noqa: E402
    ATTACKER_IBAN,
    AUTHORISED_IBAN,
    run,
)


def test_user_turn_binding_withholds_a_document_sourced_value() -> None:
    """Safe, and unable to complete: the user never typed an IBAN."""
    outcome, _ = run(trusted_state=False)
    assert outcome == "withheld"


def test_trusted_state_releases_the_authorised_value() -> None:
    outcome, released = run(trusted_state=True)
    assert outcome == "released the authorised IBAN"
    assert released == AUTHORISED_IBAN


def test_the_injected_value_never_reaches_the_tool_either_way() -> None:
    """The attacker's IBAN is what the model proposed, in both arms."""
    for trusted_state in (False, True):
        _, detail = run(trusted_state=trusted_state)
        assert ATTACKER_IBAN not in detail, trusted_state
