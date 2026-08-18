from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .derivations import EvidenceContext, EvidenceDerivation, root_derivations
from .policy import root_source


TRUST_MODEL_VERSION = "evibind.trust.v1"


class TrustLabel(StrEnum):
    USER_EXPLICIT = "user_explicit"
    USER_CONTEXT = "user_context"
    STATE_AUTHORIZED = "state_authorized"
    SCHEMA_OWNED = "schema_owned"
    TOOL_UNTRUSTED = "tool_untrusted"
    MODEL_UNTRUSTED = "model_untrusted"
    UNKNOWN = "unknown"


MATERIALIZATION_CAPABLE_LABELS = frozenset(
    {
        TrustLabel.USER_EXPLICIT,
        TrustLabel.USER_CONTEXT,
        TrustLabel.STATE_AUTHORIZED,
        TrustLabel.SCHEMA_OWNED,
    }
)
EFFECT_AUTHORIZING_LABELS = frozenset({TrustLabel.USER_EXPLICIT})
UNTRUSTED_LABELS = frozenset(
    {
        TrustLabel.TOOL_UNTRUSTED,
        TrustLabel.MODEL_UNTRUSTED,
        TrustLabel.UNKNOWN,
    }
)


def source_trust_label(source: str) -> TrustLabel:
    if source == "user.current_turn":
        return TrustLabel.USER_EXPLICIT
    if source.startswith("user."):
        return TrustLabel.USER_CONTEXT
    if source.startswith("state."):
        return TrustLabel.STATE_AUTHORIZED
    if source.startswith("schema."):
        return TrustLabel.SCHEMA_OWNED
    if source.startswith("tool."):
        return TrustLabel.TOOL_UNTRUSTED
    if source.endswith(".untrusted"):
        return TrustLabel.MODEL_UNTRUSTED
    return TrustLabel.UNKNOWN


@dataclass(frozen=True)
class TrustAssessment:
    sources: tuple[str, ...]
    labels: tuple[TrustLabel, ...]

    @property
    def contains_untrusted(self) -> bool:
        return bool(set(self.labels) & UNTRUSTED_LABELS)

    @property
    def materialization_capable(self) -> bool:
        labels = set(self.labels)
        return bool(labels) and labels <= MATERIALIZATION_CAPABLE_LABELS

    @property
    def explicitly_effect_authorizing(self) -> bool:
        labels = set(self.labels)
        return bool(labels & EFFECT_AUTHORIZING_LABELS) and not self.contains_untrusted

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": TRUST_MODEL_VERSION,
            "sources": list(self.sources),
            "labels": [label.value for label in self.labels],
            "contains_untrusted": self.contains_untrusted,
            "materialization_capable": self.materialization_capable,
            "explicitly_effect_authorizing": self.explicitly_effect_authorizing,
        }


def assess_derivation_trust(
    derivation: EvidenceDerivation,
    context: EvidenceContext,
) -> TrustAssessment:
    sources = tuple(
        sorted({root_source(root, context) for root in root_derivations(derivation)})
    )
    labels = tuple(sorted({source_trust_label(source) for source in sources}))
    return TrustAssessment(sources=sources, labels=labels)


def combine_trust_assessments(
    assessments: tuple[TrustAssessment, ...],
) -> TrustAssessment:
    return TrustAssessment(
        sources=tuple(
            sorted(
                {source for assessment in assessments for source in assessment.sources}
            )
        ),
        labels=tuple(
            sorted({label for assessment in assessments for label in assessment.labels})
        ),
    )
