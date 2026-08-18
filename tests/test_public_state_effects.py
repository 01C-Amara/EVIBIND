from __future__ import annotations

from evibind.effects import (
    EFFECT_AUTHORIZATION_VERSION,
    EffectAuthorizer,
    InMemoryConsumedNonceStore,
    EffectPolicy,
)
from evibind.execution import (
    EXECUTION_COORDINATOR_VERSION,
    EXECUTION_GRAPH_VERSION,
    ExecutionCoordinator,
    ExecutionRecord,
)


def test_state_and_effect_product_namespaces_are_importable() -> None:
    assert EXECUTION_GRAPH_VERSION == "evibind.execution_graph.v1"
    assert EXECUTION_COORDINATOR_VERSION == "evibind.execution_coordinator.v1"
    assert EFFECT_AUTHORIZATION_VERSION == "evibind.effect_authorization.v1"
    assert ExecutionCoordinator.__name__ == "ExecutionCoordinator"
    assert ExecutionRecord.__name__ == "ExecutionRecord"
    assert EffectAuthorizer.__name__ == "EffectAuthorizer"
    assert EffectPolicy.__name__ == "EffectPolicy"
    assert InMemoryConsumedNonceStore.__name__ == "InMemoryConsumedNonceStore"
