from __future__ import annotations

import threading
from typing import Protocol


class ConsumedNonceStore(Protocol):
    """Linearizable single-consumption boundary for effect tokens.

    Every gateway worker that accepts tokens from the same issuer namespace
    must share one implementation.  ``consume_once`` is a compare-and-set:
    concurrent calls for an unexpired nonce have exactly one ``True`` result;
    later calls return ``False`` through ``expires_at``.  Implementations may
    garbage-collect entries only after expiry, must use a consistent trusted
    time basis for ``now``, and must fail closed (raise) when the shared store
    is unavailable.  A process-local store satisfies the contract only for a
    single-process deployment.
    """

    def consume_once(
        self,
        *,
        nonce: str,
        expires_at: int,
        now: int,
    ) -> bool:
        """Atomically consume a nonce, returning false when already consumed."""


class InMemoryConsumedNonceStore:
    def __init__(self) -> None:
        self._entries: dict[str, int] = {}
        self._lock = threading.Lock()

    def consume_once(
        self,
        *,
        nonce: str,
        expires_at: int,
        now: int,
    ) -> bool:
        with self._lock:
            self._entries = {
                consumed_nonce: expiry
                for consumed_nonce, expiry in self._entries.items()
                if expiry >= now
            }
            if nonce in self._entries:
                return False
            self._entries[nonce] = expires_at
            return True

    @property
    def entry_count(self) -> int:
        with self._lock:
            return len(self._entries)
