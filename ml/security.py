"""API-key auth (fail closed) and a small in-process sliding-window rate limiter."""

from __future__ import annotations

import hmac
import math
import threading
import time
from collections import deque

from fastapi import Header, HTTPException, Request

from ml.config import get_settings


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """Guard write/import endpoints. Fails closed when no key is configured."""
    expected = get_settings().api_key
    if not expected:
        raise HTTPException(503, "Writes disabled: EXPENSE_API_KEY is not configured")
    if not x_api_key or not hmac.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(401, "Invalid or missing API key", headers={"WWW-Authenticate": "ApiKey"})


class SlidingWindowRateLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def hit(self, key: str, limit: int, window_s: float = 60.0) -> float | None:
        """Return None if allowed, else seconds until a retry would be allowed."""
        now = time.monotonic()
        with self._lock:
            q = self._hits.setdefault(key, deque())
            while q and now - q[0] >= window_s:
                q.popleft()
            if len(q) >= limit:
                return max(window_s - (now - q[0]), 0.0)
            q.append(now)
            return None

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


limiter = SlidingWindowRateLimiter()


def _enforce(request: Request, bucket: str, limit: int) -> None:
    host = request.client.host if request.client else "unknown"
    retry = limiter.hit(f"{bucket}:{host}", limit)
    if retry is not None:
        raise HTTPException(429, "Too many requests", headers={"Retry-After": str(max(math.ceil(retry), 1))})


def rate_limit_default(request: Request) -> None:
    _enforce(request, "default", get_settings().rate_limit_per_minute)


def rate_limit_feedback(request: Request) -> None:
    _enforce(request, "feedback", get_settings().feedback_rate_limit_per_minute)
