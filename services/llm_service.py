"""
Groq LLM wrapper.

Responsibilities:
- Call the configured Groq model at low temperature.
- Enforce JSON-only output as reliably as the API allows, with exactly
  ONE bounded repair attempt if the first response is empty, unparseable,
  or the provider itself rejects it as invalid JSON.
- Proactively throttle actual Groq calls to at most one per
  GROQ_MIN_REQUEST_INTERVAL_SECONDS (default 20s), to reduce how often a
  limited free-tier TPM ceiling is hit in the first place. This does NOT
  guarantee staying under any given TPM limit — actual token usage per
  call still varies — it only reduces 429 frequency by spacing requests.
- Retry TRANSIENT failures only. A daily token-quota (TPD) exhaustion is
  non-retryable and trips a circuit breaker. A per-minute (TPM/RPM) 429
  gets exactly one bounded retry using the provider's own suggested wait
  time (capped), separate from and not doubled up with the proactive
  throttle above.
- Never invent data on parse failure — callers get a failure result and
  must treat the field as unknown, never as a guess.

This service does not make qualification decisions — that stays in
core.validators. It only extracts/generates text or structured JSON.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from groq import Groq, APIError, APIConnectionError, APITimeoutError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.config import Settings, settings as default_settings

logger = logging.getLogger(__name__)

RETRYABLE_EXCEPTIONS = (APIConnectionError, APITimeoutError)

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

_DAILY_QUOTA_MARKERS = ("tokens per day", "tpd", "per day")
_PER_MINUTE_MARKERS = ("tokens per minute", "tpm", "per minute", "requests per minute", "rpm")

# Cap how long we'll actually wait for a per-minute rate limit, even if
# Groq's error suggests longer. This is a short, bounded wait for a
# window that resets every 60s — NOT the same class of problem as a
# daily quota exhaustion, which is never retried.
_MAX_TPM_RETRY_WAIT_SECONDS = 15.0


class LLMServiceError(Exception):
    """Raised when the LLM call fails after retries or returns unusable output."""


class LLMQuotaExceededError(LLMServiceError):
    """Daily token-quota exhaustion — never retried, trips LLMService.quota_exceeded."""


def _is_daily_quota_error(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _DAILY_QUOTA_MARKERS)


def _is_per_minute_error(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _PER_MINUTE_MARKERS)


def _extract_suggested_wait_seconds(message: str) -> float:
    """
    Groq's 429 body includes text like 'Please try again in 8.2274s'.
    Parses that out; falls back to a small default if not found.
    """
    match = re.search(r"try again in\s+([\d.]+)\s*s", message.lower())
    if not match:
        return 5.0
    try:
        return min(float(match.group(1)), _MAX_TPM_RETRY_WAIT_SECONDS)
    except ValueError:
        return 5.0


class _RateLimiter:
    """
    Thread-safe minimum-interval limiter. Ensures at least
    `min_interval_seconds` elapses between successive calls to `.wait()`,
    sleeping the caller if a call arrives too soon. The very first call
    never waits. Safe to share across threads even though the current
    pipeline calls it sequentially — a lock prevents two threads from
    both reading a stale "last call time" and proceeding simultaneously.

    This is a proactive spacing mechanism, not a token-budget tracker: it
    reduces how often the actual per-minute limit is hit, but does not
    and cannot guarantee staying under any specific token ceiling, since
    real token usage per call varies.
    """

    def __init__(self, min_interval_seconds: float):
        self.min_interval_seconds = max(0.0, min_interval_seconds)
        self._lock = threading.Lock()
        self._last_call_at: Optional[float] = None

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if self._last_call_at is not None:
                elapsed = now - self._last_call_at
                remaining = self.min_interval_seconds - elapsed
                if remaining > 0:
                    logger.info("Groq throttle: waiting %.1fs before next extraction", remaining)
                    time.sleep(remaining)
                    now = time.monotonic()
            self._last_call_at = now


@dataclass
class LLMTextResult:
    success: bool
    text: str = ""
    error: str | None = None


@dataclass
class LLMJSONResult:
    success: bool
    data: Optional[Any] = None
    raw_text: str = ""
    error: str | None = None


class LLMService:
    def __init__(self, settings: Settings | None = None, client: Groq | None = None):
        self.settings = settings or default_settings
        self._client = client
        self.quota_exceeded: bool = False
        self.quota_exceeded_reason: str | None = None
        self._rate_limiter = _RateLimiter(self.settings.groq_min_request_interval_seconds)

    @property
    def client(self) -> Groq:
        if self._client is None:
            if not self.settings.groq_api_key:
                raise LLMServiceError("GROQ_API_KEY not configured")
            # max_retries=0 disables the SDK's own built-in retry-with-backoff
            # on 429/5xx, so our own bounded retry logic below is the single
            # source of truth for retries (avoids hidden multi-second waits
            # inside the SDK stacking with our own handling).
            self._client = Groq(api_key=self.settings.groq_api_key, max_retries=0)
        return self._client

    def generate_text(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> LLMTextResult:
        if self.quota_exceeded:
            return LLMTextResult(success=False, error=f"LLM quota exhausted for this run: {self.quota_exceeded_reason}")
        try:
            completion = self._call_with_retry(system_prompt, user_prompt, max_tokens, json_mode=False)
        except LLMServiceError as exc:
            return LLMTextResult(success=False, error=str(exc))

        text = self._extract_content(completion)
        if text is None:
            return LLMTextResult(success=False, error="Empty completion from LLM")
        return LLMTextResult(success=True, text=text)

    def generate_json(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> LLMJSONResult:
        """
        Requests JSON-only output. Makes at most TWO Groq calls: the
        original request, and — only if that one came back empty,
        unparseable, or was rejected by the provider as invalid JSON — one
        repair attempt with a stricter instruction appended. Each of these
        two calls independently goes through the proactive rate limiter
        (via _call_with_retry), so a repair attempt is also spaced out
        from whatever call preceded it.
        """
        if self.quota_exceeded:
            return LLMJSONResult(success=False, error=f"LLM quota exhausted for this run: {self.quota_exceeded_reason}")

        attempts = [user_prompt, user_prompt + _REPAIR_SUFFIX]
        last_error = "Unknown error"

        for attempt_index, prompt in enumerate(attempts):
            is_last_attempt = attempt_index == len(attempts) - 1

            try:
                completion = self._call_with_retry(system_prompt, prompt, max_tokens, json_mode=True)
            except LLMQuotaExceededError as exc:
                return LLMJSONResult(success=False, error=str(exc))
            except LLMServiceError as exc:
                last_error = str(exc)
                if not is_last_attempt and _looks_like_fixable_json_error(last_error):
                    logger.info("Groq rejected JSON output, retrying once with stricter prompt")
                    continue
                return LLMJSONResult(success=False, error=last_error)

            raw_text = self._extract_content(completion)
            if raw_text is None:
                last_error = "Empty completion from LLM"
                if not is_last_attempt:
                    continue
                return LLMJSONResult(success=False, error=last_error)

            parsed = self._safe_parse_json(raw_text)
            if parsed is None:
                last_error = "Could not parse valid JSON from LLM output"
                if not is_last_attempt:
                    logger.info("Failed to parse JSON from LLM output, retrying once with stricter prompt")
                    continue
                return LLMJSONResult(success=False, raw_text=raw_text, error=last_error)

            return LLMJSONResult(success=True, data=parsed, raw_text=raw_text)

        return LLMJSONResult(success=False, error=last_error)

    @staticmethod
    def _safe_parse_json(text: str) -> Optional[Any]:
        cleaned = _JSON_FENCE_RE.sub("", text).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        for open_ch, close_ch in (("{", "}"), ("[", "]")):
            start = cleaned.find(open_ch)
            end = cleaned.rfind(close_ch)
            if start != -1 and end != -1 and end > start:
                candidate = cleaned[start:end + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue
        return None

    @staticmethod
    def _extract_content(completion: Any) -> Optional[str]:
        try:
            content = completion.choices[0].message.content
        except (AttributeError, IndexError, TypeError):
            return None
        if content is None or content.strip() == "":
            return None
        return content

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        reraise=True,
    )
    def _call_with_retry(self, system_prompt: str, user_prompt: str, max_tokens: int, json_mode: bool):
        # Proactive throttle: applied once per invocation of this method,
        # BEFORE the first attempt. Not re-applied before the TPM-retry
        # sub-call below, since that already performs its own explicit,
        # bounded wait — avoids stacking two waits for one 429.
        self._rate_limiter.wait()

        kwargs: dict[str, Any] = dict(
            model=self.settings.llm_model,
            temperature=self.settings.llm_temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            return self.client.chat.completions.create(**kwargs)
        except RateLimitError as exc:
            message = str(exc)
            if _is_daily_quota_error(message):
                self.quota_exceeded = True
                self.quota_exceeded_reason = message
                logger.error(
                    "Groq daily token quota exceeded for model %s — disabling further LLM "
                    "calls for the rest of this run: %s",
                    self.settings.llm_model, message,
                )
                raise LLMQuotaExceededError(
                    f"Daily token quota exceeded for model {self.settings.llm_model}: {message}"
                ) from exc

            if _is_per_minute_error(message):
                wait_seconds = _extract_suggested_wait_seconds(message)
                logger.warning(
                    "Transient per-minute Groq rate limit hit — waiting %.1fs and retrying once: %s",
                    wait_seconds, message,
                )
                time.sleep(wait_seconds)
                try:
                    return self.client.chat.completions.create(**kwargs)
                except RateLimitError as retry_exc:
                    retry_message = str(retry_exc)
                    if _is_daily_quota_error(retry_message):
                        self.quota_exceeded = True
                        self.quota_exceeded_reason = retry_message
                        raise LLMQuotaExceededError(
                            f"Daily token quota exceeded for model {self.settings.llm_model}: {retry_message}"
                        ) from retry_exc
                    logger.warning("Per-minute rate limit still hit after one retry, giving up on this call: %s", retry_message)
                    raise LLMServiceError(f"Groq rate limit (after retry): {retry_message}") from retry_exc

            logger.warning("Transient Groq rate limit hit: %s", message)
            raise LLMServiceError(f"Groq rate limit: {message}") from exc
        except RETRYABLE_EXCEPTIONS:
            raise
        except APIError as exc:
            raise LLMServiceError(f"Groq API error: {exc}") from exc


_REPAIR_SUFFIX = (
    "\n\nIMPORTANT: your previous response was rejected because it was not valid JSON "
    "(or was cut off). Return ONLY a single complete, valid JSON object matching the "
    "schema above. No markdown fences, no explanation, nothing before or after the JSON."
)


def _looks_like_fixable_json_error(message: str) -> bool:
    lowered = message.lower()
    return "valid json" in lowered or "adjust your prompt" in lowered