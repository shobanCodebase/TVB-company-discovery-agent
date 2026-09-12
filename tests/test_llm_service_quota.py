import pytest
from groq import RateLimitError

from core.config import Settings
from services.llm_service import LLMQuotaExceededError, LLMService, _is_daily_quota_error


def make_settings() -> Settings:
    return Settings(groq_api_key="fake-key", llm_model="test-model")


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeCompletion:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeRateLimitError(Exception):
    """Stand-in for groq.RateLimitError in unit tests (real one needs an httpx response)."""


def test_is_daily_quota_error_detects_tpd_message():
    msg = "Rate limit reached for model x. tokens per day (TPD): Limit 200000 Used 199000"
    assert _is_daily_quota_error(msg) is True


def test_is_daily_quota_error_ignores_per_minute_message():
    msg = "Rate limit reached: requests per minute exceeded, please slow down"
    assert _is_daily_quota_error(msg) is False


def test_daily_quota_error_sets_flag_and_short_circuits_further_calls(monkeypatch):
    service = LLMService(settings=make_settings())

    call_count = {"n": 0}

    def fake_call(system_prompt, user_prompt, max_tokens, json_mode):
        call_count["n"] += 1
        raise LLMQuotaExceededError("Daily token quota exceeded for model test-model: tokens per day")

    monkeypatch.setattr(service, "_call_with_retry", fake_call)

    result1 = service.generate_json("sys", "user")
    assert result1.success is False
    assert call_count["n"] == 1

    # quota_exceeded should now short-circuit WITHOUT calling _call_with_retry again
    service.quota_exceeded = True
    service.quota_exceeded_reason = "tokens per day exhausted"
    result2 = service.generate_json("sys", "user")
    assert result2.success is False
    assert "quota exhausted" in result2.error.lower()
    assert call_count["n"] == 1  # unchanged — no second API attempt


def test_generate_text_also_short_circuits_on_quota_exceeded():
    service = LLMService(settings=make_settings())
    service.quota_exceeded = True
    service.quota_exceeded_reason = "tokens per day"
    result = service.generate_text("sys", "user")
    assert result.success is False
    assert "quota exhausted" in result.error.lower()


def test_transient_rate_limit_does_not_set_quota_flag(monkeypatch):
    service = LLMService(settings=make_settings())

    def fake_call(system_prompt, user_prompt, max_tokens, json_mode):
        from services.llm_service import LLMServiceError
        raise LLMServiceError("Groq rate limit: requests per minute exceeded")

    monkeypatch.setattr(service, "_call_with_retry", fake_call)
    result = service.generate_json("sys", "user")
    assert result.success is False
    assert service.quota_exceeded is False

def test_per_minute_rate_limit_retries_once_and_succeeds(monkeypatch):
    service = LLMService(settings=make_settings())
    call_count = {"n": 0}
    sleep_calls = []

    monkeypatch.setattr("services.llm_service.time.sleep", lambda s: sleep_calls.append(s))

    class FakeRateLimitError(Exception):
        def __str__(self):
            return "Rate limit reached ... tokens per minute (TPM): Limit 8000 ... Please try again in 8.2274s."

    def fake_create(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise FakeRateLimitError()
        return FakeCompletion('{"ok": true}')

    monkeypatch.setattr("services.llm_service.RateLimitError", FakeRateLimitError)

    fake_client = type("C", (), {
        "chat": type("Chat", (), {"completions": type("Comp", (), {"create": staticmethod(fake_create)})()})()
    })()
    service._client = fake_client  # bypass the read-only `client` property

    result = service.generate_json("sys", "user")
    assert result.success is True
    assert call_count["n"] == 2
    assert sleep_calls == [8.2274]


def test_per_minute_rate_limit_gives_up_after_one_retry(monkeypatch):
    service = LLMService(settings=make_settings())
    call_count = {"n": 0}

    monkeypatch.setattr("services.llm_service.time.sleep", lambda s: None)

    class FakeRateLimitError(Exception):
        def __str__(self):
            return "Rate limit reached ... tokens per minute (TPM): Limit 8000 ... Please try again in 5s."

    def fake_create(**kwargs):
        call_count["n"] += 1
        raise FakeRateLimitError()

    monkeypatch.setattr("services.llm_service.RateLimitError", FakeRateLimitError)

    fake_client = type("C", (), {
        "chat": type("Chat", (), {"completions": type("Comp", (), {"create": staticmethod(fake_create)})()})()
    })()
    service._client = fake_client  # bypass the read-only `client` property

    result = service.generate_json("sys", "user")
    assert result.success is False
    assert call_count["n"] == 2  # initial + exactly one retry, never more


def test_extract_suggested_wait_seconds_parses_and_caps():
    from services.llm_service import _extract_suggested_wait_seconds
    assert _extract_suggested_wait_seconds("Please try again in 8.2274s.") == 8.2274
    assert _extract_suggested_wait_seconds("Please try again in 45s.") == 15.0  # capped
    assert _extract_suggested_wait_seconds("no timing info here") == 5.0  # fallback


def test_is_per_minute_error_detects_tpm_and_rpm():
    from services.llm_service import _is_per_minute_error
    assert _is_per_minute_error("Rate limit ... tokens per minute (TPM) ...") is True
    assert _is_per_minute_error("Rate limit ... requests per minute exceeded") is True
    assert _is_per_minute_error("Rate limit ... tokens per day (TPD) ...") is False