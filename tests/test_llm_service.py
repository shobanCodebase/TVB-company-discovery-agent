import pytest

from core.config import Settings
from services.llm_service import LLMService, LLMServiceError
import threading

def make_settings() -> Settings:
    return Settings(groq_api_key="fake-key", llm_model="test-model", llm_temperature=0.1)


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeCompletion:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


def test_generate_json_parses_clean_json(monkeypatch):
    service = LLMService(settings=make_settings())
    monkeypatch.setattr(
        service, "_call_with_retry",
        lambda system_prompt, user_prompt, max_tokens, json_mode: FakeCompletion('{"queries": ["a", "b"]}'),
    )
    result = service.generate_json("sys", "user")
    assert result.success is True
    assert result.data == {"queries": ["a", "b"]}


def test_generate_json_strips_markdown_fences(monkeypatch):
    service = LLMService(settings=make_settings())
    fenced = '```json\n{"queries": ["a"]}\n```'
    monkeypatch.setattr(
        service, "_call_with_retry",
        lambda system_prompt, user_prompt, max_tokens, json_mode: FakeCompletion(fenced),
    )
    result = service.generate_json("sys", "user")
    assert result.success is True
    assert result.data == {"queries": ["a"]}


def test_generate_json_handles_unparseable_output(monkeypatch):
    service = LLMService(settings=make_settings())
    monkeypatch.setattr(
        service, "_call_with_retry",
        lambda system_prompt, user_prompt, max_tokens, json_mode: FakeCompletion("this is not json at all"),
    )
    result = service.generate_json("sys", "user")
    assert result.success is False
    assert result.data is None


def test_generate_json_handles_empty_completion(monkeypatch):
    service = LLMService(settings=make_settings())
    monkeypatch.setattr(
        service, "_call_with_retry",
        lambda system_prompt, user_prompt, max_tokens, json_mode: FakeCompletion(""),
    )
    result = service.generate_json("sys", "user")
    assert result.success is False
    assert "Empty completion" in result.error


def test_generate_json_handles_llm_service_error(monkeypatch):
    service = LLMService(settings=make_settings())

    def raise_error(*args, **kwargs):
        raise LLMServiceError("simulated API failure")

    monkeypatch.setattr(service, "_call_with_retry", raise_error)
    result = service.generate_json("sys", "user")
    assert result.success is False
    assert "simulated API failure" in result.error


def test_generate_json_extracts_embedded_json_block(monkeypatch):
    service = LLMService(settings=make_settings())
    messy = 'Sure, here you go: {"queries": ["x", "y"]} — hope that helps!'
    monkeypatch.setattr(
        service, "_call_with_retry",
        lambda system_prompt, user_prompt, max_tokens, json_mode: FakeCompletion(messy),
    )
    result = service.generate_json("sys", "user")
    assert result.success is True
    assert result.data == {"queries": ["x", "y"]}


def test_generate_text_returns_plain_text(monkeypatch):
    service = LLMService(settings=make_settings())
    monkeypatch.setattr(
        service, "_call_with_retry",
        lambda system_prompt, user_prompt, max_tokens, json_mode: FakeCompletion("Hello world"),
    )
    result = service.generate_text("sys", "user")
    assert result.success is True
    assert result.text == "Hello world"


def test_client_raises_without_api_key():
    service = LLMService(settings=Settings(groq_api_key=""))
    with pytest.raises(LLMServiceError):
        _ = service.client

def test_generate_json_repairs_after_first_unparseable_response(monkeypatch):
    service = LLMService(settings=make_settings())
    call_count = {"n": 0}

    def fake_call(system_prompt, user_prompt, max_tokens, json_mode):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return FakeCompletion("this is not json")
        return FakeCompletion('{"queries": ["a"]}')

    monkeypatch.setattr(service, "_call_with_retry", fake_call)
    result = service.generate_json("sys", "user")
    assert result.success is True
    assert result.data == {"queries": ["a"]}
    assert call_count["n"] == 2


def test_generate_json_gives_up_after_repair_also_fails(monkeypatch):
    service = LLMService(settings=make_settings())
    call_count = {"n": 0}

    def fake_call(system_prompt, user_prompt, max_tokens, json_mode):
        call_count["n"] += 1
        return FakeCompletion("still not json")

    monkeypatch.setattr(service, "_call_with_retry", fake_call)
    result = service.generate_json("sys", "user")
    assert result.success is False
    assert call_count["n"] == 2  # exactly 2 attempts, never more


def test_generate_json_repairs_on_fixable_api_error(monkeypatch):
    service = LLMService(settings=make_settings())
    call_count = {"n": 0}

    def fake_call(system_prompt, user_prompt, max_tokens, json_mode):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise LLMServiceError("Groq API error: Failed to validate JSON. Please adjust your prompt.")
        return FakeCompletion('{"ok": true}')

    monkeypatch.setattr(service, "_call_with_retry", fake_call)
    result = service.generate_json("sys", "user")
    assert result.success is True
    assert call_count["n"] == 2


def test_generate_json_does_not_repair_on_non_fixable_api_error(monkeypatch):
    service = LLMService(settings=make_settings())
    call_count = {"n": 0}

    def fake_call(system_prompt, user_prompt, max_tokens, json_mode):
        call_count["n"] += 1
        raise LLMServiceError("Groq API error: internal server error")

    monkeypatch.setattr(service, "_call_with_retry", fake_call)
    result = service.generate_json("sys", "user")
    assert result.success is False
    assert call_count["n"] == 1  # not a fixable-JSON error, so no repair attempt

def test_client_is_constructed_with_retries_disabled(monkeypatch):
    captured_kwargs = {}

    class FakeGroqClient:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setattr("services.llm_service.Groq", FakeGroqClient)

    service = LLMService(settings=make_settings())
    _ = service.client  # triggers lazy construction

    assert captured_kwargs.get("max_retries") == 0

def test_rate_limiter_does_not_delay_first_call(monkeypatch):
    from services.llm_service import _RateLimiter

    sleep_calls = []
    monkeypatch.setattr("services.llm_service.time.sleep", lambda s: sleep_calls.append(s))
    monkeypatch.setattr("services.llm_service.time.monotonic", lambda: 100.0)

    limiter = _RateLimiter(min_interval_seconds=20.0)
    limiter.wait()

    assert sleep_calls == []  # first call never waits


def test_rate_limiter_delays_second_call_within_interval(monkeypatch):
    from services.llm_service import _RateLimiter

    sleep_calls = []
    monkeypatch.setattr("services.llm_service.time.sleep", lambda s: sleep_calls.append(s))

    clock = {"t": 100.0}
    monkeypatch.setattr("services.llm_service.time.monotonic", lambda: clock["t"])

    limiter = _RateLimiter(min_interval_seconds=20.0)
    limiter.wait()  # first call, t=100, no wait

    clock["t"] = 105.0  # only 5s elapsed, need to wait 15 more
    limiter.wait()

    assert sleep_calls == [15.0]


def test_rate_limiter_does_not_delay_when_interval_already_elapsed(monkeypatch):
    from services.llm_service import _RateLimiter

    sleep_calls = []
    monkeypatch.setattr("services.llm_service.time.sleep", lambda s: sleep_calls.append(s))

    clock = {"t": 100.0}
    monkeypatch.setattr("services.llm_service.time.monotonic", lambda: clock["t"])

    limiter = _RateLimiter(min_interval_seconds=20.0)
    limiter.wait()

    clock["t"] = 125.0  # 25s elapsed, already past the 20s interval
    limiter.wait()

    assert sleep_calls == []


def test_rate_limiter_is_thread_safe_and_serializes_calls(monkeypatch):
    from services.llm_service import _RateLimiter

    monkeypatch.setattr("services.llm_service.time.sleep", lambda s: None)

    limiter = _RateLimiter(min_interval_seconds=0.01)
    call_order = []
    lock = threading.Lock()

    def worker(worker_id):
        limiter.wait()
        with lock:
            call_order.append(worker_id)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(call_order) == 5  # all completed without error/deadlock


def test_llm_service_applies_rate_limiter_before_each_call(monkeypatch):
    service = LLMService(settings=make_settings())
    wait_calls = {"n": 0}

    monkeypatch.setattr(service._rate_limiter, "wait", lambda: wait_calls.__setitem__("n", wait_calls["n"] + 1))

    fake_client = type("C", (), {
        "chat": type("Chat", (), {
            "completions": type("Comp", (), {"create": staticmethod(lambda **k: FakeCompletion('{"ok": true}'))})()
        })()
    })()
    service._client = fake_client

    result = service.generate_json("sys", "user")
    assert result.success is True
    assert wait_calls["n"] == 1  # exactly one throttle check for the one successful call