import httpx
import pytest

from core.config import Settings
from services.search_service import SearchService, SearchServiceError


def make_settings(**overrides) -> Settings:
    defaults = dict(search_provider="tavily", search_api_key="fake-key", request_timeout_seconds=5.0)
    defaults.update(overrides)
    return Settings(**defaults)


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text
        self.url = "https://api.tavily.com/search"

    def json(self):
        return self._json_data


def test_search_without_api_key_returns_failure():
    settings = make_settings(search_api_key="")
    service = SearchService(settings=settings)
    result = service.search("test query")
    assert result.success is False
    assert "SEARCH_API_KEY" in result.error


def test_search_unsupported_provider_returns_failure():
    settings = make_settings(search_provider="bing")
    service = SearchService(settings=settings)
    result = service.search("test query")
    assert result.success is False
    assert "Unsupported search provider" in result.error


def test_search_parses_valid_tavily_response(monkeypatch):
    settings = make_settings()
    service = SearchService(settings=settings)

    fake_json = {
        "results": [
            {"title": "Acme raises $2M", "url": "https://acme.io/news", "content": "Acme raised...", "score": 0.9},
            {"title": "Beta Corp", "url": "https://beta.com", "content": "Beta info", "score": 0.8},
        ]
    }
    monkeypatch.setattr(service._client, "post", lambda *a, **k: FakeResponse(200, fake_json))

    result = service.search("acme funding")
    assert result.success is True
    assert len(result.results) == 2
    assert result.results[0].url == "https://acme.io/news"


def test_search_skips_malformed_result_items(monkeypatch):
    settings = make_settings()
    service = SearchService(settings=settings)

    fake_json = {"results": [{"title": "Ok Co", "url": "https://ok.com"}, {"weird": "no url or title"}]}
    monkeypatch.setattr(service._client, "post", lambda *a, **k: FakeResponse(200, fake_json))

    result = service.search("query")
    assert result.success is True
    assert len(result.results) == 1
    assert result.results[0].url == "https://ok.com"


def test_search_handles_rate_limit(monkeypatch):
    settings = make_settings()
    service = SearchService(settings=settings)
    monkeypatch.setattr(service._client, "post", lambda *a, **k: FakeResponse(429, {}, "rate limited"))

    result = service.search("query")
    assert result.success is False
    assert "429" in result.error or "rate limit" in result.error.lower()


def test_search_handles_invalid_json(monkeypatch):
    settings = make_settings()
    service = SearchService(settings=settings)

    class BadJsonResponse(FakeResponse):
        def json(self):
            raise ValueError("bad json")

    monkeypatch.setattr(service._client, "post", lambda *a, **k: BadJsonResponse(200))
    result = service.search("query")
    assert result.success is False
    assert "invalid JSON" in result.error


def test_search_many_runs_all_queries_independently(monkeypatch):
    settings = make_settings()
    service = SearchService(settings=settings)
    monkeypatch.setattr(service._client, "post", lambda *a, **k: FakeResponse(200, {"results": []}))

    responses = service.search_many(["q1", "q2", "q3"])
    assert len(responses) == 3
    assert all(r.success for r in responses)