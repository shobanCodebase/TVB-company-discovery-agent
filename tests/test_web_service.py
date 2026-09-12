import httpx
import pytest

from core.config import Settings
from services.web_service import MAX_PAGE_BYTES, WebService


def make_settings() -> Settings:
    return Settings(request_timeout_seconds=5.0)


class FakeResponse:
    def __init__(self, status_code=200, text="", content=b"", headers=None, url="https://example.com"):
        self.status_code = status_code
        self.text = text
        self.content = content or text.encode()
        self.headers = headers or {"content-type": "text/html"}
        self.url = url


def test_fetch_page_rejects_invalid_url():
    service = WebService(settings=make_settings())
    result = service.fetch_page("not-a-url")
    assert result.success is False
    assert "Invalid" in result.error


def test_fetch_page_extracts_readable_text(monkeypatch):
    service = WebService(settings=make_settings())
    html = """
    <html><head><title>Acme Inc</title><script>bad()</script></head>
    <body><nav>menu</nav><h1>Acme</h1><p>We raised $2.5M in seed funding.</p></body></html>
    """
    monkeypatch.setattr(service, "_fetch_with_retry", lambda url: FakeResponse(200, html))

    result = service.fetch_page("https://acme.io")
    assert result.success is True
    assert result.title == "Acme Inc"
    assert "raised $2.5M" in result.text
    assert "menu" not in result.text  # nav stripped


def test_fetch_page_handles_http_error_status(monkeypatch):
    service = WebService(settings=make_settings())
    monkeypatch.setattr(service, "_fetch_with_retry", lambda url: FakeResponse(404, "Not Found"))

    result = service.fetch_page("https://acme.io/missing")
    assert result.success is False
    assert result.status_code == 404


def test_fetch_page_rejects_oversized_page(monkeypatch):
    service = WebService(settings=make_settings())
    big_content = b"x" * (MAX_PAGE_BYTES + 1)
    monkeypatch.setattr(
        service, "_fetch_with_retry",
        lambda url: FakeResponse(200, "big", content=big_content),
    )

    result = service.fetch_page("https://acme.io/huge")
    assert result.success is False
    assert "too large" in result.error.lower()


def test_fetch_page_rejects_non_html_content_type(monkeypatch):
    service = WebService(settings=make_settings())
    monkeypatch.setattr(
        service, "_fetch_with_retry",
        lambda url: FakeResponse(200, "PDF binary", headers={"content-type": "application/pdf"}),
    )

    result = service.fetch_page("https://acme.io/file.pdf")
    assert result.success is False
    assert "content-type" in result.error.lower()


def test_fetch_page_handles_network_error(monkeypatch):
    service = WebService(settings=make_settings())

    def raise_timeout(url):
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr(service, "_fetch_with_retry", raise_timeout)

    result = service.fetch_page("https://slow.example.com")
    assert result.success is False
    assert "Network error" in result.error


def test_fetch_page_handles_unexpected_exception(monkeypatch):
    service = WebService(settings=make_settings())

    def raise_weird(url):
        raise RuntimeError("boom")

    monkeypatch.setattr(service, "_fetch_with_retry", raise_weird)

    result = service.fetch_page("https://weird.example.com")
    assert result.success is False
    assert "Unexpected error" in result.error


def test_fetch_many_handles_mixed_success_and_failure(monkeypatch):
    service = WebService(settings=make_settings())

    def fake_fetch(url):
        if "good" in url:
            return service.fetch_page.__wrapped__(service, url) if False else None
        return None

    # Simpler: patch fetch_page directly for this integration-style test
    from services.web_service import PageFetchResult

    def fake_fetch_page(url):
        if "good" in url:
            return PageFetchResult(url=url, success=True, text="ok")
        return PageFetchResult(url=url, success=False, error="failed")

    monkeypatch.setattr(service, "fetch_page", fake_fetch_page)
    results = service.fetch_many(["https://good.com", "https://bad.com"])
    assert results[0].success is True
    assert results[1].success is False