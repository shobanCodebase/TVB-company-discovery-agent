"""
Safe webpage fetching + readable-text extraction.

Never raises to the caller: every failure mode (timeout, HTTP error,
oversized page, unparseable HTML) results in a PageFetchResult with
success=False and a human-readable reason.

Extraction prefers <main>/<article> content when present, and strips a
wider set of boilerplate (nav, cookie banners, sidebars, iframes) than a
plain text dump would — this exists specifically to reduce the number of
tokens sent to the LLM in agents/research_agent.py without losing the
substantive facts a page contains.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.config import Settings, settings as default_settings

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; TVBLeadDiscoveryBot/1.0; "
    "+https://theventurebuild.com/)"
)

MAX_PAGE_BYTES = 3_000_000       # skip anything larger than ~3MB
MAX_EXTRACTED_CHARS = 20_000     # cap the text we carry forward per page
                                  # (research_agent applies a tighter,
                                  # configurable cap across ALL pages
                                  # combined before sending to the LLM)

RETRYABLE_EXCEPTIONS = (httpx.TimeoutException, httpx.TransportError)

# Tags whose contents are noise, not readable content
NOISE_TAGS = [
    "script", "style", "noscript", "svg", "header", "footer", "nav", "form",
    "aside", "iframe", "button",
]

# Class/id substrings commonly used for cookie-consent banners and similar
# boilerplate. Best-effort — matched case-insensitively against class/id.
NOISE_SELECTOR_MARKERS = (
    "cookie", "consent", "gdpr", "newsletter-signup", "subscribe-banner",
    "advert", "sponsor", "social-share", "share-buttons", "related-articles",
)

CONTENT_TAGS_PRIORITY = ["main", "article"]


class WebService:
    def __init__(self, settings: Settings | None = None, client: httpx.Client | None = None):
        self.settings = settings or default_settings
        self._client = client or httpx.Client(
            timeout=self.settings.request_timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": DEFAULT_USER_AGENT},
        )

    def close(self) -> None:
        self._client.close()

    def fetch_page(self, url: str) -> "PageFetchResult":
        if not url or not url.startswith(("http://", "https://")):
            return PageFetchResult(url=url, success=False, error="Invalid or missing URL scheme")

        try:
            resp = self._fetch_with_retry(url)
        except RETRYABLE_EXCEPTIONS as exc:
            logger.warning("Timeout/transport error fetching %s: %s", url, exc)
            return PageFetchResult(url=url, success=False, error=f"Network error: {exc}")
        except httpx.HTTPError as exc:
            logger.warning("HTTP error fetching %s: %s", url, exc)
            return PageFetchResult(url=url, success=False, error=f"HTTP error: {exc}")
        except Exception as exc:  # noqa: BLE001 - never crash the pipeline over one URL
            logger.exception("Unexpected error fetching %s", url)
            return PageFetchResult(url=url, success=False, error=f"Unexpected error: {exc}")

        if resp.status_code >= 400:
            return PageFetchResult(
                url=url, success=False, status_code=resp.status_code,
                final_url=str(resp.url), error=f"HTTP status {resp.status_code}",
            )

        content_length = len(resp.content)
        if content_length > MAX_PAGE_BYTES:
            return PageFetchResult(
                url=url, success=False, status_code=resp.status_code,
                final_url=str(resp.url),
                error=f"Page too large ({content_length} bytes), skipped",
            )

        content_type = resp.headers.get("content-type", "")
        if "html" not in content_type and content_type != "":
            return PageFetchResult(
                url=url, success=False, status_code=resp.status_code,
                final_url=str(resp.url),
                error=f"Unsupported content-type: {content_type}",
            )

        try:
            title, text = self._extract_readable_text(resp.text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to parse HTML for %s: %s", url, exc)
            return PageFetchResult(
                url=url, success=False, status_code=resp.status_code,
                final_url=str(resp.url), error=f"HTML parsing error: {exc}",
            )

        return PageFetchResult(
            url=url,
            success=True,
            status_code=resp.status_code,
            final_url=str(resp.url),
            title=title,
            text=text,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=6),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        reraise=True,
    )
    def _fetch_with_retry(self, url: str) -> httpx.Response:
        return self._client.get(url)

    @staticmethod
    def _looks_like_noise_element(tag) -> bool:
        try:
            class_attr = " ".join(tag.get("class", []) or [])
            id_attr = tag.get("id", "") or ""
        except Exception:  # noqa: BLE001
            return False
        combined = f"{class_attr} {id_attr}".lower()
        return any(marker in combined for marker in NOISE_SELECTOR_MARKERS)

    @classmethod
    def _extract_readable_text(cls, html: str) -> tuple[str, str]:
        soup = BeautifulSoup(html, "html.parser")

        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else None

        for tag_name in NOISE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # Best-effort removal of cookie banners / boilerplate widgets by
        # class/id, wherever they appear in the document.
        for tag in soup.find_all(True):
            if cls._looks_like_noise_element(tag):
                tag.decompose()

        # Prefer <main>/<article> content when present — this is usually
        # the substantive part of the page and excludes sidebars/menus
        # that survive plain tag-based stripping.
        content_root = None
        for tag_name in CONTENT_TAGS_PRIORITY:
            found = soup.find(tag_name)
            if found is not None:
                content_root = found
                break

        target = content_root if content_root is not None else soup
        text = target.get_text(separator=" ", strip=True)
        text = " ".join(text.split())

        if len(text) > MAX_EXTRACTED_CHARS:
            text = text[:MAX_EXTRACTED_CHARS]

        return title, text

    def fetch_many(self, urls: list[str]) -> list["PageFetchResult"]:
        return [self.fetch_page(u) for u in urls]


@dataclass
class PageFetchResult:
    url: str
    success: bool
    status_code: int | None = None
    final_url: str | None = None
    title: str | None = None
    text: str = ""
    error: str | None = None