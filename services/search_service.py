"""
Search provider abstraction.

Currently supports Tavily. Adding a new provider means adding a new
`_search_<provider>` method and registering it in `_PROVIDERS` — no
changes needed elsewhere in the pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.config import Settings, settings as default_settings

logger = logging.getLogger(__name__)


class SearchServiceError(Exception):
    """Raised when a search provider call fails after retries."""


@dataclass
class SearchResultItem:
    title: str
    url: str
    snippet: str = ""
    raw_score: float | None = None


@dataclass
class SearchResponse:
    query: str
    results: list[SearchResultItem] = field(default_factory=list)
    provider: str = ""
    success: bool = True
    error: str | None = None


RETRYABLE_EXCEPTIONS = (httpx.TimeoutException, httpx.TransportError)


class SearchService:
    def __init__(self, settings: Settings | None = None, client: httpx.Client | None = None):
        self.settings = settings or default_settings
        self._client = client or httpx.Client(timeout=self.settings.request_timeout_seconds)
        self._provider = self.settings.search_provider.lower().strip()

    def close(self) -> None:
        self._client.close()

    def search(self, query: str, max_results: int = 8) -> SearchResponse:
        """
        Executes a single search query against the configured provider.
        Never raises — failures come back as SearchResponse(success=False).
        """
        if not self.settings.search_api_key:
            logger.warning("SEARCH_API_KEY not configured; returning empty search response.")
            return SearchResponse(query=query, provider=self._provider, success=False,
                                   error="SEARCH_API_KEY not configured")

        try:
            if self._provider == "tavily":
                return self._search_tavily(query, max_results)
            raise SearchServiceError(f"Unsupported search provider: {self._provider}")
        except SearchServiceError as exc:
            logger.error("Search failed for query %r: %s", query, exc)
            return SearchResponse(query=query, provider=self._provider, success=False, error=str(exc))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        reraise=True,
    )
    def _search_tavily(self, query: str, max_results: int) -> SearchResponse:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self.settings.search_api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        }
        try:
            resp = self._client.post(url, json=payload)
        except RETRYABLE_EXCEPTIONS:
            raise
        except httpx.HTTPError as exc:
            raise SearchServiceError(f"Tavily request error: {exc}") from exc

        if resp.status_code == 429:
            raise SearchServiceError("Tavily rate limit hit (429)")
        if resp.status_code >= 500:
            # Treat as transient — let tenacity retry via a TransportError-like path
            raise httpx.TransportError(f"Tavily server error {resp.status_code}")
        if resp.status_code != 200:
            raise SearchServiceError(f"Tavily returned status {resp.status_code}: {resp.text[:200]}")

        try:
            data = resp.json()
        except ValueError as exc:
            raise SearchServiceError(f"Tavily returned invalid JSON: {exc}") from exc

        raw_results = data.get("results", [])
        items = []
        for r in raw_results:
            try:
                items.append(
                    SearchResultItem(
                        title=r.get("title", "") or "",
                        url=r.get("url", "") or "",
                        snippet=r.get("content", "") or "",
                        raw_score=r.get("score"),
                    )
                )
            except Exception:  # noqa: BLE001 - one malformed item shouldn't drop the batch
                logger.warning("Skipping malformed Tavily result item: %r", r)
                continue

        # Drop items with no usable URL
        items = [i for i in items if i.url]

        return SearchResponse(query=query, results=items, provider="tavily", success=True)

    def search_many(self, queries: list[str], max_results_per_query: int = 8) -> list[SearchResponse]:
        """Runs multiple queries sequentially, tolerating per-query failures."""
        responses = []
        for q in queries:
            responses.append(self.search(q, max_results=max_results_per_query))
        return responses