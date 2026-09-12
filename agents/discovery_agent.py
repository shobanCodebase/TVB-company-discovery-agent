"""
Stage 1 — Dynamic discovery.

Pipeline:
    generate search queries (template-based + optional LLM-augmented)
        -> execute searches
        -> extract CompanyCandidate objects
        -> normalize domains
        -> deduplicate
        -> return CompanyCandidate list

IMPORTANT: this stage never asserts anything about funding, revenue,
US presence, tech platform, or contacts. It only proposes candidates
for Stage 2 (research_agent, Phase 3) to investigate.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

from core.config import Settings, settings as default_settings
from core.deduplication import deduplicate
from core.models import CompanyCandidate
from services.llm_service import LLMService
from services.search_service import SearchResponse, SearchResultItem, SearchService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Query generation
# ---------------------------------------------------------------------------

_EXCLUDE = "-site:wikipedia.org -site:youtube.com -site:reddit.com -recipe -site:*.gov -site:*.edu"

QUERY_TEMPLATES = [
    f'"{{topic}}" startup raises "$1 million" OR "$2 million" OR "$3 million" OR "$5 million" {_EXCLUDE}',
    f'"{{topic}}" startup seed funding round 2024 2025 {_EXCLUDE}',
    f'"{{topic}}" SaaS platform funding announcement {_EXCLUDE}',
    f'"{{topic}}" technology company revenue million dollars {_EXCLUDE}',
    f'emerging market "{{topic}}" technology startup funding round {_EXCLUDE}',
    f'non-US "{{topic}}" startup CEO founder seed round {_EXCLUDE}',
]

QUERY_TOPICS = [
    "technology",
    "SaaS",
    "fintech",
    "AI platform",
    "developer platform",
    "healthtech",
    "edtech",
    "logistics tech",
    "e-commerce platform",
    "cybersecurity platform",
]

# Blocklisted domains that are never useful as "companies" (aggregators,
# social platforms, generic news homepages, etc.)
DOMAIN_BLOCKLIST = {
    "linkedin.com", "twitter.com", "x.com", "facebook.com", "instagram.com",
    "youtube.com", "wikipedia.org", "crunchbase.com", "medium.com",
    "reddit.com", "github.com", "techcrunch.com", "forbes.com",
    "bloomberg.com", "businessinsider.com", "google.com",
}


def generate_all_template_queries() -> list[str]:
    """Full deterministic template x topic product, uncapped and unrotated."""
    queries = []
    for template in QUERY_TEMPLATES:
        for topic in QUERY_TOPICS:
            queries.append(template.format(topic=topic))
    return queries


def generate_template_queries(max_queries: int, offset: int = 0) -> list[str]:
    """
    Deterministic query generation — works with zero LLM calls.

    `offset` lets callers (e.g. multiple discovery rounds in the pipeline)
    rotate through different windows of the full template x topic product
    instead of always getting the same first N queries. Once `offset`
    would run past the end of the pool, it WRAPS AROUND rather than
    returning an empty list — a later round re-issuing an earlier round's
    query still surfaces fresh live search results (rankings/results
    change over time), and re-running it is far better than discovery
    silently going empty and ending the pipeline early. Candidate-level
    deduplication in core.pipeline already protects against literal
    duplicate companies across rounds.
    """
    all_queries = generate_all_template_queries()
    if not all_queries:
        return []
    pool_size = len(all_queries)
    wrapped_offset = offset % pool_size
    # Build the window by wrapping, e.g. offset=75, pool_size=60,
    # wrapped_offset=15 -> take queries[15:15+max_queries], wrapping again
    # if that window itself runs past the end.
    end = wrapped_offset + max_queries
    if end <= pool_size:
        return all_queries[wrapped_offset:end]
    return all_queries[wrapped_offset:] + all_queries[:end - pool_size]


def generate_llm_queries(llm_service: LLMService, count: int = 5) -> list[str]:
    """
    Best-effort extra queries from the LLM for angle diversity. Any
    failure (no API key, parse error, API error) just yields an empty
    list — discovery must not depend on this succeeding.
    """
    if getattr(llm_service, "quota_exceeded", False):
        logger.info("LLM quota already exhausted — skipping LLM-generated discovery queries")
        return []

    system_prompt = (
        "You generate concise web search queries for finding technology "
        "startups that raised between $1M and $5M in funding or revenue, "
        "are not primarily US-based, and operate a tech platform. "
        "Return ONLY a JSON object: {\"queries\": [\"...\", \"...\"]}. "
        f"Return exactly {count} queries, each under 12 words, each "
        "targeting a different region, industry, or funding-stage angle. "
        "Do not include any explanation."
    )
    user_prompt = f"Generate {count} diverse search queries now."

    result = llm_service.generate_json(system_prompt, user_prompt, max_tokens=llm_service.settings.llm_query_gen_max_tokens)
    if not result.success or not isinstance(result.data, dict):
        logger.info("LLM query generation unavailable/failed: %s", result.error)
        return []

    queries = result.data.get("queries")
    if not isinstance(queries, list):
        return []

    cleaned = [str(q).strip() for q in queries if isinstance(q, (str,)) and str(q).strip()]
    return cleaned[:count]


# ---------------------------------------------------------------------------
# Candidate extraction
# ---------------------------------------------------------------------------

def extract_domain(url: str) -> str | None:
    try:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        host = host.replace("www.", "")
        return host or None
    except Exception:  # noqa: BLE001
        return None


def is_blocklisted_domain(domain: str | None) -> bool:
    if not domain:
        return False
    return any(domain == d or domain.endswith(f".{d}") for d in DOMAIN_BLOCKLIST)


def guess_company_name(item: SearchResultItem, domain: str | None) -> str:
    """
    Best-effort company name from the search result title, falling back
    to a title-cased domain. This is a *candidate* name only — Stage 2
    research should confirm/refine it against the company's own site.
    """
    title = item.title.strip()
    if title:
        # Strip common suffixes like " - Home", " | Official Site", etc.
        # Requires whitespace on BOTH sides of the separator so mid-word
        # hyphens (e-commerce, co-founder, Ride-hailing, pre-seed) are
        # never split — those have no surrounding spaces, only a real
        # " - " / " | " style suffix separator does.
        title = re.split(r"\s+[\|\-–—]\s+", title)[0].strip()
        if title:
            return title
    if domain:
        base = domain.split(".")[0]
        return base.replace("-", " ").title()
    return "Unknown Company"


def candidate_from_result(item: SearchResultItem, query: str) -> CompanyCandidate | None:
    domain = extract_domain(item.url)
    if is_blocklisted_domain(domain):
        return None
    if not item.url:
        return None

    return CompanyCandidate(
        company_name=guess_company_name(item, domain),
        domain=domain,
        source_url=item.url,
        search_snippet=item.snippet or None,
        discovery_query=query,
    )


def candidates_from_response(response: SearchResponse, max_per_query: int) -> list[CompanyCandidate]:
    if not response.success:
        logger.info("Search failed for query %r: %s", response.query, response.error)
        return []

    candidates = []
    for item in response.results[:max_per_query]:
        candidate = candidate_from_result(item, response.query)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


# ---------------------------------------------------------------------------
# Discovery agent
# ---------------------------------------------------------------------------

class DiscoveryAgent:
    def __init__(
        self,
        search_service: SearchService | None = None,
        llm_service: LLMService | None = None,
        settings: Settings | None = None,
    ):
        self.settings = settings or default_settings
        self.search_service = search_service or SearchService(self.settings)
        self.llm_service = llm_service or LLMService(self.settings)

    def build_queries(self, offset: int = 0) -> list[str]:
        max_queries = self.settings.max_discovery_queries
        base_queries = generate_template_queries(max_queries, offset=offset)

        remaining = max_queries - len(base_queries)
        if remaining > 0:
            llm_queries = generate_llm_queries(self.llm_service, count=min(remaining, 8))
            base_queries.extend(llm_queries)

        # de-dupe queries themselves (case-insensitive), preserve order
        seen = set()
        unique_queries = []
        for q in base_queries:
            key = q.strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique_queries.append(q)

        return unique_queries[:max_queries]

    def discover(self, offset: int = 0) -> list[CompanyCandidate]:
        queries = self.build_queries(offset=offset)
        logger.info("Discovery running with %d queries", len(queries))

        all_candidates: list[CompanyCandidate] = []
        for query in queries:
            if len(all_candidates) >= self.settings.max_candidates_total:
                logger.info("Reached MAX_CANDIDATES_TOTAL, stopping query loop early")
                break

            response = self.search_service.search(
                query, max_results=self.settings.max_candidates_per_query
            )
            candidates = candidates_from_response(response, self.settings.max_candidates_per_query)
            all_candidates.extend(candidates)

        all_candidates = all_candidates[: self.settings.max_candidates_total]

        deduped = deduplicate(
            all_candidates,
            get_company_name=lambda c: c.company_name,
            get_domain=lambda c: c.domain,
        )

        logger.info(
            "Discovery complete: %d raw candidates -> %d after dedup",
            len(all_candidates), len(deduped),
        )
        return deduped