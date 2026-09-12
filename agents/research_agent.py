"""
Stage 2 — Deep research + Stage 3 — deterministic qualification wiring.

Token/quota-aware changes vs. earlier phases:
  - Combined source text sent to the LLM is capped by
    settings.research_max_source_chars (configurable, default 6000 —
    down from an earlier hardcoded 12000) instead of a fixed constant.
  - max_tokens for the extraction call comes from
    settings.llm_extraction_max_tokens (configurable, default 700).
  - A domain-level in-run cache avoids ever sending the same domain's
    text to the LLM twice within one ResearchAgent instance's lifetime.
  - If the LLM's daily quota has already been exhausted (llm_service.
    quota_exceeded), research SKIPS gathering source pages entirely for
    remaining candidates (no wasted web/search calls) and returns a
    CompanyResearch with everything unknown, which qualify() correctly
    rejects — the SAME outcome as an extraction failure, just without
    burning further web-fetch/search budget on doomed candidates.

Nothing here invents data: any field the LLM can't support with
evidence comes back as None, and validators.py treats that as
UNKNOWN -> reject where the field is mandatory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from core.config import Settings, settings as default_settings
from core.models import (
    CompanyCandidate,
    CompanyResearch,
    Evidence,
    QualificationStatus,
    USPresence,
)
from core.scoring import score_company
from core.validators import evidence_is_plausible, qualify
from services.llm_service import LLMService
from services.search_service import SearchService
from services.web_service import PageFetchResult, WebService

logger = logging.getLogger(__name__)


@dataclass
class SourcePage:
    url: str
    text: str


def gather_source_pages(
    candidate: CompanyCandidate,
    web_service: WebService,
    search_service: SearchService,
    max_extra_searches: int = 1,
    max_results_per_search: int = 2,
) -> list[SourcePage]:
    """
    max_extra_searches defaults to 1 (was 2) to cut Tavily request volume
    roughly in half per candidate. The single targeted query is written
    to cover every missing research dimension at once — funding/revenue,
    tech platform/product, US presence, and CEO/co-founder — rather than
    dropping one dimension outright. It does not add any extra LLM call;
    extraction remains one combined call per candidate as before, and
    every fact pulled from this search still goes through the same
    evidence-plausibility check in _build_research before it's trusted.
    """
    pages: list[SourcePage] = []
    seen_urls: set[str] = set()

    def add_page(result: PageFetchResult) -> None:
        if result.success and result.text and result.url not in seen_urls:
            pages.append(SourcePage(url=result.final_url or result.url, text=result.text))
            seen_urls.add(result.url)

    add_page(web_service.fetch_page(candidate.source_url))

    if candidate.domain:
        homepage_url = f"https://{candidate.domain}"
        if homepage_url != candidate.source_url:
            add_page(web_service.fetch_page(homepage_url))

    targeted_queries = [
        f'"{candidate.company_name}" funding OR revenue OR raised OR platform '
        f'OR CEO OR co-founder OR founder OR "United States" OR US',
    ][:max_extra_searches]

    targeted_searches_run = 0
    for query in targeted_queries:
        targeted_searches_run += 1
        try:
            response = search_service.search(query, max_results=max_results_per_search)
        except Exception:  # noqa: BLE001 - a search failure must not crash this candidate
            logger.warning("Targeted search raised unexpectedly for %s, continuing with pages gathered so far", candidate.company_name)
            continue

        if not response.success:
            logger.info(
                "Targeted search failed for %s (%s) — continuing with pages already gathered",
                candidate.company_name, response.error,
            )
            continue

        for item in response.results[:max_results_per_search]:
            if item.url in seen_urls:
                continue
            add_page(web_service.fetch_page(item.url))

    logger.info("targeted_searches=%d for %s", targeted_searches_run, candidate.company_name)

    return pages


def build_combined_text(pages: list[SourcePage], max_chars: int) -> str:
    """Concatenates page texts with a URL header per section, capped in length."""
    parts = []
    total = 0
    for page in pages:
        header = f"\n\n===== SOURCE: {page.url} =====\n"
        chunk = header + page.text
        if total + len(chunk) > max_chars:
            remaining = max_chars - total
            if remaining <= len(header):
                break
            chunk = chunk[:remaining]
        parts.append(chunk)
        total += len(chunk)
        if total >= max_chars:
            break
    return "".join(parts)


EXTRACTION_SYSTEM_PROMPT = """You are a careful research analyst extracting facts about a company \
from provided web page excerpts. You must NEVER guess or infer facts that are not explicitly \
stated in the provided text. If information is not present, use null. Be concise — keep every \
"claim" and "excerpt" string short (excerpt under 120 characters, claim under 100 characters) so \
the full JSON response stays well within the output token budget.

For every non-null factual field, you MUST provide supporting evidence: a SHORT excerpt (under 120 \
characters) from the provided source text that supports the claim, and which SOURCE URL (copied \
exactly from the "===== SOURCE: ... =====" headers) it came from.

Return ONLY a JSON object with this exact shape:
{
  "description": string or null,
  "industry": string or null,
  "funding_amount_usd": number or null,
  "funding_evidence": {"claim": string, "source_url": string, "excerpt": string} or null,
  "revenue_amount_usd": number or null,
  "revenue_evidence": {"claim": string, "source_url": string, "excerpt": string} or null,
  "tech_platform": true or false or null,
  "tech_platform_evidence": {"claim": string, "source_url": string, "excerpt": string} or null,
  "us_presence": "NONE" or "LOW" or "MODERATE" or "HIGH" or "UNKNOWN",
  "us_presence_evidence": {"claim": string, "source_url": string, "excerpt": string} or null,
  "headquarters_country": string or null,
  "headquarters_evidence": {"claim": string, "source_url": string, "excerpt": string} or null,
  "contact_name": string or null,
  "contact_role": string or null,
  "contact_evidence": {"claim": string, "source_url": string, "excerpt": string} or null
}

Rules:
- funding_amount_usd / revenue_amount_usd must be plain numbers in USD (convert if another currency \
is mentioned and state the conversion in the claim), or null if no dollar figure is stated.
- tech_platform is true only if the text describes a software/technology platform, app, or SaaS \
product the company operates — not just "uses technology internally".
- us_presence: NONE = no mention of any US office/team/customers-focus; LOW = minor US presence \
(e.g. a couple of remote US employees, or lists US as one of many markets); MODERATE = a real US \
office or US go-to-market push; HIGH = headquartered in the US or clearly US-focused. If the text \
does not discuss this at all, use "UNKNOWN".
- contact_name must be a real full person name explicitly identified as CEO, founder, or co-founder \
in the text. Do not extract generic references like "the team" or "our founders" as a name.
- Every excerpt must be copied verbatim (or near-verbatim) from the provided source text, not \
paraphrased or invented.
- Keep ALL string values SHORT and concise. Do not write full sentences where a few words suffice.
- Output ONLY the JSON object, no other text, no markdown fences.
"""


def _evidence_from_dict(data: dict | None) -> Evidence | None:
    if not data or not isinstance(data, dict):
        return None
    claim = data.get("claim")
    source_url = data.get("source_url")
    excerpt = data.get("excerpt")
    if not claim or not source_url:
        return None
    try:
        return Evidence(claim=claim, source_url=source_url, excerpt=excerpt, source_type="web")
    except Exception:  # noqa: BLE001
        return None


def extract_research_fields(
    llm_service: LLMService,
    company_name: str,
    combined_text: str,
    max_tokens: int,
) -> dict:
    """
    Returns a dict of raw fields, or {} on any failure (no source text,
    LLM quota exhausted, API error, unparseable output). Callers treat
    {} exactly like "everything unknown".
    """
    if not combined_text.strip():
        logger.info("No source text available for %s; skipping LLM extraction", company_name)
        return {}

    user_prompt = (
        f"Company being researched: {company_name}\n\n"
        f"Source text follows:\n{combined_text}"
    )
    result = llm_service.generate_json(EXTRACTION_SYSTEM_PROMPT, user_prompt, max_tokens=max_tokens)
    if not result.success or not isinstance(result.data, dict):
        logger.info("Extraction failed for %s: %s", company_name, result.error)
        return {}
    return result.data


class ResearchAgent:
    def __init__(
        self,
        web_service: WebService | None = None,
        search_service: SearchService | None = None,
        llm_service: LLMService | None = None,
        settings: Settings | None = None,
    ):
        self.settings = settings or default_settings
        self.web_service = web_service or WebService(self.settings)
        self.search_service = search_service or SearchService(self.settings)
        self.llm_service = llm_service or LLMService(self.settings)
        # Domain -> CompanyResearch cache, scoped to this agent instance
        # (i.e. one run). Prevents ever researching the same normalized
        # domain twice even if it slips past discovery-level dedup.
        self._domain_cache: dict[str, CompanyResearch] = {}

    def research_candidate(self, candidate: CompanyCandidate) -> CompanyResearch:
        cache_key = (candidate.domain or candidate.source_url).strip().lower()
        if self.settings.enable_cache and cache_key in self._domain_cache:
            logger.info("Domain %s already researched this run — reusing cached result", cache_key)
            return self._domain_cache[cache_key]

        if self.llm_service.quota_exceeded:
            logger.info(
                "LLM quota already exhausted — skipping page gathering for %s to save "
                "web/search budget; fields will be UNKNOWN and correctly rejected.",
                candidate.company_name,
            )
            raw: dict = {}
            source_texts_by_url: dict[str, str] = {}
        else:
            pages = gather_source_pages(candidate, self.web_service, self.search_service)
            combined_text = build_combined_text(pages, max_chars=self.settings.research_max_source_chars)
            source_texts_by_url = {p.url: p.text for p in pages}
            raw = extract_research_fields(
                self.llm_service,
                candidate.company_name,
                combined_text,
                max_tokens=self.settings.llm_extraction_max_tokens,
            )

        research = self._build_research(candidate, raw, source_texts_by_url)

        passed, reasons = qualify(
            research,
            min_amount=self.settings.min_amount_usd,
            max_amount=self.settings.max_amount_usd,
        )
        research.rejection_reasons = reasons
        research.status = QualificationStatus.QUALIFIED if passed else QualificationStatus.REJECTED
        research.score = score_company(research)

        if self.settings.enable_cache:
            self._domain_cache[cache_key] = research

        return research

    def research_many(self, candidates: list[CompanyCandidate]) -> list[CompanyResearch]:
        logger.info("Research candidate cap: %d", self.settings.max_candidates_for_research)
        results: list[CompanyResearch] = []
        for i, candidate in enumerate(candidates):
            if i >= self.settings.max_research_iterations:
                logger.info("Reached MAX_RESEARCH_ITERATIONS, stopping research loop early")
                break
            try:
                results.append(self.research_candidate(candidate))
            except Exception:  # noqa: BLE001
                logger.exception("Research failed unexpectedly for %s", candidate.company_name)
                continue
        return results

    def _build_research(
        self,
        candidate: CompanyCandidate,
        raw: dict,
        source_texts_by_url: dict[str, str],
    ) -> CompanyResearch:
        domain = candidate.domain or candidate.source_url

        def get_evidence(key: str) -> Evidence | None:
            evidence = _evidence_from_dict(raw.get(key))
            if evidence is None:
                return None
            source_text = source_texts_by_url.get(evidence.source_url)
            if source_text is not None and not evidence_is_plausible(evidence, source_text):
                logger.info(
                    "Rejecting unsupported evidence for %s field %s: excerpt not found in source",
                    candidate.company_name, key,
                )
                return None
            return evidence

        funding_evidence = get_evidence("funding_evidence")
        revenue_evidence = get_evidence("revenue_evidence")
        tech_platform_evidence = get_evidence("tech_platform_evidence")
        us_presence_evidence = get_evidence("us_presence_evidence")
        headquarters_evidence = get_evidence("headquarters_evidence")
        contact_evidence = get_evidence("contact_evidence")

        us_presence_raw = raw.get("us_presence")
        try:
            us_presence = USPresence(us_presence_raw) if us_presence_raw else USPresence.UNKNOWN
        except ValueError:
            us_presence = USPresence.UNKNOWN

        return CompanyResearch(
            company_name=candidate.company_name,
            domain=domain,
            description=raw.get("description"),
            industry=raw.get("industry"),
            funding_amount_usd=_safe_float(raw.get("funding_amount_usd")) if funding_evidence else None,
            funding_evidence=funding_evidence,
            revenue_amount_usd=_safe_float(raw.get("revenue_amount_usd")) if revenue_evidence else None,
            revenue_evidence=revenue_evidence,
            tech_platform=raw.get("tech_platform") if tech_platform_evidence else None,
            tech_platform_evidence=tech_platform_evidence,
            us_presence=(us_presence if us_presence_evidence else USPresence.UNKNOWN),
            us_presence_evidence=us_presence_evidence,
            headquarters_country=raw.get("headquarters_country") if headquarters_evidence else None,
            headquarters_evidence=headquarters_evidence,
            contact_name=raw.get("contact_name") if contact_evidence else None,
            contact_role=raw.get("contact_role") if contact_evidence else None,
            contact_evidence=contact_evidence,
        )


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None