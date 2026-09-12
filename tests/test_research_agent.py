from unittest.mock import MagicMock

import pytest

from agents.research_agent import (
    ResearchAgent,
    build_combined_text,
    extract_research_fields,
    gather_source_pages,
)
from core.config import Settings
from core.models import CompanyCandidate, QualificationStatus, RejectionReason, USPresence
from services.llm_service import LLMJSONResult
from services.search_service import SearchResponse, SearchResultItem
from services.web_service import PageFetchResult


def make_candidate(**overrides) -> CompanyCandidate:
    defaults = dict(
        company_name="Acme SaaS",
        domain="acme.io",
        source_url="https://acme.io/news",
    )
    defaults.update(overrides)
    return CompanyCandidate(**defaults)


# --- gather_source_pages ---

def test_gather_source_pages_dedupes_urls_and_skips_failures():
    candidate = make_candidate()
    web = MagicMock()
    search = MagicMock()

    def fake_fetch(url):
        if "acme.io" in url:
            return PageFetchResult(url=url, success=True, final_url=url, text=f"content for {url}")
        return PageFetchResult(url=url, success=False, error="fail")

    web.fetch_page.side_effect = fake_fetch
    search.search.return_value = SearchResponse(query="q", success=False, error="no results")

    pages = gather_source_pages(candidate, web, search)
    urls = {p.url for p in pages}
    assert "https://acme.io/news" in urls or "https://acme.io" in urls
    assert len(pages) <= 2  # source_url and homepage are the same domain, no search hits


def test_gather_source_pages_includes_search_results():
    candidate = make_candidate()
    web = MagicMock()
    search = MagicMock()

    web.fetch_page.side_effect = lambda url: PageFetchResult(
        url=url, success=True, final_url=url, text=f"text for {url}"
    )
    search.search.return_value = SearchResponse(
        query="q", success=True,
        results=[SearchResultItem(title="News", url="https://news.example.com/acme-funding")],
    )

    pages = gather_source_pages(candidate, web, search)
    urls = {p.url for p in pages}
    assert "https://news.example.com/acme-funding" in urls


def test_gather_source_pages_performs_at_most_one_targeted_search():
    candidate = make_candidate()
    web = MagicMock()
    search = MagicMock()

    web.fetch_page.side_effect = lambda url: PageFetchResult(
        url=url, success=True, final_url=url, text=f"text for {url}"
    )
    search.search.return_value = SearchResponse(query="q", success=True, results=[])

    gather_source_pages(candidate, web, search)

    assert search.search.call_count == 1  # exactly one targeted search, not two


def test_gather_source_pages_preserves_source_and_homepage_pages_regardless_of_search():
    candidate = make_candidate()
    web = MagicMock()
    search = MagicMock()

    web.fetch_page.side_effect = lambda url: PageFetchResult(
        url=url, success=True, final_url=url, text=f"text for {url}"
    )
    search.search.return_value = SearchResponse(query="q", success=False, error="boom")

    pages = gather_source_pages(candidate, web, search)
    urls = {p.url for p in pages}

    # source_url and homepage must still be present even though the search failed
    assert candidate.source_url in urls or f"https://{candidate.domain}" in urls
    assert len(pages) >= 1


def test_gather_source_pages_tolerates_failed_targeted_search():
    candidate = make_candidate()
    web = MagicMock()
    search = MagicMock()

    web.fetch_page.side_effect = lambda url: PageFetchResult(
        url=url, success=True, final_url=url, text=f"text for {url}"
    )
    search.search.return_value = SearchResponse(query="q", success=False, error="Tavily returned status 432")

    # Should not raise
    pages = gather_source_pages(candidate, web, search)
    assert isinstance(pages, list)


def test_gather_source_pages_tolerates_search_raising_exception():
    candidate = make_candidate()
    web = MagicMock()
    search = MagicMock()

    web.fetch_page.side_effect = lambda url: PageFetchResult(
        url=url, success=True, final_url=url, text=f"text for {url}"
    )
    search.search.side_effect = RuntimeError("unexpected network failure")

    # Should not raise — falls back to whatever pages were already gathered
    pages = gather_source_pages(candidate, web, search)
    assert isinstance(pages, list)
    assert len(pages) >= 1  # source_url/homepage pages still present


# --- build_combined_text ---

def test_build_combined_text_includes_source_headers():
    from agents.research_agent import SourcePage
    pages = [SourcePage(url="https://a.com", text="hello world")]
    combined = build_combined_text(pages, max_chars=6000)
    assert "SOURCE: https://a.com" in combined
    assert "hello world" in combined


def test_build_combined_text_caps_length():
    from agents.research_agent import SourcePage
    cap = 6000
    pages = [SourcePage(url="https://a.com", text="x" * (cap * 2))]
    combined = build_combined_text(pages, max_chars=cap)
    assert len(combined) <= cap + 100  # small allowance for header


# --- extract_research_fields ---

def test_extract_research_fields_returns_empty_on_no_text():
    llm = MagicMock()
    result = extract_research_fields(llm, "Acme", "", max_tokens=700)
    assert result == {}
    llm.generate_json.assert_not_called()


def test_extract_research_fields_returns_empty_on_llm_failure():
    llm = MagicMock()
    llm.generate_json.return_value = LLMJSONResult(success=False, error="boom")
    result = extract_research_fields(llm, "Acme", "some text", max_tokens=700)
    assert result == {}


def test_extract_research_fields_returns_parsed_dict():
    llm = MagicMock()
    llm.generate_json.return_value = LLMJSONResult(success=True, data={"description": "A SaaS company"})
    result = extract_research_fields(llm, "Acme", "some text", max_tokens=700)
    assert result == {"description": "A SaaS company"}


# --- ResearchAgent end-to-end (all deps mocked) ---

def make_agent_with_mocks(llm_data: dict, source_texts: dict[str, str]):
    settings = Settings(min_amount_usd=1_000_000, max_amount_usd=5_000_000, max_research_iterations=10)

    web = MagicMock()
    search = MagicMock()
    llm = MagicMock()
    llm.quota_exceeded = False

    def fake_fetch(url):
        text = source_texts.get(url, "")
        if not text:
            return PageFetchResult(url=url, success=False, error="not found")
        return PageFetchResult(url=url, success=True, final_url=url, text=text)

    web.fetch_page.side_effect = fake_fetch
    search.search.return_value = SearchResponse(query="q", success=False, error="skip")
    llm.generate_json.return_value = LLMJSONResult(success=True, data=llm_data)
    agent = ResearchAgent(web_service=web, search_service=search, llm_service=llm, settings=settings)
    return agent


def test_research_candidate_qualifies_when_all_evidence_supported():
    source_url = "https://acme.io/news"
    source_text = (
        "Acme is a SaaS platform for logistics. Acme raised $2.5M in seed funding in 2024. "
        "The company has no US office and focuses on the European market. "
        "Jane Doe is the CEO of Acme."
    )
    llm_data = {
        "description": "SaaS platform for logistics",
        "industry": "Logistics",
        "funding_amount_usd": 2_500_000,
        "funding_evidence": {"claim": "Raised $2.5M seed funding", "source_url": source_url, "excerpt": "Acme raised $2.5M in seed funding"},
        "tech_platform": True,
        "tech_platform_evidence": {"claim": "Operates a SaaS platform", "source_url": source_url, "excerpt": "Acme is a SaaS platform for logistics"},
        "us_presence": "NONE",
        "us_presence_evidence": {"claim": "No US office", "source_url": source_url, "excerpt": "no US office and focuses on the European market"},
        "contact_name": "Jane Doe",
        "contact_role": "CEO",
        "contact_evidence": {"claim": "Jane Doe is CEO", "source_url": source_url, "excerpt": "Jane Doe is the CEO of Acme"},
    }

    agent = make_agent_with_mocks(llm_data, {source_url: source_text, "https://acme.io": source_text})
    candidate = make_candidate(source_url=source_url)

    research = agent.research_candidate(candidate)

    assert research.funding_amount_usd == 2_500_000
    assert research.tech_platform is True
    assert research.us_presence == USPresence.NONE
    assert research.contact_name == "Jane Doe"
    # Note: email is not yet set (Phase 4), so full qualification should NOT pass yet
    assert research.status == QualificationStatus.REJECTED
    assert RejectionReason.NO_VERIFIED_EMAIL in research.rejection_reasons
    # But the fields that Phase 3 owns should all have passed their own checks
    assert RejectionReason.FINANCIAL_OUT_OF_RANGE not in research.rejection_reasons
    assert RejectionReason.FINANCIAL_UNKNOWN not in research.rejection_reasons
    assert RejectionReason.NOT_TECH_PLATFORM not in research.rejection_reasons
    assert RejectionReason.US_PRESENCE_TOO_HIGH not in research.rejection_reasons


def test_research_candidate_rejects_unsupported_evidence():
    source_url = "https://acme.io/news"
    source_text = "Acme is a small consulting shop with no public financial details."
    llm_data = {
        "funding_amount_usd": 3_000_000,
        # excerpt does NOT appear anywhere in source_text -> must be discarded
        "funding_evidence": {"claim": "Raised $3M", "source_url": source_url, "excerpt": "Acme raised $3M in a massive round"},
    }

    agent = make_agent_with_mocks(llm_data, {source_url: source_text})
    candidate = make_candidate(source_url=source_url)

    research = agent.research_candidate(candidate)

    assert research.funding_evidence is None
    assert research.funding_amount_usd is None
    assert RejectionReason.FINANCIAL_UNKNOWN in research.rejection_reasons


def test_research_candidate_handles_total_llm_failure_gracefully():
    settings = Settings()
    web = MagicMock()
    search = MagicMock()
    llm = MagicMock()
    llm.quota_exceeded = False

    web.fetch_page.return_value = PageFetchResult(url="x", success=True, final_url="x", text="some text")
    search.search.return_value = SearchResponse(query="q", success=False, error="skip")
    llm.generate_json.return_value = LLMJSONResult(success=False, error="LLM down")

    agent = ResearchAgent(web_service=web, search_service=search, llm_service=llm, settings=settings)
    candidate = make_candidate()

    research = agent.research_candidate(candidate)
    assert research.status == QualificationStatus.REJECTED
    assert len(research.rejection_reasons) > 0  # everything unknown -> everything rejected


def test_research_many_respects_max_iterations():
    settings = Settings(max_research_iterations=2)
    web = MagicMock()
    search = MagicMock()
    llm = MagicMock()
    llm.quota_exceeded = False

    web.fetch_page.return_value = PageFetchResult(url="x", success=True, final_url="x", text="text")
    search.search.return_value = SearchResponse(query="q", success=False, error="skip")
    llm.generate_json.return_value = LLMJSONResult(success=False, error="skip")

    agent = ResearchAgent(web_service=web, search_service=search, llm_service=llm, settings=settings)
    candidates = [make_candidate(company_name=f"Co{i}", domain=f"co{i}.com", source_url=f"https://co{i}.com") for i in range(5)]

    results = agent.research_many(candidates)
    assert len(results) == 2


def test_research_many_tolerates_one_candidate_raising():
    settings = Settings(max_research_iterations=10)
    web = MagicMock()
    search = MagicMock()
    llm = MagicMock()
    llm.quota_exceeded = False

    call_count = {"n": 0}
    
    def flaky_fetch(url):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated crash")
        return PageFetchResult(url=url, success=True, final_url=url, text="text")

    web.fetch_page.side_effect = flaky_fetch
    search.search.return_value = SearchResponse(query="q", success=False, error="skip")
    llm.generate_json.return_value = LLMJSONResult(success=False, error="skip")

    agent = ResearchAgent(web_service=web, search_service=search, llm_service=llm, settings=settings)
    candidates = [make_candidate(company_name="A", domain="a.com", source_url="https://a.com"),
                  make_candidate(company_name="B", domain="b.com", source_url="https://b.com")]

    results = agent.research_many(candidates)
    assert len(results) == 1  # first one crashed and was skipped, second succeeded