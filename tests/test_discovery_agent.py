from unittest.mock import MagicMock

import pytest

from agents.discovery_agent import (
    DiscoveryAgent,
    candidate_from_result,
    extract_domain,
    generate_llm_queries,
    generate_template_queries,
    guess_company_name,
    is_blocklisted_domain,
)
from core.config import Settings
from services.llm_service import LLMJSONResult
from services.search_service import SearchResponse, SearchResultItem
from agents.discovery_agent import generate_all_template_queries


def test_generate_template_queries_respects_max():
    queries = generate_template_queries(max_queries=10)
    assert len(queries) == 10
    assert all(isinstance(q, str) and q for q in queries)


def test_generate_template_queries_are_diverse():
    queries = generate_template_queries(max_queries=60)
    assert len(set(queries)) == len(queries)  # no duplicates from the template product


def test_extract_domain_strips_www_and_path():
    assert extract_domain("https://www.acme.io/about/team") == "acme.io"
    assert extract_domain("https://beta.com") == "beta.com"


def test_extract_domain_handles_malformed_url():
    assert extract_domain("not a url") in (None, "")


def test_is_blocklisted_domain():
    assert is_blocklisted_domain("linkedin.com") is True
    assert is_blocklisted_domain("news.techcrunch.com") is True
    assert is_blocklisted_domain("acme.io") is False
    assert is_blocklisted_domain(None) is False


def test_guess_company_name_from_title():
    item = SearchResultItem(title="Acme Inc | Official Site", url="https://acme.io")
    assert guess_company_name(item, "acme.io") == "Acme Inc"


def test_guess_company_name_falls_back_to_domain():
    item = SearchResultItem(title="", url="https://beta-tech.io")
    assert guess_company_name(item, "beta-tech.io") == "Beta Tech"


def test_candidate_from_result_skips_blocklisted():
    item = SearchResultItem(title="LinkedIn profile", url="https://linkedin.com/company/acme")
    assert candidate_from_result(item, "query") is None


def test_candidate_from_result_builds_candidate():
    item = SearchResultItem(title="Acme raises $2M", url="https://acme.io/news", snippet="snippet text")
    candidate = candidate_from_result(item, "acme funding")
    assert candidate is not None
    assert candidate.domain == "acme.io"
    assert candidate.source_url == "https://acme.io/news"
    assert candidate.discovery_query == "acme funding"


def test_generate_llm_queries_returns_empty_on_failure():
    fake_llm = MagicMock()
    fake_llm.generate_json.return_value = LLMJSONResult(success=False, error="no key")
    queries = generate_llm_queries(fake_llm, count=5)
    assert queries == []


def test_generate_llm_queries_parses_valid_response():
    fake_llm = MagicMock()
    fake_llm.quota_exceeded = False
    fake_llm.generate_json.return_value = LLMJSONResult(
        success=True, data={"queries": ["query one", "query two", ""]}
    )
    queries = generate_llm_queries(fake_llm, count=5)
    assert queries == ["query one", "query two"]


def test_generate_llm_queries_handles_wrong_shape():
    fake_llm = MagicMock()
    fake_llm.generate_json.return_value = LLMJSONResult(success=True, data={"unexpected": "shape"})
    queries = generate_llm_queries(fake_llm, count=5)
    assert queries == []


# --- DiscoveryAgent integration (search + llm mocked) ---

def make_agent(max_queries=6, max_per_query=3, max_total=50):
    settings = Settings(
        max_discovery_queries=max_queries,
        max_candidates_per_query=max_per_query,
        max_candidates_total=max_total,
    )
    fake_search = MagicMock()
    fake_llm = MagicMock()
    fake_llm.quota_exceeded = False
    fake_llm.generate_json.return_value = LLMJSONResult(success=False, error="not configured")
    return DiscoveryAgent(search_service=fake_search, llm_service=fake_llm, settings=settings), fake_search
    agent, fake_search = make_agent(max_queries=2, max_per_query=5, max_total=50)

    fake_search.search.return_value = SearchResponse(
        query="q",
        success=True,
        results=[
            SearchResultItem(title="Acme Inc", url="https://acme.io/news"),
            SearchResultItem(title="Acme SaaS", url="https://www.acme.io/blog"),  # same domain -> dedup
            SearchResultItem(title="Beta Corp", url="https://beta.com"),
        ],
    )

    candidates = agent.discover()
    domains = {c.domain for c in candidates}
    assert "acme.io" in domains
    assert "beta.com" in domains
    assert len(candidates) == 2  # deduped


def test_discover_tolerates_failed_search_responses():
    agent, fake_search = make_agent(max_queries=2, max_per_query=5, max_total=50)
    fake_search.search.return_value = SearchResponse(query="q", success=False, error="boom")

    candidates = agent.discover()
    assert candidates == []


def test_discover_respects_max_candidates_total():
    agent, fake_search = make_agent(max_queries=5, max_per_query=10, max_total=3)

    def fake_search_fn(query, max_results):
        return SearchResponse(
            query=query,
            success=True,
            results=[
                SearchResultItem(title=f"Company {query} {i}", url=f"https://company-{query}-{i}.com")
                for i in range(10)
            ],
        )

    fake_search.search.side_effect = fake_search_fn
    candidates = agent.discover()
    assert len(candidates) <= 3

def test_generate_all_template_queries_is_full_product():
    all_queries = generate_all_template_queries()
    assert len(all_queries) == 60  # 6 templates x 10 topics


def test_generate_template_queries_with_offset_returns_different_window():
    first_batch = generate_template_queries(max_queries=10, offset=0)
    second_batch = generate_template_queries(max_queries=10, offset=10)
    assert first_batch != second_batch
    assert len(set(first_batch) & set(second_batch)) == 0


def test_generate_template_queries_offset_past_end_wraps_around():
    all_queries = generate_all_template_queries()
    pool_size = len(all_queries)

    result = generate_template_queries(max_queries=10, offset=pool_size + 5)
    expected = generate_template_queries(max_queries=10, offset=5)
    assert result == expected
    assert len(result) == 10  # never empty just because offset ran past the pool


def test_generate_template_queries_wraps_across_pool_boundary():
    all_queries = generate_all_template_queries()
    pool_size = len(all_queries)

    # Choose an offset near the end so the requested window straddles
    # the wrap-around point.
    offset = pool_size - 3
    result = generate_template_queries(max_queries=10, offset=offset)

    assert len(result) == 10
    assert result[:3] == all_queries[-3:]
    assert result[3:] == all_queries[:7]


def test_discover_with_offset_can_return_different_candidates():
    agent, fake_search = make_agent(max_queries=5, max_per_query=5, max_total=50)

    def fake_search_fn(query, max_results):
        return SearchResponse(
            query=query, success=True,
            results=[SearchResultItem(title=f"Co for {query}", url=f"https://{abs(hash(query))}.com")],
        )
    fake_search.search.side_effect = fake_search_fn

    round1 = agent.discover(offset=0)
    round2 = agent.discover(offset=5)
    urls_1 = {c.source_url for c in round1}
    urls_2 = {c.source_url for c in round2}
    assert urls_1 != urls_2

def test_guess_company_name_does_not_split_mid_word_hyphens():
    item = SearchResultItem(title="Egyptian e-commerce startup Foo raises $2M", url="https://foo.io")
    assert guess_company_name(item, "foo.io") == "Egyptian e-commerce startup Foo raises $2M"


def test_guess_company_name_does_not_split_ride_hailing():
    item = SearchResultItem(title="Tunisian Ride-hailing startup Bar raises $1M", url="https://bar.io")
    assert guess_company_name(item, "bar.io") == "Tunisian Ride-hailing startup Bar raises $1M"


def test_guess_company_name_still_strips_real_suffix_separator():
    item = SearchResultItem(title="Acme Inc - Official Site", url="https://acme.io")
    assert guess_company_name(item, "acme.io") == "Acme Inc"


def test_guess_company_name_still_strips_pipe_separator():
    item = SearchResultItem(title="Acme Inc | Home", url="https://acme.io")
    assert guess_company_name(item, "acme.io") == "Acme Inc"