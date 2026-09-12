from unittest.mock import MagicMock

from agents.research_agent import ResearchAgent
from core.config import Settings
from core.models import CompanyCandidate
from services.llm_service import LLMJSONResult
from services.search_service import SearchResponse
from services.web_service import PageFetchResult


def make_candidate(name="Acme", domain="acme.io"):
    return CompanyCandidate(company_name=name, domain=domain, source_url=f"https://{domain}")


def make_agent(enable_cache=True):
    settings = Settings(enable_cache=enable_cache)
    web = MagicMock()
    search = MagicMock()
    llm = MagicMock()

    web.fetch_page.return_value = PageFetchResult(url="x", success=True, final_url="x", text="some text")
    search.search.return_value = SearchResponse(query="q", success=False, error="skip")
    llm.generate_json.return_value = LLMJSONResult(success=True, data={"description": "test"})
    llm.quota_exceeded = False

    return ResearchAgent(web_service=web, search_service=search, llm_service=llm, settings=settings), llm


def test_same_domain_not_researched_twice():
    agent, llm = make_agent()
    c1 = make_candidate(name="Acme Inc", domain="acme.io")
    c2 = make_candidate(name="Acme SaaS", domain="acme.io")  # same domain, different discovered name

    agent.research_candidate(c1)
    agent.research_candidate(c2)

    assert llm.generate_json.call_count == 1  # second call served from cache


def test_cache_disabled_researches_every_time():
    agent, llm = make_agent(enable_cache=False)
    c1 = make_candidate(name="Acme Inc", domain="acme.io")
    c2 = make_candidate(name="Acme SaaS", domain="acme.io")

    agent.research_candidate(c1)
    agent.research_candidate(c2)

    assert llm.generate_json.call_count == 2


def test_quota_exceeded_skips_page_gathering():
    agent, llm = make_agent()
    llm.quota_exceeded = True
    llm.quota_exceeded_reason = "tokens per day exceeded"

    candidate = make_candidate()
    research = agent.research_candidate(candidate)

    # web/search should never have been called once quota is known exhausted
    agent.web_service.fetch_page.assert_not_called()
    agent.search_service.search.assert_not_called()
    llm.generate_json.assert_not_called()
    assert research.description is None


def test_research_many_respects_domain_cache_across_batch():
    agent, llm = make_agent()
    candidates = [make_candidate(f"Acme {i}", domain="acme.io") for i in range(5)]
    results = agent.research_many(candidates)

    assert len(results) == 5
    assert llm.generate_json.call_count == 1  # only researched once despite 5 candidates, same domain