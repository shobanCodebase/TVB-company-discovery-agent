from unittest.mock import MagicMock

from core.config import Settings
from core.models import CompanyCandidate, CompanyResearch, QualificationStatus
from core.pipeline import run_pipeline


def make_candidate(name, domain, snippet=""):
    return CompanyCandidate(company_name=name, domain=domain, source_url=f"https://{domain}", search_snippet=snippet)


def test_pipeline_only_researches_relevant_candidates_up_to_cap():
    settings = Settings(max_discovery_queries=5, max_candidates_for_research=1)

    discovery_agent = MagicMock()
    discovery_agent.discover.return_value = [
        make_candidate("Wiki Page", "en.wikipedia.org"),  # filtered out
        make_candidate("Acme SaaS", "acme.io", snippet="SaaS startup raised funding"),
        make_candidate("Beta Startup", "beta.com", snippet="tech startup CEO founder"),
    ]

    research_agent = MagicMock()
    research_agent.llm_service = MagicMock()
    research_agent.llm_service.quota_exceeded = False
    research_agent.research_many.side_effect = lambda candidates: [
        CompanyResearch(company_name=c.company_name, domain=c.domain, status=QualificationStatus.REJECTED)
        for c in candidates
    ]

    contact_agent = MagicMock()
    contact_agent.process_many.side_effect = lambda researches, min_amount, max_amount: researches

    run_pipeline(
        target_leads=100,
        settings=settings,
        discovery_agent=discovery_agent,
        research_agent=research_agent,
        contact_agent=contact_agent,
        max_rounds=1,
    )

    # Wikipedia page must never reach research_many; only 1 candidate (the cap) should
    called_candidates = research_agent.research_many.call_args[0][0]
    assert len(called_candidates) == 1
    assert called_candidates[0].domain in {"acme.io", "beta.com"}

