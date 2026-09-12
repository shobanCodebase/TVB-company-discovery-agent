from unittest.mock import MagicMock

from core.config import Settings
from core.models import CompanyCandidate, CompanyResearch, Evidence, QualificationStatus, USPresence
from core.pipeline import run_pipeline


def make_candidate(name, domain):
    return CompanyCandidate(company_name=name, domain=domain, source_url=f"https://{domain}")


def make_evidence():
    return Evidence(claim="claim", source_url="https://acme.io")


def test_pipeline_stops_early_when_llm_quota_exceeded():
    settings = Settings(max_discovery_queries=5)

    discovery_agent = MagicMock()
    counter = {"n": 0}

    def fake_discover(offset=0):
        counter["n"] += 1
        return [make_candidate(f"Co{counter['n']}", f"co{counter['n']}.com")]

    discovery_agent.discover.side_effect = fake_discover

    research_agent = MagicMock()
    research_agent.llm_service = MagicMock()
    research_agent.llm_service.quota_exceeded = True
    research_agent.llm_service.quota_exceeded_reason = "tokens per day exceeded"
    research_agent.research_many.side_effect = lambda candidates: [
        CompanyResearch(company_name=c.company_name, domain=c.domain, status=QualificationStatus.REJECTED)
        for c in candidates
    ]

    contact_agent = MagicMock()
    contact_agent.process_many.side_effect = lambda researches, min_amount, max_amount: researches

    result = run_pipeline(
        target_leads=100,
        settings=settings,
        discovery_agent=discovery_agent,
        research_agent=research_agent,
        contact_agent=contact_agent,
        max_rounds=5,
    )

    assert result.llm_quota_exceeded is True
    assert result.llm_quota_exceeded_reason == "tokens per day exceeded"
    assert result.rounds_run == 1  # stopped after the FIRST round once quota was seen
    discovery_agent.discover.assert_called_once()  # never attempted round 2


def test_pipeline_continues_normally_when_quota_not_exceeded():
    settings = Settings(max_discovery_queries=5)

    discovery_agent = MagicMock()
    discovery_agent.discover.return_value = []

    research_agent = MagicMock()
    research_agent.llm_service = MagicMock()
    research_agent.llm_service.quota_exceeded = False

    contact_agent = MagicMock()

    result = run_pipeline(
        target_leads=10,
        settings=settings,
        discovery_agent=discovery_agent,
        research_agent=research_agent,
        contact_agent=contact_agent,
    )

    assert result.llm_quota_exceeded is False
    assert result.llm_quota_exceeded_reason is None