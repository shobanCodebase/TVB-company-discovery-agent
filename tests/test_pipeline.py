from unittest.mock import MagicMock

import pytest

from core.config import Settings
from core.models import CompanyCandidate, CompanyResearch, Evidence, QualificationStatus, USPresence
from core.pipeline import PipelineProgress, run_pipeline


def make_candidate(name, domain):
    return CompanyCandidate(company_name=name, domain=domain, source_url=f"https://{domain}")


def make_evidence():
    return Evidence(claim="claim", source_url="https://acme.io")


def qualifying_after_email(name, domain):
    """A CompanyResearch that passes everything except email (as research_agent would leave it)."""
    return CompanyResearch(
        company_name=name,
        domain=domain,
        funding_amount_usd=2_000_000,
        funding_evidence=make_evidence(),
        tech_platform=True,
        tech_platform_evidence=make_evidence(),
        us_presence=USPresence.NONE,
        us_presence_evidence=make_evidence(),
        contact_name="Jane Doe",
        contact_role="CEO",
        contact_evidence=make_evidence(),
        status=QualificationStatus.REJECTED,  # as research_agent would leave it pre-email
    )


def test_run_pipeline_stops_when_target_reached():
    settings = Settings(max_discovery_queries=5, max_amount_usd=5_000_000, min_amount_usd=1_000_000)

    discovery_agent = MagicMock()
    discovery_agent.discover.return_value = [
        make_candidate("Acme", "acme.io"),
        make_candidate("Beta", "beta.com"),
    ]

    research_agent = MagicMock()
    research_agent.llm_service = MagicMock()
    research_agent.llm_service.quota_exceeded = False
    research_agent.research_many.side_effect = lambda candidates: [
        qualifying_after_email(c.company_name, c.domain) for c in candidates
    ]

    contact_agent = MagicMock()

    def fake_process_many(researches, min_amount, max_amount):
        for r in researches:
            r.contact_email = f"jane@{r.domain}"
            r.email_verified = True
            r.email_verification_method = "deterministic_only"
        return researches

    contact_agent.process_many.side_effect = fake_process_many

    result = run_pipeline(
        target_leads=2,
        settings=settings,
        discovery_agent=discovery_agent,
        research_agent=research_agent,
        contact_agent=contact_agent,
    )

    assert len(result.qualified_leads) == 2
    assert result.rounds_run == 1
    discovery_agent.discover.assert_called_once()  # stopped after round 1, target already met


def test_run_pipeline_runs_multiple_rounds_when_needed():
    settings = Settings(max_discovery_queries=5)

    discovery_agent = MagicMock()
    call_count = {"n": 0}

    def fake_discover(offset=0):
        call_count["n"] += 1
        return [make_candidate(f"Co{call_count['n']}", f"co{call_count['n']}.com")]

    discovery_agent.discover.side_effect = fake_discover

    research_agent = MagicMock()
    research_agent.llm_service = MagicMock()
    research_agent.llm_service.quota_exceeded = False
    research_agent.research_many.side_effect = lambda candidates: [
        qualifying_after_email(c.company_name, c.domain) for c in candidates
    ]

    contact_agent = MagicMock()

    def fake_process_many(researches, min_amount, max_amount):
        for r in researches:
            r.contact_email = f"jane@{r.domain}"
            r.email_verified = True
            r.email_verification_method = "deterministic_only"
        return researches

    contact_agent.process_many.side_effect = fake_process_many

    result = run_pipeline(
        target_leads=3,
        settings=settings,
        discovery_agent=discovery_agent,
        research_agent=research_agent,
        contact_agent=contact_agent,
    )

    assert len(result.qualified_leads) == 3
    assert result.rounds_run == 3


def test_run_pipeline_stops_when_no_new_candidates():
    settings = Settings(max_discovery_queries=5)

    discovery_agent = MagicMock()
    discovery_agent.discover.return_value = []  # nothing found, ever

    research_agent = MagicMock()
    contact_agent = MagicMock()

    result = run_pipeline(
        target_leads=10,
        settings=settings,
        discovery_agent=discovery_agent,
        research_agent=research_agent,
        contact_agent=contact_agent,
    )

    assert result.qualified_leads == []
    assert result.rounds_run == 1
    research_agent.research_many.assert_not_called()


def test_run_pipeline_stops_at_max_rounds_even_if_target_not_met():
    settings = Settings(max_discovery_queries=5)

    discovery_agent = MagicMock()
    counter = {"n": 0}

    def fake_discover(offset=0):
        counter["n"] += 1
        return [make_candidate(f"Co{counter['n']}", f"co{counter['n']}.com")]

    discovery_agent.discover.side_effect = fake_discover

    research_agent = MagicMock()
    research_agent.llm_service = MagicMock()
    research_agent.llm_service.quota_exceeded = False
    research_agent.research_many.side_effect = lambda candidates: [
        # never actually qualifies (missing tech platform evidence) -> target never reached
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
        max_rounds=3,
    )

    assert result.qualified_leads == []
    assert result.rounds_run == 3


def test_run_pipeline_deduplicates_candidates_across_rounds():
    settings = Settings(max_discovery_queries=5)

    discovery_agent = MagicMock()
    # Every round returns the SAME candidate -> should be filtered as duplicate after round 1
    discovery_agent.discover.return_value = [make_candidate("Acme", "acme.io")]

    research_agent = MagicMock()
    research_agent.llm_service = MagicMock()
    research_agent.llm_service.quota_exceeded = False
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
        max_rounds=3,
    )


    # Round 1 finds 1 new candidate and researches it; rounds 2-3 find 0 NEW candidates -> pipeline stops early
    assert research_agent.research_many.call_count == 1
    assert result.rounds_run == 2  # round 1 researches, round 2 detects no new candidates and breaks


def test_on_progress_callback_is_invoked():
    settings = Settings(max_discovery_queries=5)
    discovery_agent = MagicMock()
    discovery_agent.discover.return_value = [make_candidate("Acme", "acme.io")]

    research_agent = MagicMock()
    research_agent.llm_service = MagicMock()
    research_agent.llm_service.quota_exceeded = False
    research_agent.research_many.side_effect = lambda candidates: [
        qualifying_after_email(c.company_name, c.domain) for c in candidates
    ]
    contact_agent = MagicMock()
    contact_agent.process_many.side_effect = lambda researches, min_amount, max_amount: researches

    progress_calls = []
    result = run_pipeline(
        target_leads=1,
        settings=settings,
        discovery_agent=discovery_agent,
        research_agent=research_agent,
        contact_agent=contact_agent,
        on_progress=lambda p: progress_calls.append(p.message),
    )
    assert len(progress_calls) > 0
    assert any("Round 1" in msg for msg in progress_calls)