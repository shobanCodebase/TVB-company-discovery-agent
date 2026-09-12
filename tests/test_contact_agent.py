from unittest.mock import MagicMock

import pytest

from agents.contact_agent import ContactAgent, _prior_checks_pass
from core.models import CompanyResearch, Evidence, USPresence
from services.email_service import EmailCandidate, EmailVerificationResult


def make_evidence(claim="claim") -> Evidence:
    return Evidence(claim=claim, source_url="https://acme.io")


def qualifying_research(**overrides) -> CompanyResearch:
    defaults = dict(
        company_name="Acme",
        domain="acme.io",
        funding_amount_usd=2_500_000,
        funding_evidence=make_evidence(),
        tech_platform=True,
        tech_platform_evidence=make_evidence(),
        us_presence=USPresence.NONE,
        us_presence_evidence=make_evidence(),
        contact_name="Jane Doe",
        contact_role="CEO",
        contact_evidence=make_evidence(),
    )
    defaults.update(overrides)
    return CompanyResearch(**defaults)


def test_prior_checks_pass_true_for_qualifying_research():
    research = qualifying_research()
    assert _prior_checks_pass(research, 1_000_000, 5_000_000) is True


def test_prior_checks_pass_false_when_financial_fails():
    research = qualifying_research(funding_amount_usd=50_000_000)
    assert _prior_checks_pass(research, 1_000_000, 5_000_000) is False


def test_find_and_verify_email_skips_when_prior_checks_fail():
    research = qualifying_research(tech_platform=False)
    email_service = MagicMock()
    agent = ContactAgent(email_service=email_service)

    result = agent.find_and_verify_email(research)

    email_service.hunter_find_email.assert_not_called()
    email_service.verify_email.assert_not_called()
    assert result.email_verified is False


def test_find_and_verify_email_uses_hunter_candidate_first():
    research = qualifying_research()
    email_service = MagicMock()
    email_service.hunter_find_email.return_value = EmailCandidate(email="jane.doe@acme.io", pattern="hunter_finder")
    email_service.verify_email.return_value = EmailVerificationResult(
        email="jane.doe@acme.io", verified=True, method="hunter_verify_email", confidence=90,
    )

    agent = ContactAgent(email_service=email_service)
    result = agent.find_and_verify_email(research)

    assert result.email_verified is True
    assert result.contact_email == "jane.doe@acme.io"
    assert result.email_verification_method == "hunter_verify_email"
    assert result.email_evidence is not None


def test_find_and_verify_email_falls_back_to_pattern_generation():
    research = qualifying_research()
    email_service = MagicMock()
    email_service.hunter_find_email.return_value = None  # Hunter has no match

    def fake_verify(email, domain):
        if email == "jane.doe@acme.io":
            return EmailVerificationResult(email=email, verified=True, method="deterministic_only")
        return EmailVerificationResult(email=email, verified=False, method="mx_check")

    email_service.verify_email.side_effect = fake_verify

    agent = ContactAgent(email_service=email_service)
    result = agent.find_and_verify_email(research)

    assert result.email_verified is True
    assert result.contact_email == "jane.doe@acme.io"


def test_find_and_verify_email_leaves_unverified_when_nothing_passes():
    research = qualifying_research()
    email_service = MagicMock()
    email_service.hunter_find_email.return_value = None
    email_service.verify_email.return_value = EmailVerificationResult(
        email="x", verified=False, method="mx_check", reason="no mx",
    )

    agent = ContactAgent(email_service=email_service)
    result = agent.find_and_verify_email(research)

    assert result.email_verified is False
    assert result.contact_email is None


def test_find_and_verify_email_handles_unparseable_contact_name():
    research = qualifying_research(contact_name="Cher")
    email_service = MagicMock()

    agent = ContactAgent(email_service=email_service)
    result = agent.find_and_verify_email(research)

    assert result.email_verified is False
    email_service.verify_email.assert_not_called()


def test_find_and_verify_email_tolerates_hunter_exception():
    research = qualifying_research()
    email_service = MagicMock()
    email_service.hunter_find_email.side_effect = RuntimeError("Hunter API down")
    email_service.verify_email.return_value = EmailVerificationResult(
        email="jane.doe@acme.io", verified=True, method="deterministic_only",
    )

    agent = ContactAgent(email_service=email_service)
    result = agent.find_and_verify_email(research)

    # Should still fall through to pattern generation despite Hunter crashing
    assert result.email_verified is True


def test_find_and_verify_email_tolerates_verification_exception_per_candidate():
    research = qualifying_research()
    email_service = MagicMock()
    email_service.hunter_find_email.return_value = None

    call_count = {"n": 0}

    def flaky_verify(email, domain):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("verification crashed")
        return EmailVerificationResult(email=email, verified=True, method="deterministic_only")

    email_service.verify_email.side_effect = flaky_verify

    agent = ContactAgent(email_service=email_service)
    result = agent.find_and_verify_email(research)

    assert result.email_verified is True  # second candidate succeeded


def test_process_many_handles_mixed_results():
    research1 = qualifying_research(company_name="Acme")
    research2 = qualifying_research(company_name="Beta", domain="beta.com", contact_name="Bob Smith")

    email_service = MagicMock()
    email_service.hunter_find_email.return_value = None
    email_service.verify_email.return_value = EmailVerificationResult(
        email="verified@example.com", verified=True, method="deterministic_only",
    )

    agent = ContactAgent(email_service=email_service)
    results = agent.process_many([research1, research2])

    assert len(results) == 2
    assert all(r.email_verified for r in results)


def test_process_many_tolerates_one_company_raising(monkeypatch):
    research1 = qualifying_research(company_name="Acme")
    research2 = qualifying_research(company_name="Beta", domain="beta.com", contact_name="Bob Smith")

    email_service = MagicMock()
    agent = ContactAgent(email_service=email_service)

    call_count = {"n": 0}
    original = agent.find_and_verify_email

    def flaky(research, min_amount=1_000_000, max_amount=5_000_000):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("boom")
        return original(research, min_amount, max_amount)

    agent.find_and_verify_email = flaky
    results = agent.process_many([research1, research2])
    assert len(results) == 2  # first one kept (unverified) rather than dropped