from core.models import CompanyResearch, Evidence, USPresence
from core.scoring import score_company


def make_evidence(claim="claim") -> Evidence:
    return Evidence(claim=claim, source_url="https://example.com")


def test_full_score_for_complete_qualified_company():
    r = CompanyResearch(
        company_name="Acme",
        domain="acme.io",
        funding_amount_usd=2_500_000,
        funding_evidence=make_evidence(),
        revenue_amount_usd=3_000_000,
        revenue_evidence=make_evidence(),
        tech_platform=True,
        tech_platform_evidence=make_evidence(),
        us_presence=USPresence.NONE,
        us_presence_evidence=make_evidence(),
        contact_name="Jane Doe",
        contact_evidence=make_evidence(),
        contact_email="jane@acme.io",
        email_verified=True,
    )
    assert score_company(r) == 100


def test_zero_score_for_empty_research():
    r = CompanyResearch(company_name="Acme", domain="acme.io")
    assert score_company(r) == 0


def test_partial_financial_scores_less_than_full():
    r_full = CompanyResearch(
        company_name="Acme", domain="acme.io",
        funding_amount_usd=2_000_000, funding_evidence=make_evidence(),
        revenue_amount_usd=2_000_000, revenue_evidence=make_evidence(),
    )
    r_partial = CompanyResearch(
        company_name="Acme", domain="acme.io",
        funding_amount_usd=2_000_000, funding_evidence=make_evidence(),
    )
    assert score_company(r_full) > score_company(r_partial)


def test_low_us_presence_scores_less_than_none():
    r_none = CompanyResearch(
        company_name="Acme", domain="acme.io",
        us_presence=USPresence.NONE, us_presence_evidence=make_evidence(),
    )
    r_low = CompanyResearch(
        company_name="Acme", domain="acme.io",
        us_presence=USPresence.LOW, us_presence_evidence=make_evidence(),
    )
    assert score_company(r_none) > score_company(r_low)


def test_score_capped_at_100():
    r = CompanyResearch(
        company_name="Acme", domain="acme.io",
        funding_amount_usd=2_000_000, funding_evidence=make_evidence(),
        revenue_amount_usd=2_000_000, revenue_evidence=make_evidence(),
        tech_platform=True, tech_platform_evidence=make_evidence(),
        us_presence=USPresence.NONE, us_presence_evidence=make_evidence(),
        contact_name="Jane Doe", contact_evidence=make_evidence(),
        contact_email="jane@acme.io", email_verified=True,
    )
    assert score_company(r) <= 100