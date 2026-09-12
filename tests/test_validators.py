from datetime import datetime

import pytest

from core.models import CompanyResearch, Evidence, RejectionReason, USPresence
from core.validators import (
    email_domain_matches_company,
    is_generic_email,
    is_valid_email_syntax,
    qualify,
    validate_contact,
    validate_email_verified,
    validate_financial,
    validate_tech_platform,
    validate_us_presence,
)


def make_evidence(claim="Raised $2.5M seed round", url="https://example.com/news") -> Evidence:
    return Evidence(claim=claim, source_url=url, source_type="news")


def base_research(**overrides) -> CompanyResearch:
    defaults = dict(
        company_name="Acme SaaS",
        domain="acme.io",
        funding_amount_usd=2_500_000,
        funding_evidence=make_evidence(),
        tech_platform=True,
        tech_platform_evidence=make_evidence("Operates a SaaS platform"),
        us_presence=USPresence.NONE,
        us_presence_evidence=make_evidence("HQ in Berlin, no US office"),
        contact_name="Jane Doe",
        contact_role="CEO",
        contact_evidence=make_evidence("Jane Doe is CEO"),
        contact_email="jane@acme.io",
        email_verified=True,
    )
    defaults.update(overrides)
    return CompanyResearch(**defaults)


# --- Financial ---

def test_financial_in_range_passes():
    r = base_research()
    ok, reason = validate_financial(r)
    assert ok is True
    assert reason is None


def test_financial_out_of_range_fails():
    r = base_research(funding_amount_usd=10_000_000)
    ok, reason = validate_financial(r)
    assert ok is False
    assert reason == RejectionReason.FINANCIAL_OUT_OF_RANGE


def test_financial_unknown_fails():
    r = base_research(funding_amount_usd=None, funding_evidence=None,
                       revenue_amount_usd=None, revenue_evidence=None)
    ok, reason = validate_financial(r)
    assert ok is False
    assert reason == RejectionReason.FINANCIAL_UNKNOWN


def test_financial_amount_without_evidence_fails():
    r = base_research(funding_evidence=None)
    ok, reason = validate_financial(r)
    assert ok is False


def test_financial_revenue_can_pass_when_funding_out_of_range():
    r = base_research(
        funding_amount_usd=50_000_000,  # out of range
        revenue_amount_usd=3_000_000,   # in range
        revenue_evidence=make_evidence("Revenue of $3M in 2024"),
    )
    ok, reason = validate_financial(r)
    assert ok is True


# --- Tech platform ---

def test_tech_platform_true_with_evidence_passes():
    r = base_research()
    ok, reason = validate_tech_platform(r)
    assert ok is True


def test_tech_platform_false_fails():
    r = base_research(tech_platform=False)
    ok, reason = validate_tech_platform(r)
    assert ok is False
    assert reason == RejectionReason.NOT_TECH_PLATFORM


def test_tech_platform_unknown_fails():
    r = base_research(tech_platform=None, tech_platform_evidence=None)
    ok, reason = validate_tech_platform(r)
    assert ok is False
    assert reason == RejectionReason.TECH_PLATFORM_UNKNOWN


# --- US presence ---

@pytest.mark.parametrize("presence", [USPresence.NONE, USPresence.LOW])
def test_us_presence_allowed_values_pass(presence):
    r = base_research(us_presence=presence)
    ok, reason = validate_us_presence(r)
    assert ok is True


@pytest.mark.parametrize("presence", [USPresence.MODERATE, USPresence.HIGH])
def test_us_presence_disallowed_values_fail(presence):
    r = base_research(us_presence=presence)
    ok, reason = validate_us_presence(r)
    assert ok is False
    assert reason == RejectionReason.US_PRESENCE_TOO_HIGH


def test_us_presence_unknown_fails():
    r = base_research(us_presence=USPresence.UNKNOWN)
    ok, reason = validate_us_presence(r)
    assert ok is False
    assert reason == RejectionReason.US_PRESENCE_UNKNOWN


# --- Contact ---

def test_named_ceo_passes():
    r = base_research()
    ok, reason = validate_contact(r)
    assert ok is True


def test_generic_role_fails():
    r = base_research(contact_role="Team")
    ok, reason = validate_contact(r)
    assert ok is False
    assert reason == RejectionReason.GENERIC_CONTACT_ROLE


def test_missing_name_fails():
    r = base_research(contact_name=None, contact_evidence=None)
    ok, reason = validate_contact(r)
    assert ok is False
    assert reason == RejectionReason.NO_NAMED_CONTACT


def test_single_word_name_fails():
    r = base_research(contact_name="Jane")
    ok, reason = validate_contact(r)
    assert ok is False


def test_cofounder_role_passes():
    r = base_research(contact_role="Co-founder")
    ok, reason = validate_contact(r)
    assert ok is True


# --- Email ---

@pytest.mark.parametrize("email,expected", [
    ("jane@acme.io", True),
    ("jane.doe@acme.co.uk", True),
    ("not-an-email", False),
    ("jane@", False),
    ("@acme.io", False),
    ("jane doe@acme.io", False),
])
def test_email_syntax(email, expected):
    assert is_valid_email_syntax(email) is expected


@pytest.mark.parametrize("email,expected", [
    ("info@acme.io", True),
    ("hello@acme.io", True),
    ("contact@acme.io", True),
    ("jane@acme.io", False),
    ("jane.doe@acme.io", False),
])
def test_generic_email_detection(email, expected):
    assert is_generic_email(email) is expected


def test_email_domain_matches_company():
    assert email_domain_matches_company("jane@acme.io", "acme.io") is True
    assert email_domain_matches_company("jane@acme.io", "www.acme.io") is True
    assert email_domain_matches_company("jane@gmail.com", "acme.io") is False


def test_validate_email_verified_requires_flag():
    r = base_research(email_verified=False)
    ok, reason = validate_email_verified(r)
    assert ok is False
    assert reason == RejectionReason.NO_VERIFIED_EMAIL


def test_validate_email_verified_rejects_generic():
    r = base_research(contact_email="info@acme.io", email_verified=True)
    ok, reason = validate_email_verified(r)
    assert ok is False


# --- Full qualification ---

def test_qualify_all_pass():
    r = base_research()
    passed, reasons = qualify(r)
    assert passed is True
    assert reasons == []


def test_qualify_collects_all_failures():
    r = base_research(
        funding_amount_usd=None,
        funding_evidence=None,
        tech_platform=None,
        tech_platform_evidence=None,
        us_presence=USPresence.UNKNOWN,
        contact_role="Team",
        email_verified=False,
    )
    passed, reasons = qualify(r)
    assert passed is False
    assert RejectionReason.FINANCIAL_UNKNOWN in reasons
    assert RejectionReason.TECH_PLATFORM_UNKNOWN in reasons
    assert RejectionReason.US_PRESENCE_UNKNOWN in reasons
    assert RejectionReason.GENERIC_CONTACT_ROLE in reasons
    assert RejectionReason.NO_VERIFIED_EMAIL in reasons