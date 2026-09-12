"""
Display-only scoring. This NEVER affects qualification — it only ranks
already-qualified (or in-progress) leads for the UI.

Weights (sum to 100):
    Financial evidence:    25
    Tech platform evidence: 20
    US presence evidence:   20
    CEO/co-founder evidence: 15
    Email verification:     20
"""

from __future__ import annotations

from core.models import CompanyResearch

WEIGHT_FINANCIAL = 25
WEIGHT_TECH_PLATFORM = 20
WEIGHT_US_PRESENCE = 20
WEIGHT_CONTACT = 15
WEIGHT_EMAIL = 20


def score_company(research: CompanyResearch) -> int:
    score = 0

    # Financial: full points if amount + evidence present, half if only one
    has_funding = research.funding_amount_usd is not None and research.funding_evidence is not None
    has_revenue = research.revenue_amount_usd is not None and research.revenue_evidence is not None
    if has_funding and has_revenue:
        score += WEIGHT_FINANCIAL
    elif has_funding or has_revenue:
        score += WEIGHT_FINANCIAL - 5

    if research.tech_platform is True and research.tech_platform_evidence is not None:
        score += WEIGHT_TECH_PLATFORM

    if research.us_presence_evidence is not None:
        if research.us_presence.value == "NONE":
            score += WEIGHT_US_PRESENCE
        elif research.us_presence.value == "LOW":
            score += WEIGHT_US_PRESENCE - 5

    if research.contact_name and research.contact_evidence is not None:
        score += WEIGHT_CONTACT

    if research.email_verified and research.contact_email:
        score += WEIGHT_EMAIL

    return min(score, 100)