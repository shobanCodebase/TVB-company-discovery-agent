"""
Deterministic, mandatory-criteria validators.

These are pure functions — no LLM calls, no network. The LLM's job
(later phases) is only to extract structured facts + evidence; the
decision of pass/fail always happens here in plain Python, per the
"never let scoring compensate for a failed mandatory condition" rule.
"""

from __future__ import annotations

import re
from typing import Optional

from core.models import CompanyResearch, Evidence, RejectionReason, USPresence

GENERIC_CONTACT_ROLES = {
    "team", "founder", "founders", "management", "leadership",
    "the team", "our team", "staff", "employees", "company",
}

GENERIC_EMAIL_LOCAL_PARTS = {
    "info", "hello", "contact", "support", "sales", "careers",
    "admin", "office", "press", "media", "marketing", "help",
    "team", "enquiries", "inquiries", "noreply", "no-reply",
}

EMAIL_REGEX = re.compile(
    r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?"
    r"(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+$"
)

ALLOWED_US_PRESENCE = {USPresence.NONE, USPresence.LOW}


# ---------------------------------------------------------------------------
# 1. Financial criterion
# ---------------------------------------------------------------------------

def validate_financial(
    research: CompanyResearch,
    min_amount: float = 1_000_000,
    max_amount: float = 5_000_000,
) -> tuple[bool, Optional[RejectionReason]]:
    """
    Passes if EITHER funding OR revenue falls within [min_amount, max_amount],
    AND that amount has supporting evidence.
    """
    candidates = [
        (research.funding_amount_usd, research.funding_evidence),
        (research.revenue_amount_usd, research.revenue_evidence),
    ]

    saw_any_amount = False
    for amount, evidence in candidates:
        if amount is None:
            continue
        saw_any_amount = True
        if evidence is None:
            continue  # amount without evidence never counts as a pass
        if min_amount <= amount <= max_amount:
            return True, None

    if not saw_any_amount:
        return False, RejectionReason.FINANCIAL_UNKNOWN

    return False, RejectionReason.FINANCIAL_OUT_OF_RANGE


# ---------------------------------------------------------------------------
# 2. Tech platform criterion
# ---------------------------------------------------------------------------

def validate_tech_platform(research: CompanyResearch) -> tuple[bool, Optional[RejectionReason]]:
    if research.tech_platform is None or research.tech_platform_evidence is None:
        return False, RejectionReason.TECH_PLATFORM_UNKNOWN
    if research.tech_platform is True:
        return True, None
    return False, RejectionReason.NOT_TECH_PLATFORM


# ---------------------------------------------------------------------------
# 3. US presence criterion
# ---------------------------------------------------------------------------

def validate_us_presence(research: CompanyResearch) -> tuple[bool, Optional[RejectionReason]]:
    if research.us_presence == USPresence.UNKNOWN:
        return False, RejectionReason.US_PRESENCE_UNKNOWN
    if research.us_presence in ALLOWED_US_PRESENCE:
        return True, None
    return False, RejectionReason.US_PRESENCE_TOO_HIGH


# ---------------------------------------------------------------------------
# 4. CEO / co-founder criterion
# ---------------------------------------------------------------------------

def validate_contact(research: CompanyResearch) -> tuple[bool, Optional[RejectionReason]]:
    name = (research.contact_name or "").strip()
    role = (research.contact_role or "").strip()

    if not name or research.contact_evidence is None:
        return False, RejectionReason.NO_NAMED_CONTACT

    if role.lower() in GENERIC_CONTACT_ROLES:
        return False, RejectionReason.GENERIC_CONTACT_ROLE

    # A named contact needs at least a first + last token to count as a real name
    if len(name.split()) < 2:
        return False, RejectionReason.NO_NAMED_CONTACT

    if "ceo" not in role.lower() and "founder" not in role.lower():
        return False, RejectionReason.GENERIC_CONTACT_ROLE

    return True, None


# ---------------------------------------------------------------------------
# 5. Email criteria (syntax / genericness — verification API is Phase 4)
# ---------------------------------------------------------------------------

def is_valid_email_syntax(email: str) -> bool:
    return bool(EMAIL_REGEX.match(email.strip()))


def is_generic_email(email: str) -> bool:
    local_part = email.split("@")[0].strip().lower()
    return local_part in GENERIC_EMAIL_LOCAL_PARTS


def email_domain_matches_company(email: str, company_domain: str) -> bool:
    """
    Professional email should be on the company's own domain (or a close
    variant). This is a heuristic gate, not final verification.
    """
    email_domain = email.split("@")[-1].strip().lower()
    company_domain = company_domain.strip().lower().replace("www.", "")
    return email_domain == company_domain


def validate_email_verified(research: CompanyResearch) -> tuple[bool, Optional[RejectionReason]]:
    email = (research.contact_email or "").strip()
    if not email or not research.email_verified:
        return False, RejectionReason.NO_VERIFIED_EMAIL
    if not is_valid_email_syntax(email):
        return False, RejectionReason.NO_VERIFIED_EMAIL
    if is_generic_email(email):
        return False, RejectionReason.NO_VERIFIED_EMAIL
    return True, None


# ---------------------------------------------------------------------------
# 6. Evidence sanity check
# ---------------------------------------------------------------------------

def evidence_is_plausible(evidence: Evidence, source_text: Optional[str] = None) -> bool:
    """
    Cheap guard against hallucinated evidence: if we have the raw source
    text available, require that the excerpt (or claim) actually appears
    in it. This is a heuristic substring check — a stronger check could
    use an LLM 'does this source support this claim?' call in research_agent.
    """
    if not evidence.claim or not evidence.source_url:
        return False
    if source_text and evidence.excerpt:
        return evidence.excerpt.strip().lower()[:80] in source_text.lower()
    return True


# ---------------------------------------------------------------------------
# 7. Full mandatory qualification (hard AND logic)
# ---------------------------------------------------------------------------

def qualify(
    research: CompanyResearch,
    min_amount: float = 1_000_000,
    max_amount: float = 5_000_000,
) -> tuple[bool, list[RejectionReason]]:
    """
    Runs every mandatory validator. Returns (passed, reasons).
    ALL checks run (not short-circuited) so rejection_reasons is complete
    and useful for debugging/UI, even though a single failure is enough
    to reject.
    """
    reasons: list[RejectionReason] = []

    ok, reason = validate_financial(research, min_amount, max_amount)
    if not ok and reason:
        reasons.append(reason)

    ok, reason = validate_tech_platform(research)
    if not ok and reason:
        reasons.append(reason)

    ok, reason = validate_us_presence(research)
    if not ok and reason:
        reasons.append(reason)

    ok, reason = validate_contact(research)
    if not ok and reason:
        reasons.append(reason)

    ok, reason = validate_email_verified(research)
    if not ok and reason:
        reasons.append(reason)

    return (len(reasons) == 0), reasons