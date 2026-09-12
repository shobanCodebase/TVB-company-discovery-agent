"""
Stage 4 — Contact + email discovery/verification.

Only does email work for a CompanyResearch that has already passed the
financial, tech-platform, US-presence, and named-contact checks — doing
this earlier would waste Hunter API calls on companies that were going
to be rejected regardless of their email.

Order of attempts for finding an email:
    1. Hunter.io email-finder (if HUNTER_API_KEY configured) — most
       reliable, since it queries real mailbox data.
    2. Pattern generation (first.last@domain, etc.) — verified via the
       deterministic pipeline (syntax + domain + MX [+ Hunter verify]).
       The FIRST pattern that passes verification is used.

Never marks an email verified without running it through
EmailService.verify_email().
"""

from __future__ import annotations

import logging

from core.models import CompanyResearch, Evidence
from core.validators import (
    validate_contact,
    validate_financial,
    validate_tech_platform,
    validate_us_presence,
)
from services.email_service import EmailService, generate_email_candidates

logger = logging.getLogger(__name__)


def _prior_checks_pass(research: CompanyResearch, min_amount: float, max_amount: float) -> bool:
    checks = [
        validate_financial(research, min_amount, max_amount),
        validate_tech_platform(research),
        validate_us_presence(research),
        validate_contact(research),
    ]
    return all(ok for ok, _ in checks)


class ContactAgent:
    def __init__(self, email_service: EmailService | None = None):
        self.email_service = email_service or EmailService()

    def find_and_verify_email(
        self,
        research: CompanyResearch,
        min_amount: float = 1_000_000,
        max_amount: float = 5_000_000,
    ) -> CompanyResearch:
        """
        Mutates and returns `research` with contact_email / email_verified /
        email_verification_method / email_evidence populated if a verified
        email is found. Leaves those fields as-is (unverified) otherwise.
        """
        if not _prior_checks_pass(research, min_amount, max_amount):
            logger.info(
                "Skipping email lookup for %s: already fails a prior mandatory check",
                research.company_name,
            )
            return research

        contact_name = research.contact_name or ""
        domain = research.domain

        # 1. Try Hunter's email-finder first, if configured
        hunter_candidate = None
        parsed = contact_name.strip().split()
        if len(parsed) >= 2:
            first, last = parsed[0], parsed[-1]
            try:
                hunter_candidate = self.email_service.hunter_find_email(first, last, domain)
            except Exception:  # noqa: BLE001 - Hunter outage must not crash the pipeline
                logger.warning("Hunter email-finder failed for %s", research.company_name, exc_info=True)
                hunter_candidate = None

        candidates = []
        if hunter_candidate is not None:
            candidates.append(hunter_candidate)
        candidates.extend(generate_email_candidates(contact_name, domain))

        if not candidates:
            logger.info("No email candidates could be generated for %s (unparseable name)", research.company_name)
            return research

        for candidate in candidates:
            try:
                result = self.email_service.verify_email(candidate.email, domain)
            except Exception:  # noqa: BLE001 - one bad candidate must not stop the loop
                logger.warning(
                    "Verification error for candidate %s (%s)", candidate.email, research.company_name,
                    exc_info=True,
                )
                continue

            if result.verified:
                research.contact_email = result.email
                research.email_verified = True
                research.email_verification_method = result.method
                research.email_evidence = Evidence(
                    claim=f"Verified email for {contact_name} via {result.method} ({result.reason})",
                    source_url=f"https://{domain}",
                    source_type="email_verification",
                )
                logger.info(
                    "Verified email for %s: %s (%s)", research.company_name, result.email, result.method,
                )
                return research

        logger.info("No candidate email could be verified for %s", research.company_name)
        return research

    def process_many(
        self,
        researches: list[CompanyResearch],
        min_amount: float = 1_000_000,
        max_amount: float = 5_000_000,
    ) -> list[CompanyResearch]:
        results = []
        for research in researches:
            try:
                results.append(self.find_and_verify_email(research, min_amount, max_amount))
            except Exception:  # noqa: BLE001
                logger.exception("Unexpected error in contact agent for %s", research.company_name)
                results.append(research)  # keep it unverified rather than dropping it
        return results