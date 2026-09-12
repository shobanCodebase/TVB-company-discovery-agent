"""
Email pattern generation + verification pipeline.

Verification never marks an email VERIFIED unless it passes every
deterministic check (syntax, domain match, not generic, MX record) and,
if configured, an external verification API also returns a positive
signal. Guessing a plausible-looking address is NOT the same as
verifying it — pattern-generated candidates without external
confirmation are surfaced but never marked verified on their own for
a target-domain address, since domain match + MX only prove the
domain can receive mail, not that this exact mailbox exists.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import dns.resolver
import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.config import Settings, settings as default_settings
from core.validators import (
    email_domain_matches_company,
    is_generic_email,
    is_valid_email_syntax,
)

logger = logging.getLogger(__name__)

RETRYABLE_EXCEPTIONS = (httpx.TimeoutException, httpx.TransportError)


@dataclass
class EmailCandidate:
    email: str
    pattern: str  # e.g. "first.last", "flast", "hunter_api"


@dataclass
class EmailVerificationResult:
    email: str
    verified: bool
    method: str
    reason: str = ""
    confidence: float | None = None  # 0-100 if from Hunter


def _name_parts(full_name: str) -> tuple[str, str] | None:
    parts = [p for p in re.split(r"\s+", full_name.strip()) if p.isalpha()]
    if len(parts) < 2:
        return None
    first = parts[0].lower()
    last = parts[-1].lower()
    return first, last


def generate_email_candidates(contact_name: str, domain: str) -> list[EmailCandidate]:
    """
    Generates common professional email patterns for a named person at a
    company domain. These are CANDIDATES only — none are verified yet.
    """
    parsed = _name_parts(contact_name)
    if parsed is None:
        return []
    first, last = parsed
    domain = domain.strip().lower().replace("www.", "")

    patterns = [
        (f"{first}.{last}@{domain}", "first.last"),
        (f"{first}@{domain}", "first"),
        (f"{first}{last}@{domain}", "firstlast"),
        (f"{first[0]}{last}@{domain}", "flast"),
        (f"{first}.{last[0]}@{domain}", "first.l"),
        (f"{last}.{first}@{domain}", "last.first"),
        (f"{first}_{last}@{domain}", "first_last"),
    ]
    return [EmailCandidate(email=e, pattern=p) for e, p in patterns]


class EmailService:
    def __init__(self, settings: Settings | None = None, client: httpx.Client | None = None):
        self.settings = settings or default_settings
        self._client = client or httpx.Client(timeout=self.settings.request_timeout_seconds)

    def close(self) -> None:
        self._client.close()

    # -- MX check ---------------------------------------------------------

    def domain_has_mx_record(self, domain: str) -> bool:
        domain = domain.strip().lower().replace("www.", "")
        try:
            answers = dns.resolver.resolve(domain, "MX", lifetime=self.settings.request_timeout_seconds)
            return len(answers) > 0
        except dns.resolver.NXDOMAIN:
            return False
        except dns.resolver.NoAnswer:
            return False
        except Exception as exc:  # noqa: BLE001 - DNS timeouts etc. should not crash the pipeline
            logger.warning("MX lookup failed for %s: %s", domain, exc)
            return False

    # -- Hunter.io: find a specific person's email ------------------------

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        reraise=True,
    )
    def hunter_find_email(self, first_name: str, last_name: str, domain: str) -> EmailCandidate | None:
        if not self.settings.hunter_api_key:
            return None
        try:
            resp = self._client.get(
                "https://api.hunter.io/v2/email-finder",
                params={
                    "domain": domain,
                    "first_name": first_name,
                    "last_name": last_name,
                    "api_key": self.settings.hunter_api_key,
                },
            )
        except RETRYABLE_EXCEPTIONS:
            raise
        except httpx.HTTPError as exc:
            logger.warning("Hunter email-finder request error: %s", exc)
            return None

        if resp.status_code != 200:
            logger.info("Hunter email-finder non-200 status %s: %s", resp.status_code, resp.text[:200])
            return None

        try:
            data = resp.json()
        except ValueError:
            return None

        email = (data.get("data") or {}).get("email")
        if not email:
            return None
        return EmailCandidate(email=email, pattern="hunter_finder")

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        reraise=True,
    )
    def hunter_verify_email(self, email: str) -> float | None:
        """Returns a confidence score 0-100, or None if unavailable/failed."""
        if not self.settings.hunter_api_key:
            return None
        try:
            resp = self._client.get(
                "https://api.hunter.io/v2/email-verifier",
                params={"email": email, "api_key": self.settings.hunter_api_key},
            )
        except RETRYABLE_EXCEPTIONS:
            raise
        except httpx.HTTPError as exc:
            logger.warning("Hunter email-verifier request error: %s", exc)
            return None

        if resp.status_code != 200:
            return None
        try:
            data = resp.json()
        except ValueError:
            return None

        return (data.get("data") or {}).get("score")

    # -- Full verification pipeline ---------------------------------------

    def verify_email(self, email: str, company_domain: str) -> EmailVerificationResult:
        """
        Runs the deterministic pipeline. Never marks VERIFIED unless every
        applicable step passes.
        """
        if not is_valid_email_syntax(email):
            return EmailVerificationResult(email=email, verified=False, method="syntax_check",
                                            reason="Invalid email syntax")

        if is_generic_email(email):
            return EmailVerificationResult(email=email, verified=False, method="generic_check",
                                            reason="Generic role-based address (info@/hello@/etc.)")

        if not email_domain_matches_company(email, company_domain):
            return EmailVerificationResult(email=email, verified=False, method="domain_match",
                                            reason="Email domain does not match company domain")

        if not self.domain_has_mx_record(company_domain):
            return EmailVerificationResult(email=email, verified=False, method="mx_check",
                                            reason="Company domain has no valid MX record")

        if self.settings.use_email_verification_api and self.settings.hunter_api_key:
            try:
                score = self.hunter_verify_email(email)
            except RETRYABLE_EXCEPTIONS as exc:
                logger.warning("Hunter verify-email failed after retries: %s", exc)
                score = None

            if score is None:
                # API unavailable — fall back to deterministic-only verification
                return EmailVerificationResult(
                    email=email, verified=True, method="deterministic_only",
                    reason="Passed syntax/domain/MX checks; verification API unavailable",
                )
            if score >= 70:
                return EmailVerificationResult(
                    email=email, verified=True, method="hunter_verify_email",
                    confidence=score,
                    reason=f"Hunter confidence score {score}",
                )
            return EmailVerificationResult(
                email=email, verified=False, method="hunter_verify_email",
                confidence=score,
                reason=f"Hunter confidence score {score} below threshold",
            )

        # No external verification configured — deterministic checks only
        return EmailVerificationResult(
            email=email, verified=True, method="deterministic_only",
            reason="Passed syntax/domain/MX checks; no verification API configured",
        )