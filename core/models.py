"""
Evidence-first Pydantic data model for the TVB lead discovery pipeline.

Design principle: every important factual claim (funding, revenue, tech
platform, US presence, contact) must be backed by an Evidence object that
points to the source text and URL it was extracted from. If a field has
no evidence, it must be None — never guessed.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class USPresence(str, Enum):
    NONE = "NONE"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"


class QualificationStatus(str, Enum):
    PENDING = "PENDING"
    CANDIDATE = "CANDIDATE"          # discovered, not yet researched
    RESEARCHED = "RESEARCHED"        # research done, not yet qualified
    QUALIFIED = "QUALIFIED"          # passed all mandatory criteria
    REJECTED = "REJECTED"            # failed at least one mandatory criterion


class RejectionReason(str, Enum):
    FINANCIAL_OUT_OF_RANGE = "FINANCIAL_OUT_OF_RANGE"
    FINANCIAL_UNKNOWN = "FINANCIAL_UNKNOWN"
    NOT_TECH_PLATFORM = "NOT_TECH_PLATFORM"
    TECH_PLATFORM_UNKNOWN = "TECH_PLATFORM_UNKNOWN"
    US_PRESENCE_TOO_HIGH = "US_PRESENCE_TOO_HIGH"
    US_PRESENCE_UNKNOWN = "US_PRESENCE_UNKNOWN"
    NO_NAMED_CONTACT = "NO_NAMED_CONTACT"
    GENERIC_CONTACT_ROLE = "GENERIC_CONTACT_ROLE"
    NO_VERIFIED_EMAIL = "NO_VERIFIED_EMAIL"
    DUPLICATE = "DUPLICATE"
    EVIDENCE_MISSING = "EVIDENCE_MISSING"
    EVIDENCE_UNSUPPORTED = "EVIDENCE_UNSUPPORTED"


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

class Evidence(BaseModel):
    """
    A single factual claim, tied to the exact source it came from.

    `claim` should be a short, specific statement (e.g. "Raised $2.5M seed
    round in March 2024"), not a paraphrase of an entire page. `source_url`
    must be a real URL that was actually fetched during research — never
    fabricated.
    """
    claim: str = Field(..., min_length=1)
    source_url: str = Field(..., min_length=1)
    source_type: Optional[str] = None  # e.g. "company_website", "news", "crunchbase", "linkedin"
    excerpt: Optional[str] = None      # raw snippet from the source text supporting the claim
    retrieved_at: Optional[datetime] = None

    @field_validator("claim")
    @classmethod
    def claim_not_placeholder(cls, v: str) -> str:
        banned = {"unknown", "n/a", "none", "tbd", ""}
        if v.strip().lower() in banned:
            raise ValueError(
                "Evidence.claim cannot be a placeholder — if there is no "
                "real claim, do not create an Evidence object at all (use None)."
            )
        return v


# ---------------------------------------------------------------------------
# Candidate (Stage 1 output — before research)
# ---------------------------------------------------------------------------

class CompanyCandidate(BaseModel):
    """Raw output of the discovery stage, before any research is done."""
    company_name: str
    domain: Optional[str] = None
    source_url: str
    search_snippet: Optional[str] = None
    discovery_query: Optional[str] = None
    discovered_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Company Research (Stage 2 output)
# ---------------------------------------------------------------------------

class CompanyResearch(BaseModel):
    company_name: str
    domain: str

    description: Optional[str] = None
    industry: Optional[str] = None

    # Financial
    funding_amount_usd: Optional[float] = None
    funding_evidence: Optional[Evidence] = None
    revenue_amount_usd: Optional[float] = None
    revenue_evidence: Optional[Evidence] = None

    # Tech platform
    tech_platform: Optional[bool] = None
    tech_platform_evidence: Optional[Evidence] = None

    # US presence
    us_presence: USPresence = USPresence.UNKNOWN
    us_presence_evidence: Optional[Evidence] = None

    # Headquarters
    headquarters_country: Optional[str] = None
    headquarters_evidence: Optional[Evidence] = None

    # Contact (CEO / co-founder)
    contact_name: Optional[str] = None
    contact_role: Optional[str] = None
    contact_evidence: Optional[Evidence] = None

    # Email (Stage 4 fills these in)
    contact_email: Optional[str] = None
    email_verified: bool = False
    email_verification_method: Optional[str] = None
    email_evidence: Optional[Evidence] = None

    # Pipeline bookkeeping
    status: QualificationStatus = QualificationStatus.PENDING
    rejection_reasons: list[RejectionReason] = Field(default_factory=list)
    score: Optional[int] = None

    researched_at: Optional[datetime] = None

    @field_validator("domain")
    @classmethod
    def normalize_domain(cls, v: str) -> str:
        v = v.strip().lower()
        v = v.replace("http://", "").replace("https://", "")
        v = v.replace("www.", "")
        v = v.split("/")[0]
        return v


# ---------------------------------------------------------------------------
# Final qualified lead (what gets shown/exported)
# ---------------------------------------------------------------------------

class QualifiedLead(BaseModel):
    company_name: str
    domain: str
    description: Optional[str] = None
    industry: Optional[str] = None

    funding_amount_usd: Optional[float] = None
    revenue_amount_usd: Optional[float] = None

    tech_platform: bool
    us_presence: USPresence
    headquarters_country: Optional[str] = None

    contact_name: str
    contact_role: str
    contact_email: str
    email_verification_method: str

    score: int

    evidence: list[Evidence] = Field(default_factory=list)

    @classmethod
    def from_research(cls, research: "CompanyResearch") -> "QualifiedLead":
        evidence = [
            e for e in [
                research.funding_evidence,
                research.revenue_evidence,
                research.tech_platform_evidence,
                research.us_presence_evidence,
                research.contact_evidence,
                research.email_evidence,
            ] if e is not None
        ]
        return cls(
            company_name=research.company_name,
            domain=research.domain,
            description=research.description,
            industry=research.industry,
            funding_amount_usd=research.funding_amount_usd,
            revenue_amount_usd=research.revenue_amount_usd,
            tech_platform=bool(research.tech_platform),
            us_presence=research.us_presence,
            headquarters_country=research.headquarters_country,
            contact_name=research.contact_name or "",
            contact_role=research.contact_role or "",
            contact_email=research.contact_email or "",
            email_verification_method=research.email_verification_method or "",
            score=research.score or 0,
            evidence=evidence,
        )