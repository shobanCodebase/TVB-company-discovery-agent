"""
End-to-end pipeline orchestration: rounds of discovery -> relevance
pre-filter/priority cap -> research -> contact/email -> re-qualification,
stopping once MIN_QUALIFIED_LEADS is hit, no new candidates are found,
the LLM's daily token quota is exhausted, or a hard round limit is
reached.

The relevance pre-filter (core.relevance) is a discovery-quality gate
ONLY — it decides which candidates are worth an expensive Groq research
call, never whether a company qualifies. Qualification remains entirely
in core.validators.qualify(), run unchanged after research + contact.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

from agents.contact_agent import ContactAgent
from agents.discovery_agent import DiscoveryAgent
from agents.research_agent import ResearchAgent
from core.config import Settings, settings as default_settings
from core.deduplication import dedup_key
from core.models import CompanyResearch, QualificationStatus, QualifiedLead
from core.relevance import filter_and_prioritize
from core.scoring import score_company
from core.validators import qualify

logger = logging.getLogger(__name__)

# Default kept here for callers that don't pass an explicit settings
# object; run_pipeline's actual default now reads from
# settings.max_discovery_rounds so it's configurable via .env/Streamlit
# secrets (e.g. lowering it for hosted deployments where a long-running
# single request risks a tab-disconnect or platform idle/resource reset).
DEFAULT_MAX_DISCOVERY_ROUNDS = 5


@dataclass
class PipelineProgress:
    round_num: int = 0
    candidates_discovered: int = 0
    companies_researched: int = 0
    qualified_count: int = 0
    verified_email_count: int = 0
    message: str = ""


@dataclass
class PipelineResult:
    qualified_leads: list[QualifiedLead] = field(default_factory=list)
    all_researched: list[CompanyResearch] = field(default_factory=list)
    rounds_run: int = 0
    total_discovered: int = 0
    llm_quota_exceeded: bool = False
    llm_quota_exceeded_reason: str | None = None


ProgressCallback = Callable[[PipelineProgress], None]


def _log_rejection_breakdown(researched: list[CompanyResearch]) -> None:
    counts = Counter()
    for r in researched:
        for reason in r.rejection_reasons:
            counts[reason.value] += 1
    if counts:
        logger.info("Qualification rejection breakdown this round: %s", dict(counts))


def run_pipeline(
    target_leads: int,
    settings: Optional[Settings] = None,
    discovery_agent: Optional[DiscoveryAgent] = None,
    research_agent: Optional[ResearchAgent] = None,
    contact_agent: Optional[ContactAgent] = None,
    on_progress: Optional[ProgressCallback] = None,
    max_rounds: Optional[int] = None,
) -> PipelineResult:
    settings = settings or default_settings
    if max_rounds is None:
        max_rounds = settings.max_discovery_rounds
    discovery_agent = discovery_agent or DiscoveryAgent(settings=settings)
    research_agent = research_agent or ResearchAgent(settings=settings)
    contact_agent = contact_agent or ContactAgent()

    seen_keys: set[str] = set()
    all_researched: list[CompanyResearch] = []
    qualified: list[CompanyResearch] = []
    progress = PipelineProgress()

    def emit(message: str) -> None:
        progress.message = message
        if on_progress:
            on_progress(progress)

    for round_num in range(1, max_rounds + 1):
        progress.round_num = round_num
        emit(f"Round {round_num}: discovering candidates...")

        offset = (round_num - 1) * settings.max_discovery_queries
        candidates = discovery_agent.discover(offset=offset)
        logger.info("Round %d: discovery returned %d deduplicated candidates", round_num, len(candidates))

        new_candidates = []
        for c in candidates:
            key = dedup_key(c.company_name, c.domain)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            new_candidates.append(c)

        if not new_candidates:
            emit(f"Round {round_num}: no new candidates found — stopping discovery.")
            break

        logger.info("Round %d: %d candidates are new this run (not seen in a prior round)", round_num, len(new_candidates))

        selected_candidates, rejected_irrelevant = filter_and_prioritize(
            new_candidates, max_candidates=settings.max_candidates_for_research
        )
        logger.info(
            "Round %d: relevance pre-filter rejected %d non-company candidates; "
            "researching top %d of %d relevant candidates",
            round_num, rejected_irrelevant, len(selected_candidates),
            len(new_candidates) - rejected_irrelevant,
        )

        progress.candidates_discovered += len(new_candidates)
        emit(
            f"Round {round_num}: researching {len(selected_candidates)} prioritized candidates "
            f"(filtered {rejected_irrelevant} non-company results)..."
        )

        researched = research_agent.research_many(selected_candidates)
        all_researched.extend(researched)
        progress.companies_researched = len(all_researched)
        _log_rejection_breakdown(researched)

        emit(f"Round {round_num}: verifying contact emails...")
        with_email = contact_agent.process_many(
            researched,
            min_amount=settings.min_amount_usd,
            max_amount=settings.max_amount_usd,
        )
        logger.info(
            "Round %d: %d/%d researched companies have a verified email",
            round_num, sum(1 for r in with_email if r.email_verified), len(with_email),
        )

        for research in with_email:
            _requalify(research, settings)
            if research.status == QualificationStatus.QUALIFIED:
                qualified.append(research)

        progress.qualified_count = len(qualified)
        progress.verified_email_count = sum(1 for r in all_researched if r.email_verified)
        emit(f"Round {round_num} complete: {len(qualified)} qualified so far.")
        logger.info("Round %d complete: %d qualified total, %d researched total", round_num, len(qualified), len(all_researched))

        if research_agent.llm_service.quota_exceeded:
            reason = research_agent.llm_service.quota_exceeded_reason
            logger.warning("Stopping pipeline early — Groq daily token quota exhausted: %s", reason)
            emit(
                "LLM daily token quota exhausted — stopping further research rounds. "
                "Results below reflect what was verified before the quota was hit."
            )
            return PipelineResult(
                qualified_leads=[QualifiedLead.from_research(r) for r in qualified],
                all_researched=all_researched,
                rounds_run=round_num,
                total_discovered=len(seen_keys),
                llm_quota_exceeded=True,
                llm_quota_exceeded_reason=reason,
            )

        if len(qualified) >= target_leads:
            emit(f"Target of {target_leads} qualified leads reached.")
            break

    emit(f"Pipeline finished: {len(qualified)} qualified leads from {len(all_researched)} researched companies.")

    qualified_leads = [QualifiedLead.from_research(r) for r in qualified]
    return PipelineResult(
        qualified_leads=qualified_leads,
        all_researched=all_researched,
        rounds_run=progress.round_num,
        total_discovered=len(seen_keys),
    )


def _requalify(research: CompanyResearch, settings: Settings) -> None:
    passed, reasons = qualify(research, min_amount=settings.min_amount_usd, max_amount=settings.max_amount_usd)
    research.rejection_reasons = reasons
    research.status = QualificationStatus.QUALIFIED if passed else QualificationStatus.REJECTED
    research.score = score_company(research)