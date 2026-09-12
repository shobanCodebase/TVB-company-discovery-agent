"""
TVB Company Discovery Agent — Streamlit UI.

Wires together: DiscoveryAgent -> ResearchAgent -> ContactAgent via
core.pipeline.run_pipeline(), with live progress display, a results
table with per-lead evidence, and CSV/JSON export.

Secrets (API keys) are read from Streamlit's st.secrets when deployed
(Streamlit Community Cloud) and mirrored into os.environ so that
core.config.Settings (which reads from os.environ / .env) picks them
up transparently in both local and hosted environments.
"""

from __future__ import annotations

import io
import json
import logging
import os

import pandas as pd
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Mirror Streamlit secrets into environment variables (hosted deploys) ---
# Must happen BEFORE importing core.config, since Settings reads os.environ
# at import time via dataclass field defaults.
if hasattr(st, "secrets"):
    try:
        for key, value in st.secrets.items():
            if key not in os.environ:
                os.environ[key] = str(value)
    except Exception:  # noqa: BLE001 - no secrets.toml locally is fine
        pass

from core.config import settings  # noqa: E402
from core.pipeline import PipelineProgress, run_pipeline  # noqa: E402
from core.models import QualifiedLead  # noqa: E402

st.set_page_config(page_title="TVB Company Discovery Agent", layout="wide")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("TVB Company Discovery Agent")
st.caption(
    "Discovers technology companies with $1M–$5M in funding/revenue, minimal US presence, "
    "and a named CEO/co-founder with a verified professional email — from live web sources."
)

with st.expander("Qualification criteria", expanded=False):
    st.markdown(
        """
- Revenue **or** funding between **$1M and $5M USD**
- Operates a **technology platform**
- **Minimal to no US presence** (NONE or LOW only — UNKNOWN and MODERATE/HIGH are rejected)
- **Named CEO or co-founder** (not "the team" / "management")
- **Verified professional email** belonging to that person (no info@/hello@/contact@)

Every claim above is backed by evidence pulled from a real, fetched web page — nothing is guessed.
        """
    )

missing_keys = settings.validate_required()
if missing_keys:
    st.error(
        f"Missing required configuration: {', '.join(missing_keys)}. "
        "Set these in your `.env` file (local) or app secrets (hosted) before running discovery."
    )

if not settings.hunter_api_key:
    st.warning(
        "HUNTER_API_KEY is not set — email verification will fall back to syntax + domain + MX "
        "checks only (lower precision, no external confidence score)."
    )

# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

col1, col2 = st.columns([1, 3])
with col1:
    target_leads = st.number_input(
        "Target leads", min_value=5, max_value=50,
        value=settings.target_qualified_leads, step=5,
    )
    run_clicked = st.button("RUN DISCOVERY", type="primary", disabled=bool(missing_keys))

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------

if run_clicked:
    status_placeholder = st.empty()
    metrics_placeholder = st.empty()

    def on_progress(progress: PipelineProgress) -> None:
        status_placeholder.info(progress.message)
        with metrics_placeholder.container():
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Candidates discovered", progress.candidates_discovered)
            m2.metric("Companies researched", progress.companies_researched)
            m3.metric("Qualified", progress.qualified_count)
            m4.metric("Verified contacts", progress.verified_email_count)

    with st.spinner("Running discovery pipeline..."):
        result = run_pipeline(target_leads=int(target_leads), on_progress=on_progress)

    st.session_state["pipeline_result"] = result

    if result.llm_quota_exceeded:
        status_placeholder.warning(
            f"Run stopped early — the LLM's daily token quota was exhausted after "
            f"{result.rounds_run} round(s). Found {len(result.qualified_leads)} qualified "
            f"leads from {result.total_discovered} candidates before stopping. "
            f"Results below are genuine and evidence-backed — there just wasn't enough "
            f"quota left to research further candidates today."
        )
    else:
        status_placeholder.success(
            f"Done — {len(result.qualified_leads)} qualified leads from "
            f"{result.total_discovered} discovered candidates across {result.rounds_run} round(s)."
        )

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

result = st.session_state.get("pipeline_result")

if result is None:
    st.info("Click **RUN DISCOVERY** to start a fresh discovery run.")
else:
    leads: list[QualifiedLead] = result.qualified_leads

    if not leads:
        st.warning(
            "No qualifying leads were found in this run. This can happen if the live web sources "
            "found this time don't contain enough companies matching every mandatory criterion — "
            "try running discovery again, since query results vary."
        )
    else:
        st.subheader(f"Qualified leads ({len(leads)})")

        table_rows = [
            {
                "Company": lead.company_name,
                "Industry": lead.industry or "—",
                "Funding/Revenue": (
                    f"${lead.funding_amount_usd:,.0f}" if lead.funding_amount_usd
                    else (f"${lead.revenue_amount_usd:,.0f}" if lead.revenue_amount_usd else "—")
                ),
                "HQ": lead.headquarters_country or "—",
                "Contact": f"{lead.contact_name} ({lead.contact_role})",
                "Email": lead.contact_email,
                "Verification": lead.email_verification_method,
                "Score": lead.score,
            }
            for lead in leads
        ]
        df = pd.DataFrame(table_rows).sort_values("Score", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("#### Evidence detail")
        for lead in sorted(leads, key=lambda l: l.score, reverse=True):
            with st.expander(f"{lead.company_name} — score {lead.score}"):
                st.write(lead.description or "_No description extracted._")
                st.markdown(f"**Contact:** {lead.contact_name} — {lead.contact_role}")
                st.markdown(f"**Email:** {lead.contact_email} (`{lead.email_verification_method}`)")
                st.markdown("**Evidence:**")
                for ev in lead.evidence:
                    st.markdown(f"- *{ev.claim}* — [{ev.source_url}]({ev.source_url})")

        # --- Export ---
        st.markdown("#### Export")
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        json_data = json.dumps([lead.model_dump(mode="json") for lead in leads], indent=2)

        ecol1, ecol2 = st.columns(2)
        with ecol1:
            st.download_button(
                "Download CSV", data=csv_buffer.getvalue(),
                file_name="tvb_qualified_leads.csv", mime="text/csv",
            )
        with ecol2:
            st.download_button(
                "Download JSON", data=json_data,
                file_name="tvb_qualified_leads.json", mime="application/json",
            )

    with st.expander(f"All researched companies ({len(result.all_researched)}) — including rejected"):
        rejected_rows = [
            {
                "Company": r.company_name,
                "Status": r.status.value,
                "Rejection reasons": ", ".join(rr.value for rr in r.rejection_reasons) or "—",
            }
            for r in result.all_researched
        ]
        st.dataframe(pd.DataFrame(rejected_rows), use_container_width=True, hide_index=True)