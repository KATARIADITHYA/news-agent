"""
report/generator.py
────────────────────
Stage 3 — Report Generator

Assembles pipeline outputs into a structured report.
No entities, no market data — just verification + risk score + Claude narrative.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, Optional

import anthropic

from config import ANTHROPIC_API_KEY
from verification.verifier import ClaimVerification


# ── Report data structure ──────────────────────────────────────────────────────

@dataclass
class VolatilityReport:
    # Metadata
    generated_at:        str
    news_text:           str
    policy_fingerprint:  Dict[str, Any]

    # Stage 2 — Verification
    verification_status:      str
    verification_confidence:  int
    verification_explanation: str
    top_source_title:         str
    top_source_url:           str
    top_source_date:          str

    # Stage 3 — Risk Score
    risk_score:      int  = 0
    risk_label:      str  = ""
    risk_breakdown:  Dict = field(default_factory=dict)
    analyst_narrative: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ── ReportGenerator ────────────────────────────────────────────────────────────

class ReportGenerator:

    def __init__(self) -> None:
        self._claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) \
                       if ANTHROPIC_API_KEY else None

    def generate(
        self,
        news_text:          str,
        policy_fingerprint: Dict[str, Any],
        verification:       ClaimVerification,
        risk_score:         int,
        risk_label:         str,
        risk_breakdown:     Dict[str, Any],
    ) -> VolatilityReport:

        report = VolatilityReport(
            generated_at=datetime.utcnow().isoformat() + "Z",
            news_text=news_text,
            policy_fingerprint=policy_fingerprint,
            verification_status=verification.status,
            verification_confidence=verification.confidence,
            verification_explanation=verification.explanation,
            top_source_title=verification.top_hit.title if verification.top_hit else "",
            top_source_url=verification.top_hit.url   if verification.top_hit else "",
            top_source_date=verification.top_hit.date_iso[:10] if verification.top_hit else "",
            risk_score=risk_score,
            risk_label=risk_label,
            risk_breakdown=risk_breakdown,
        )

        report.analyst_narrative = self._narrative(report)
        return report

    # ── Analyst narrative ──────────────────────────────────────────────────────

    def _narrative(self, report: VolatilityReport) -> str:
        if not self._claude:
            return self._template_narrative(report)

        prompt = f"""You are a senior geopolitical risk analyst.
Write a concise 3-4 sentence analyst narrative. Be specific, use the data,
avoid generic language. No bullet points.

NEWS: {report.news_text[:400]}

VERIFICATION: {report.verification_status.upper()} ({report.verification_confidence}/100)
SOURCE: {report.top_source_title} ({report.top_source_date})
URL: {report.top_source_url}

RISK SCORE: {report.risk_score}/90 ({report.risk_label})
BREAKDOWN:
  - Verification signal : {report.risk_breakdown.get('verification_signal', 0)}/30
  - Policy type         : {report.risk_breakdown.get('policy_type', 0)}/20
  - Legal authority     : {report.risk_breakdown.get('legal_authority', 0)}/15
  - Scope breadth       : {report.risk_breakdown.get('scope_breadth', 0)}/25

Write analyst narrative (3-4 sentences, no bullets, no headers):"""

        try:
            msg = self._claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
        except Exception as e:
            print(f"[ReportGenerator] Claude error: {e}")
            return self._template_narrative(report)

    @staticmethod
    def _template_narrative(report: VolatilityReport) -> str:
        return (
            f"This {report.policy_fingerprint.get('policy_type', 'policy action')} "
            f"was verified against whitehouse.gov records "
            f"(confidence: {report.verification_confidence}/100) and carries a "
            f"{report.risk_label} risk rating of {report.risk_score}/90. "
            f"Closest official source: {report.top_source_title} ({report.top_source_date})."
        )


# ── Plain-text export ──────────────────────────────────────────────────────────

def report_to_text(report: VolatilityReport) -> str:
    sep  = "=" * 70
    sep2 = "-" * 70
    lines = [
        sep,
        "POLITISENSE — TRUMP POLICY VOLATILITY REPORT",
        sep,
        f"Generated  : {report.generated_at}",
        f"Risk Score : {report.risk_score}/90  [{report.risk_label}]",
        "",
        "POLICY NEWS",
        sep2,
        report.news_text,
        "",
        f"Policy type    : {report.policy_fingerprint.get('policy_type', 'N/A')}",
        f"Legal authority: {report.policy_fingerprint.get('legal_authority', 'N/A')}",
        f"Scope          : {report.policy_fingerprint.get('scope', 'N/A')}",
        f"Target         : {report.policy_fingerprint.get('target', 'N/A')}",
        "",
        "STAGE 2 — FACT-CHECK",
        sep2,
        f"Status     : {report.verification_status.upper()}",
        f"Confidence : {report.verification_confidence}/100  (avg cosine similarity × 100)",
        f"Explanation: {report.verification_explanation}",
        f"Source     : {report.top_source_title}",
        f"URL        : {report.top_source_url}",
        f"Date       : {report.top_source_date}",
        "",
        "STAGE 3 — RISK SCORE",
        sep2,
        f"  Verification signal : {report.risk_breakdown.get('verification_signal', 0):>3}/30",
        f"  Policy type         : {report.risk_breakdown.get('policy_type', 0):>3}/20",
        f"  Legal authority     : {report.risk_breakdown.get('legal_authority', 0):>3}/15",
        f"  Scope breadth       : {report.risk_breakdown.get('scope_breadth', 0):>3}/25",
        f"  {'─'*25}",
        f"  TOTAL               : {report.risk_score:>3}/90  [{report.risk_label}]",
        "",
        "ANALYST NARRATIVE",
        sep2,
        report.analyst_narrative,
        "",
        sep,
    ]
    return "\n".join(lines)
