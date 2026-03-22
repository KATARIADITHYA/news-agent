"""
agent/nodes.py
──────────────
PolitiSense Agent — LangGraph node functions.

Each function is one node in the agent graph.
Every node receives the full AgentState, does its job,
and returns a DICT of only the fields it wants to update.

Nodes:
  1. fingerprint_node    → Stage 1: extract policy fingerprint (Claude NLP)
  2. query_rewrite_node  → rewrite news text for better ChromaDB retrieval
  3. retrieval_node      → Stage 2: search ChromaDB + compute confidence
  4. guard_node          → check confidence, decide: retry or continue
  5. report_node         → Stage 3: compute risk score + Claude narrative

Routing functions:
  route_after_guard      → "retry" / "report" / "report" (max retries hit)
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict

import anthropic

from config import ANTHROPIC_API_KEY, SIMILARITY_THRESHOLD, CONFIDENCE_DISPUTED
from rag.retriever import Retriever
from agent.state import AgentState


# ── constants ──────────────────────────────────────────────────────────────────
MAX_RETRIES = 2          # max retrieval retries before forcing report
LOW_CONFIDENCE_THRESHOLD = CONFIDENCE_DISPUTED   # below this → retry


# ── Node 1: Fingerprint ────────────────────────────────────────────────────────

def fingerprint_node(state: AgentState) -> Dict[str, Any]:
    """
    Stage 1 — Extract policy fingerprint using Claude NLP.

    No hardcoded lists. Claude extracts ANY country, commodity,
    legal authority, and policy type from the raw news text.
    """
    news_text = state["news_text"]
    print(f"\n[Node: fingerprint] Extracting fingerprint for: {news_text[:80]}…")

    if not ANTHROPIC_API_KEY:
        return {"fingerprint": _regex_fallback(news_text)}

    prompt = f"""Extract a structured policy fingerprint from this news text.

NEWS: {news_text}

Return ONLY a valid JSON object:
{{
  "legal_authority": "e.g. Section 232, IEEPA — or empty string if none",
  "policy_type": "one of: tariff / sanction / ban / export control / executive order / subsidy / investigation / penalty / policy action",
  "scope": "comma-separated list of ALL countries/regions mentioned",
  "target": "comma-separated list of ALL goods/commodities/industries affected"
}}

Rules:
- legal_authority: only if explicitly mentioned
- scope: include ALL countries, not just major ones
- Return JSON only, no markdown"""

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        msg    = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=250,
            messages=[{"role": "user", "content": prompt}],
        )
        raw  = re.sub(r"```(?:json)?", "", msg.content[0].text.strip()).strip().rstrip("`")
        data = json.loads(raw)
        fp   = {
            "legal_authority": data.get("legal_authority", ""),
            "policy_type":     data.get("policy_type", "policy action"),
            "scope":           data.get("scope", ""),
            "target":          data.get("target", ""),
        }
        print(f"[Node: fingerprint] → {fp}")
        return {"fingerprint": fp}
    except Exception as e:
        print(f"[Node: fingerprint] Claude failed ({e}) — regex fallback")
        return {"fingerprint": _regex_fallback(news_text)}


# ── Node 2: Query Rewriter ─────────────────────────────────────────────────────

def query_rewrite_node(state: AgentState) -> Dict[str, Any]:
    """
    Rewrite the news text into a keyword-dense search query.

    On retry (retry_count > 0), expands the query with related terms
    to try to find a better ChromaDB match.

    Same approach as deep-learning-rag-agent query_rewrite_node.
    """
    news_text   = state["news_text"]
    retry_count = state.get("retry_count", 0)
    fingerprint = state.get("fingerprint", {})

    print(f"\n[Node: query_rewrite] retry_count={retry_count}")

    if not ANTHROPIC_API_KEY:
        return {"rewritten_query": news_text}

    if retry_count == 0:
        # First attempt — clean up the query
        prompt = f"""Rewrite this news headline into a short keyword-dense search query
for a whitehouse.gov policy database. Remove conversational words.
Include: policy type, countries, commodities, legal authority if present.
Max 20 words. Return only the rewritten query, nothing else.

News: {news_text}
Policy context: {fingerprint}

Rewritten query:"""
    else:
        # Retry — expand with synonyms and related terms
        prompt = f"""The first search query did not find a confident match.
Rewrite this news into a BROADER search query using synonyms and related policy terms.
Try different angles — use the legal authority name, alternative country names, related trade concepts.
Max 20 words. Return only the rewritten query.

News: {news_text}
Policy context: {fingerprint}
Retry number: {retry_count}

Broader rewritten query:"""

    try:
        client   = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        msg      = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        rewritten = msg.content[0].text.strip()
        print(f"[Node: query_rewrite] '{news_text[:50]}' → '{rewritten}'")
        return {"rewritten_query": rewritten}
    except Exception as e:
        print(f"[Node: query_rewrite] Claude failed ({e}) — using original")
        return {"rewritten_query": news_text}


# ── Node 3: Retrieval ──────────────────────────────────────────────────────────

def retrieval_node(state: AgentState) -> Dict[str, Any]:
    """
    Stage 2 — Search ChromaDB and compute confidence.

    Uses rewritten_query for retrieval.
    Confidence = avg cosine similarity of chunks above threshold.
    Sets low_confidence=True if confidence < LOW_CONFIDENCE_THRESHOLD.
    """
    query       = state.get("rewritten_query") or state["news_text"]
    retry_count = state.get("retry_count", 0)

    print(f"\n[Node: retrieval] Searching ChromaDB with: '{query[:80]}'")

    retriever = Retriever()
    hits      = retriever.retrieve_whitehouse_only(query, top_k=5)

    if not hits:
        print("[Node: retrieval] No hits returned — low confidence")
        return {
            "verification": {
                "status":           "unverified",
                "confidence":       0,
                "confidence_raw":   0.0,
                "chunks_used":      0,
                "chunks_retrieved": 0,
                "explanation":      "No documents found in whitehouse.gov corpus.",
                "top_hit":          None,
                "all_hits":         [],
            },
            "low_confidence": True,
            "retry_count":    retry_count,
        }

    # Filter by threshold + compute average similarity
    above = [h for h in hits if h.similarity >= SIMILARITY_THRESHOLD]
    if not above:
        best = max(hits, key=lambda h: h.similarity)
        confidence_pct = 0
        confidence_raw = 0.0
        low_conf       = True
        status         = "unverified"
        explanation    = (
            f"No chunks met similarity threshold ({SIMILARITY_THRESHOLD}). "
            f"Best score was {best.similarity:.3f}."
        )
        top_hit = best
    else:
        avg_sim        = sum(h.similarity for h in above) / len(above)
        confidence_pct = round(avg_sim * 100)
        confidence_raw = avg_sim
        top_hit        = max(above, key=lambda h: h.similarity)
        low_conf       = confidence_pct < LOW_CONFIDENCE_THRESHOLD

        if confidence_pct >= 75:
            status      = "verified"
            explanation = f"Strong match. Source: \"{top_hit.title}\" ({top_hit.date_iso[:10]})"
        elif confidence_pct >= 55:
            status      = "likely"
            explanation = f"Partial match. Source: \"{top_hit.title}\" ({top_hit.date_iso[:10]})"
        elif confidence_pct >= 35:
            status      = "disputed"
            explanation = f"Weak match. Source: \"{top_hit.title}\" ({top_hit.date_iso[:10]})"
        else:
            status      = "unverified"
            explanation = f"No sufficient match found. Best: \"{top_hit.title}\""

    print(
        f"[Node: retrieval] confidence={confidence_pct}/100  "
        f"status={status}  low_confidence={low_conf}  "
        f"chunks_above_threshold={len(above)}"
    )

    # Serialise WhitehouseHit to dict for state storage
    top_hit_dict = {
        "title":      top_hit.title,
        "url":        top_hit.url,
        "date_iso":   top_hit.date_iso,
        "categories": top_hit.categories,
        "text":       top_hit.text[:500],
        "similarity": top_hit.similarity,
    } if top_hit else None

    return {
        "verification": {
            "status":           status,
            "confidence":       confidence_pct,
            "confidence_raw":   confidence_raw,
            "chunks_used":      len(above),
            "chunks_retrieved": len(hits),
            "explanation":      explanation,
            "top_hit":          top_hit_dict,
            "all_hits":         [{"title": h.title, "similarity": h.similarity} for h in hits],
        },
        "low_confidence": low_conf,
        "retry_count":    retry_count,
    }


# ── Node 4: Guard ──────────────────────────────────────────────────────────────

def guard_node(state: AgentState) -> Dict[str, Any]:
    """
    Confidence guard — decides whether to retry or continue.

    This is what makes PolitiSense an AGENT:
    - It observes the confidence score
    - It decides autonomously: retry with a better query, or proceed
    - It tracks how many retries have happened

    If confidence is low AND retries < MAX_RETRIES → increment retry count
    If confidence is OK  OR  retries >= MAX_RETRIES → proceed to report
    """
    low_confidence = state.get("low_confidence", False)
    retry_count    = state.get("retry_count", 0)
    confidence     = state.get("verification", {}).get("confidence", 0)

    print(f"\n[Node: guard] confidence={confidence}  low={low_confidence}  retry={retry_count}/{MAX_RETRIES}")

    if low_confidence and retry_count < MAX_RETRIES:
        new_retry = retry_count + 1
        print(f"[Node: guard] Low confidence → RETRY #{new_retry}")
        return {"retry_count": new_retry, "low_confidence": True}
    else:
        if retry_count >= MAX_RETRIES:
            print(f"[Node: guard] Max retries reached → proceeding to report")
        else:
            print(f"[Node: guard] Confidence OK → proceeding to report")
        return {"low_confidence": False}


# ── Node 5: Report ─────────────────────────────────────────────────────────────

def report_node(state: AgentState) -> Dict[str, Any]:
    """
    Stage 3 — Compute risk score and generate Claude narrative.

    Uses fingerprint + verification to compute risk score.
    Calls Claude to write a 3-4 sentence analyst narrative.
    """
    news_text    = state["news_text"]
    fp           = state.get("fingerprint", {})
    verification = state.get("verification", {})
    retry_count  = state.get("retry_count", 0)

    print(f"\n[Node: report] Building report (retries used: {retry_count})")

    # ── Risk score (4 signals, max 90) ────────────────────────────────────
    v_map = {"verified": 30, "likely": 22, "disputed": 12, "unverified": 5}
    v_sig = v_map.get(verification.get("status", "unverified"), 5)

    pt_map = {"tariff": 20, "export control": 18, "sanction": 18,
              "ban": 15, "executive order": 12, "policy action": 5}
    p_sig  = pt_map.get(fp.get("policy_type", "policy action"), 5)

    l_sig  = 15 if fp.get("legal_authority") else 0

    scope_count = len([x for x in fp.get("scope",  "").split(",") if x.strip()])
    tgt_count   = len([x for x in fp.get("target", "").split(",") if x.strip()])
    s_sig = min(25, (scope_count + tgt_count) * 5) if (scope_count + tgt_count) > 0 else 5

    total = min(90, v_sig + p_sig + l_sig + s_sig)
    label = ("CRITICAL" if total >= 75 else
             "HIGH"     if total >= 55 else
             "MEDIUM"   if total >= 35 else "LOW")

    risk_breakdown = {
        "verification_signal": v_sig,
        "policy_type":         p_sig,
        "legal_authority":     l_sig,
        "scope_breadth":       s_sig,
    }

    # ── Claude narrative ──────────────────────────────────────────────────
    narrative = _generate_narrative(
        news_text=news_text,
        fp=fp,
        verification=verification,
        risk_score=total,
        risk_label=label,
        risk_breakdown=risk_breakdown,
        retry_count=retry_count,
    )

    top_hit  = verification.get("top_hit") or {}
    confidence = verification.get("confidence", 0)
    print(f"[Node: report] Risk={total}/90 [{label}]  Confidence={confidence}/100")

    final_report = {
        "news_text":                news_text,
        "policy_fingerprint":       fp,
        "verification_status":      verification.get("status", "unverified"),
        "verification_confidence":  confidence,
        "verification_explanation": verification.get("explanation", ""),
        "top_source_title":         top_hit.get("title", ""),
        "top_source_url":           top_hit.get("url", ""),
        "top_source_date":          top_hit.get("date_iso", "")[:10] if top_hit.get("date_iso") else "",
        "risk_score":               total,
        "risk_label":               label,
        "risk_breakdown":           risk_breakdown,
        "analyst_narrative":        narrative,
        "retries_used":             retry_count,
    }

    return {
        "risk_score":    total,
        "risk_label":    label,
        "risk_breakdown": risk_breakdown,
        "narrative":     narrative,
        "final_report":  final_report,
    }


# ── Routing function ───────────────────────────────────────────────────────────

def route_after_guard(state: AgentState) -> str:
    """
    Conditional edge — called after guard_node.

    Returns:
      "retry"  → go back to query_rewrite → retrieval (agent loops)
      "report" → proceed to report_node (agent finishes)
    """
    if state.get("low_confidence") and state.get("retry_count", 0) <= MAX_RETRIES:
        return "retry"
    return "report"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _generate_narrative(
    news_text: str,
    fp: dict,
    verification: dict,
    risk_score: int,
    risk_label: str,
    risk_breakdown: dict,
    retry_count: int,
) -> str:
    """Ask Claude to write a 3-4 sentence analyst narrative."""
    if not ANTHROPIC_API_KEY:
        return (
            f"This {fp.get('policy_type', 'policy action')} was verified against "
            f"whitehouse.gov (confidence: {verification.get('confidence', 0)}/100) "
            f"and carries a {risk_label} risk score of {risk_score}/90."
        )

    retry_note = f" (required {retry_count} retrieval {'retry' if retry_count == 1 else 'retries'})" if retry_count > 0 else ""

    prompt = f"""You are a senior geopolitical risk analyst.
Write a concise 3-4 sentence analyst narrative. Be specific, use the data provided.
No bullet points, no headers.

NEWS: {news_text[:400]}
VERIFICATION: {verification.get('status', '').upper()} ({verification.get('confidence', 0)}/100){retry_note}
SOURCE: {verification.get('top_hit', {}).get('title', 'N/A')}
RISK SCORE: {risk_score}/90 ({risk_label})
BREAKDOWN:
  Verification: {risk_breakdown.get('verification_signal', 0)}/30
  Policy type:  {risk_breakdown.get('policy_type', 0)}/20
  Authority:    {risk_breakdown.get('legal_authority', 0)}/15
  Scope:        {risk_breakdown.get('scope_breadth', 0)}/25

Analyst narrative (3-4 sentences):"""

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        msg    = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception as e:
        print(f"[Node: report] Claude narrative error: {e}")
        return (
            f"This {fp.get('policy_type', 'policy action')} was verified against "
            f"whitehouse.gov (confidence: {verification.get('confidence', 0)}/100) "
            f"and carries a {risk_label} risk score of {risk_score}/90."
        )


def _regex_fallback(news_text: str) -> dict:
    """Offline regex fingerprint fallback."""
    import re as _re
    text = news_text.lower()
    _LEGAL = [r"\bsection\s+232\b", r"\bsection\s+301\b", r"\bieepa\b", r"\bofac\b"]
    _TYPES = {"tariff": r"\btariff\b", "sanction": r"\bsanction\b",
              "ban": r"\bban\b", "executive order": r"\bexecutive\s+order\b"}
    _COUNTRIES  = ["china","canada","mexico","russia","iran","india","japan",
                   "germany","france","uk","brazil","vietnam","turkey","israel"]
    _COMMODITIES = ["steel","aluminum","oil","gas","semiconductor","chip",
                    "lumber","wheat","copper","lithium"]
    auth = ""
    for p in _LEGAL:
        m = _re.search(p, text, _re.IGNORECASE)
        if m:
            auth = m.group(0).title()
            break
    ptype = "policy action"
    for t, p in _TYPES.items():
        if _re.search(p, text, _re.IGNORECASE):
            ptype = t
            break
    return {
        "legal_authority": auth,
        "policy_type":     ptype,
        "scope":  ", ".join(c.title() for c in _COUNTRIES   if c in text),
        "target": ", ".join(c.title() for c in _COMMODITIES if c in text),
    }
