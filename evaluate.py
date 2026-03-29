"""
evaluate.py — End-to-end evaluation of the ranking pipeline.

Metrics computed
────────────────
  Retrieval quality  (LLM-judged ground truth, independent of pipeline reranker)
    NDCG@5, NDCG@10          – graded relevance (0-3 judge scale)
    MAP@10                    – binary: judge_score >= 2 counts as relevant
    MRR                       – position of first result with judge_score >= 2
    P@1, P@3, P@5, P@10      – precision at K (binary threshold >= 2)
    ROC-AUC                   – area under ROC; binary labels vs pipeline scores,
                                evaluated on the full stage-3 top-20 candidate pool

  Label types
    graded     (0-3)          – raw LLM judge scores
    binary     (score >= 2)   – standard relevant/not-relevant
    strict     (score == 3)   – highly-relevant only

  Ablations compared per query
    llm        – full pipeline with LLM reranker (gpt-4o-mini)
    ce         – cross-encoder only  (ms-marco-MiniLM, no LLM call)
    stage3     – structured score only (no neural reranker)

Efficiency & overhead
    Per-stage wall time, total latency, estimated API token cost

Usage
    python evaluate.py [--queries N] [--out eval_results.json]
    python evaluate.py --queries 5       # quick smoke-test (5 queries)
    python evaluate.py                   # full 12-query suite
"""

from __future__ import annotations

import argparse
import json
import math
import re
import os
import sys
import time
import tracemalloc

from dotenv import load_dotenv
load_dotenv()
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from solution import (
    FINAL_TOP,
    LLM_MODEL,
    RERANK_TOP,
    RankingEngine,
    _fmt_result,
    parse_intent,
    _passes_hard_filter,
)
from dataclasses import replace

# ── Evaluation query set ──────────────────────────────────────────────────────

EVAL_QUERIES: list[str] = [
    "Find logistics companies in Germany",
    "Public software companies with more than 1,000 employees",
    "B2B SaaS companies with annual revenue over $10M",
    "Fast-growing fintech companies competing with traditional banks in Europe",
    "Manufacturing companies in the DACH region founded before 2000",
    "Logistic companies in Romania",
    "Food and beverage manufacturers in France",
    "Companies that could supply packaging materials for a direct-to-consumer cosmetics brand",
    "Pharmaceutical companies in Switzerland",
    "B2B SaaS companies providing HR solutions in Europe",
    "Clean energy startups founded after 2018 with fewer than 200 employees",
    "Renewable energy equipment manufacturers in Scandinavia",
]

# ── Relevance thresholds ──────────────────────────────────────────────────────

BINARY_THRESHOLD = 2   # judge_score >= this → relevant
STRICT_THRESHOLD = 3   # judge_score == 3    → highly relevant

# ── Estimated API costs (USD per 1K tokens, as of 2025) ──────────────────────

COST_PER_1K_IN  = {"gpt-4o-mini": 0.00015, "gpt-4o": 0.0025}
COST_PER_1K_OUT = {"gpt-4o-mini": 0.00060, "gpt-4o": 0.01000}

JUDGE_MODEL = "gpt-4o"   # independent from pipeline reranker (gpt-4o-mini)


# ── Metrics ───────────────────────────────────────────────────────────────────

def dcg_at_k(relevances: list[float], k: int) -> float:
    """Discounted Cumulative Gain at position k (graded relevance)."""
    total = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        total += (2**rel - 1) / math.log2(i + 1)
    return total


def ndcg_at_k(relevances: list[float], k: int) -> float:
    """Normalised DCG@k. Returns 0.0 if ideal DCG is 0."""
    ideal = dcg_at_k(sorted(relevances, reverse=True), k)
    if ideal == 0.0:
        return 0.0
    return dcg_at_k(relevances, k) / ideal


def average_precision(relevances: list[float], threshold: float = BINARY_THRESHOLD) -> float:
    """Average Precision (for MAP). Binary labels derived from threshold."""
    hits = 0
    total_ap = 0.0
    for i, rel in enumerate(relevances, start=1):
        if rel >= threshold:
            hits += 1
            total_ap += hits / i
    n_relevant = sum(1 for r in relevances if r >= threshold)
    if n_relevant == 0:
        return 0.0
    return total_ap / n_relevant


def mean_reciprocal_rank(relevances: list[float], threshold: float = BINARY_THRESHOLD) -> float:
    """Reciprocal rank of first relevant result."""
    for i, rel in enumerate(relevances, start=1):
        if rel >= threshold:
            return 1.0 / i
    return 0.0


def precision_at_k(relevances: list[float], k: int, threshold: float = BINARY_THRESHOLD) -> float:
    """Fraction of top-k results that are relevant."""
    top = relevances[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if r >= threshold) / k


def roc_auc_score(y_true: list[int], y_score: list[float]) -> float:
    """
    ROC-AUC via trapezoidal rule.

    y_true:  binary labels (0/1)
    y_score: continuous scores (higher = more likely relevant)

    Returns 0.5 (random) if only one class is present.
    """
    if len(set(y_true)) < 2:
        return float("nan")   # undefined with a single class

    paired = sorted(zip(y_score, y_true), key=lambda x: x[0], reverse=True)
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos

    tp, fp = 0, 0
    prev_tp, prev_fp = 0, 0
    auc = 0.0

    for score, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos if n_pos else 0
        fpr = fp / n_neg if n_neg else 0
        prev_tpr = prev_tp / n_pos if n_pos else 0
        prev_fpr = prev_fp / n_neg if n_neg else 0
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        prev_tp, prev_fp = tp, fp

    return auc


# ── LLM Judge (Claude, independent of pipeline reranker) ─────────────────────

@dataclass
class JudgeResult:
    scores: dict[str, int]   # company name → 0-3
    input_tokens: int = 0
    output_tokens: int = 0
    latency: float = 0.0
    model: str = JUDGE_MODEL


def llm_judge(query: str, candidates: list[dict], engine: RankingEngine) -> JudgeResult:
    """
    Score each candidate's relevance to the query using an OpenAI model as an independent judge.

    Returns JudgeResult with scores dict keyed by company name.
    Falls back to a simple heuristic if the OpenAI API is unavailable.
    """
    import openai

    lines: list[str] = []
    for i, r in enumerate(candidates, start=1):
        name  = r.get("name") or "?"
        cc    = r.get("country") or ""
        ind   = r.get("naics_label") or ""
        bm    = ", ".join(r.get("business_model") or [])
        desc  = (r.get("description") or "")[:250]
        lines.append(f'{i}. {name} [{cc}] | {ind} | {bm}\n   {desc}')

    companies_block = "\n\n".join(lines)
    prompt = (
        f'Query: "{query}"\n\n'
        f"You are an independent relevance judge. "
        f"Score each company 0-3 based on how well it matches the query.\n\n"
        f"Scoring scale:\n"
        f"  3 = Highly relevant — clearly matches all key criteria\n"
        f"  2 = Relevant — matches the main intent with minor gaps\n"
        f"  1 = Tangential — partial or indirect match\n"
        f"  0 = Not relevant\n\n"
        f"Strictness rules:\n"
        f"  - A pharma company is NOT a 'manufacturing company' unless pharma is explicitly requested.\n"
        f"  - A cosmetics brand is NOT a packaging supplier.\n"
        f"  - Match industry, geography, size, and business model implied by the query.\n\n"
        f"Companies:\n{companies_block}\n\n"
        f"Return ONLY a JSON array, no explanation, no markdown:\n"
        f'[{{"id": 1, "score": 0}}, {{"id": 2, "score": 3}}, ...]'
    )

    t0 = time.perf_counter()
    try:
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            max_tokens=512,
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a precise relevance judge. Return only valid JSON arrays."},
                {"role": "user", "content": prompt},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if not m:
            raise ValueError(f"No JSON array in judge response: {raw[:200]}")
        scores_list = json.loads(m.group(0))

        name_map = {i + 1: (c.get("name") or "") for i, c in enumerate(candidates)}
        scores = {name_map[item["id"]]: int(item["score"]) for item in scores_list if item["id"] in name_map}

        return JudgeResult(
            scores=scores,
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            latency=round(time.perf_counter() - t0, 3),
            model=JUDGE_MODEL,
        )

    except Exception as exc:
        print(f"  [JUDGE WARN] {exc} — using stage3 score as proxy", flush=True)
        # Fallback: use stage3_score rescaled to 0-3 as a proxy for the judge
        scores = {
            (c.get("name") or ""): round(c.get("stage3_score", 0.0) * 3)
            for c in candidates
        }
        return JudgeResult(scores=scores, latency=round(time.perf_counter() - t0, 3))


# ── Instrumented engine ────────────────────────────────────────────────────────

@dataclass
class StageOutput:
    n_after_filter: int = 0
    n_candidates: int = 0
    stage3_pool: list[tuple[int, float]] = field(default_factory=list)
    results_llm: list[dict] = field(default_factory=list)
    results_ce:  list[dict] = field(default_factory=list)
    results_stage3: list[dict] = field(default_factory=list)
    timing: dict[str, float] = field(default_factory=dict)
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0


class EvalEngine(RankingEngine):
    """RankingEngine with timing instrumentation and per-ablation result exposure."""

    def _stage3_to_results(self, q, top_stage3: list[tuple[int, float]]) -> list[dict]:
        """Convert stage3 (idx, score) pairs to result dicts without neural reranking."""
        results = []
        for rank, (idx, s3) in enumerate(top_stage3[:FINAL_TOP], start=1):
            r    = self.records[idx]
            addr = r.get("address") or {}
            naics = r.get("primary_naics") or {}
            results.append({
                "rank":           rank,
                "name":           r.get("operational_name"),
                "website":        r.get("website"),
                "country":        (addr.get("country_code") or "").upper(),
                "region":         addr.get("region_name"),
                "employees":      int(r["employee_count"]) if r.get("employee_count") else None,
                "revenue":        r.get("revenue"),
                "naics_code":     naics.get("code") if isinstance(naics, dict) else None,
                "naics_label":    naics.get("label") if isinstance(naics, dict) else None,
                "year_founded":   int(r["year_founded"]) if r.get("year_founded") else None,
                "is_public":      r.get("is_public"),
                "business_model": r.get("business_model"),
                "description":    r.get("description"),
                "stage3_score":   round(float(s3), 4),
                "ce_score":       round(float(s3), 4),   # same score, no extra reranker
            })
        return results

    def _rerank_ce_full(self, q, top_stage3: list[tuple[int, float]]) -> list[dict]:
        """Cross-encoder on all stage3 candidates; returns full RERANK_TOP list."""
        if not top_stage3:
            return []
        pairs    = [(q.raw, self.texts[idx]) for idx, _ in top_stage3]
        ce_scores = self.cross_encoder.predict(pairs)
        merged = sorted(
            [(idx, s3, float(ce)) for (idx, s3), ce in zip(top_stage3, ce_scores)],
            key=lambda x: x[2], reverse=True,
        )
        results = []
        for rank, (idx, s3, ce) in enumerate(merged[:FINAL_TOP], start=1):
            r    = self.records[idx]
            addr = r.get("address") or {}
            naics = r.get("primary_naics") or {}
            results.append({
                "rank":           rank,
                "name":           r.get("operational_name"),
                "website":        r.get("website"),
                "country":        (addr.get("country_code") or "").upper(),
                "region":         addr.get("region_name"),
                "employees":      int(r["employee_count"]) if r.get("employee_count") else None,
                "revenue":        r.get("revenue"),
                "naics_code":     naics.get("code") if isinstance(naics, dict) else None,
                "naics_label":    naics.get("label") if isinstance(naics, dict) else None,
                "year_founded":   int(r["year_founded"]) if r.get("year_founded") else None,
                "is_public":      r.get("is_public"),
                "business_model": r.get("business_model"),
                "description":    r.get("description"),
                "stage3_score":   round(float(s3), 4),
                "ce_score":       round(float(ce), 4),
            })
        return results

    def _rerank_llm_full(
        self, q, top_stage3: list[tuple[int, float]]
    ) -> tuple[list[dict], int, int]:
        """
        LLM reranker that returns ALL scored candidates (not just top-10).
        Returns (results, input_tokens, output_tokens).
        """
        if not top_stage3:
            return [], 0, 0

        import openai
        client = openai.OpenAI()

        lines: list[str] = []
        for i, (idx, _) in enumerate(top_stage3, start=1):
            r    = self.records[idx]
            name = r.get("operational_name") or "?"
            addr = r.get("address") or {}
            cc   = (addr.get("country_code") or "").upper()
            naics = r.get("primary_naics") or {}
            ind  = naics.get("label", "") if isinstance(naics, dict) else ""
            bm   = ", ".join(r.get("business_model") or [])
            desc = (r.get("description") or "")[:200]
            lines.append(f"{i}. {name} [{cc}] | {ind} | {bm}\n   {desc}")

        prompt = (
            f'Query: "{q.raw}"\n\n'
            f"Score each company 0-3 (0=not relevant, 1=tangential, 2=relevant, 3=highly relevant).\n"
            f"Be strict about industry, geography, size, and business model.\n\n"
            f"Companies:\n" + "\n\n".join(lines) +
            f"\n\nReturn ONLY a JSON array:\n"
            f'[{{"id": 1, "score": 0}}, {{"id": 2, "score": 3}}, ...]'
        )

        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                max_tokens=512,
                temperature=0,
                messages=[
                    {"role": "system", "content": "You are a precise company qualification system. Return only valid JSON arrays."},
                    {"role": "user",   "content": prompt},
                ],
            )
            raw = resp.choices[0].message.content.strip()
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            if not m:
                raise ValueError(f"No JSON array: {raw[:200]}")
            scores_list = json.loads(m.group(0))
            scores_map  = {item["id"]: float(item["score"]) for item in scores_list}
            in_tok  = resp.usage.prompt_tokens
            out_tok = resp.usage.completion_tokens
        except Exception as exc:
            print(f"  [LLM WARN] {exc}", flush=True)
            scores_map, in_tok, out_tok = {}, 0, 0

        merged = sorted(
            [(idx, s3, scores_map.get(i + 1, 0.0)) for i, (idx, s3) in enumerate(top_stage3)],
            key=lambda x: (x[2], x[1]), reverse=True,
        )

        results = []
        for rank, (idx, s3, llm_s) in enumerate(merged[:FINAL_TOP], start=1):
            r    = self.records[idx]
            addr = r.get("address") or {}
            naics = r.get("primary_naics") or {}
            results.append({
                "rank":           rank,
                "name":           r.get("operational_name"),
                "website":        r.get("website"),
                "country":        (addr.get("country_code") or "").upper(),
                "region":         addr.get("region_name"),
                "employees":      int(r["employee_count"]) if r.get("employee_count") else None,
                "revenue":        r.get("revenue"),
                "naics_code":     naics.get("code") if isinstance(naics, dict) else None,
                "naics_label":    naics.get("label") if isinstance(naics, dict) else None,
                "year_founded":   int(r["year_founded"]) if r.get("year_founded") else None,
                "is_public":      r.get("is_public"),
                "business_model": r.get("business_model"),
                "description":    r.get("description"),
                "stage3_score":   round(float(s3), 4),
                "ce_score":       round(float(llm_s), 4),   # 0-3 LLM score
                "llm_score":      round(float(llm_s), 4),
            })
        return results, in_tok, out_tok

    def eval_rank(self, query: str) -> StageOutput:
        """
        Run all three pipeline variants and return timing + intermediate results.
        """
        out = StageOutput()
        q   = parse_intent(query)

        # Stage 0 — Query expansion
        t0 = time.perf_counter()
        q.expanded = self._expand_query(query)
        out.timing["expand"] = round(time.perf_counter() - t0, 3)

        # Stage 1 — Hard filter
        t1 = time.perf_counter()
        filtered = self._hard_filter_tiered(q)
        out.timing["filter"] = round(time.perf_counter() - t1, 3)
        out.n_after_filter = len(filtered)

        # Stage 2 — FAISS + BM25 → RRF
        t2 = time.perf_counter()
        candidates = self._candidate_gen(q, filtered)
        out.timing["cand"] = round(time.perf_counter() - t2, 3)
        out.n_candidates = len(candidates)

        if not candidates:
            return out

        # Stage 3 — Structured score
        t3 = time.perf_counter()
        top_stage3 = self._structured_score(q, candidates)
        out.timing["struct"] = round(time.perf_counter() - t3, 3)
        out.stage3_pool = top_stage3

        # Stage3-only results (no neural reranker)
        out.results_stage3 = self._stage3_to_results(q, top_stage3)

        # Stage 4a — Cross-encoder
        t4 = time.perf_counter()
        out.results_ce = self._rerank_ce_full(q, top_stage3)
        out.timing["ce"] = round(time.perf_counter() - t4, 3)

        # Stage 4b — LLM reranker
        t5 = time.perf_counter()
        out.results_llm, out.llm_input_tokens, out.llm_output_tokens = (
            self._rerank_llm_full(q, top_stage3)
        )
        out.timing["llm"] = round(time.perf_counter() - t5, 3)

        out.timing["total"] = round(sum(out.timing.values()), 3)
        return out


# ── Per-query evaluation ───────────────────────────────────────────────────────

@dataclass
class QueryMetrics:
    query: str
    n_filtered: int
    n_candidates: int
    timing: dict[str, float]
    judge: JudgeResult
    llm_api_tokens: dict[str, int]

    # Metrics per ablation: {ablation_name: {metric_name: value}}
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    # ROC-AUC per ablation (evaluated on full stage3 pool)
    roc_auc: dict[str, float] = field(default_factory=dict)
    # Raw ranked scores for each ablation
    ranked_judge_scores: dict[str, list[float]] = field(default_factory=dict)


def evaluate_query(engine: EvalEngine, query: str) -> QueryMetrics:
    print(f"\n{'─'*60}", flush=True)
    print(f"Query: {query}", flush=True)

    stage = engine.eval_rank(query)

    # Get judge labels for all result candidates (use LLM results as the main set)
    # We evaluate ROC-AUC on the stage3_pool (up to RERANK_TOP candidates)
    pool_results = stage.results_stage3  # same set of companies as stage3 pool, structured

    print(f"  Candidates: {stage.n_candidates} | Stage3 pool: {len(stage.stage3_pool)}", flush=True)

    judge = llm_judge(query, pool_results, engine)
    print(
        f"  Judge: {len(judge.scores)} scores | "
        f"{judge.input_tokens}in/{judge.output_tokens}out tokens | "
        f"{judge.latency}s",
        flush=True,
    )

    def get_judge_scores_for(results: list[dict]) -> list[float]:
        return [float(judge.scores.get(r.get("name") or "", 0)) for r in results]

    def compute_metrics(results: list[dict]) -> dict[str, float]:
        graded  = get_judge_scores_for(results)
        binary  = [1 if s >= BINARY_THRESHOLD else 0 for s in graded]
        strict  = [1 if s == STRICT_THRESHOLD else 0 for s in graded]

        return {
            "ndcg@5":    round(ndcg_at_k(graded, 5), 4),
            "ndcg@10":   round(ndcg_at_k(graded, 10), 4),
            "map@10":    round(average_precision(graded), 4),
            "mrr":       round(mean_reciprocal_rank(graded), 4),
            "p@1":       round(precision_at_k(graded, 1), 4),
            "p@3":       round(precision_at_k(graded, 3), 4),
            "p@5":       round(precision_at_k(graded, 5), 4),
            "p@10":      round(precision_at_k(graded, 10), 4),
            "p@1_strict":  round(precision_at_k(graded, 1, STRICT_THRESHOLD), 4),
            "p@3_strict":  round(precision_at_k(graded, 3, STRICT_THRESHOLD), 4),
            "n_relevant":  sum(binary),
            "n_strict":    sum(strict),
            "mean_judge_score": round(float(np.mean(graded)) if graded else 0.0, 4),
        }

    def compute_roc_auc(results: list[dict], score_key: str) -> float:
        graded = get_judge_scores_for(results)
        y_true = [1 if s >= BINARY_THRESHOLD else 0 for s in graded]
        y_score = [r.get(score_key, 0.0) or 0.0 for r in results]
        return round(roc_auc_score(y_true, y_score), 4)

    ablations = {
        "llm":    stage.results_llm,
        "ce":     stage.results_ce,
        "stage3": stage.results_stage3,
    }

    metrics_out: dict[str, dict[str, float]] = {}
    roc_out: dict[str, float] = {}
    ranked_judge: dict[str, list[float]] = {}

    for name, results in ablations.items():
        metrics_out[name] = compute_metrics(results)
        roc_out[name] = compute_roc_auc(
            results,
            score_key="llm_score" if name == "llm" else "ce_score",
        )
        ranked_judge[name] = get_judge_scores_for(results)

        print(
            f"  [{name:6s}] NDCG@10={metrics_out[name]['ndcg@10']:.3f} "
            f"MAP={metrics_out[name]['map@10']:.3f} "
            f"MRR={metrics_out[name]['mrr']:.3f} "
            f"P@5={metrics_out[name]['p@5']:.3f} "
            f"ROC-AUC={roc_out[name]:.3f}",
            flush=True,
        )

    print(
        f"  Timing: expand={stage.timing.get('expand',0):.2f}s "
        f"filter={stage.timing.get('filter',0):.2f}s "
        f"cand={stage.timing.get('cand',0):.2f}s "
        f"struct={stage.timing.get('struct',0):.2f}s "
        f"ce={stage.timing.get('ce',0):.2f}s "
        f"llm={stage.timing.get('llm',0):.2f}s "
        f"total={stage.timing.get('total',0):.2f}s",
        flush=True,
    )

    return QueryMetrics(
        query=query,
        n_filtered=stage.n_after_filter,
        n_candidates=stage.n_candidates,
        timing=stage.timing,
        judge=judge,
        llm_api_tokens={
            "reranker_in":  stage.llm_input_tokens,
            "reranker_out": stage.llm_output_tokens,
            "judge_in":     judge.input_tokens,
            "judge_out":    judge.output_tokens,
        },
        metrics=metrics_out,
        roc_auc=roc_out,
        ranked_judge_scores=ranked_judge,
    )


# ── Aggregate report ───────────────────────────────────────────────────────────

def aggregate_metrics(all_qm: list[QueryMetrics]) -> dict[str, Any]:
    ablations = ["llm", "ce", "stage3"]
    metric_keys = [
        "ndcg@5", "ndcg@10", "map@10", "mrr",
        "p@1", "p@3", "p@5", "p@10",
        "p@1_strict", "p@3_strict",
        "mean_judge_score",
    ]

    agg: dict[str, Any] = {}
    for abl in ablations:
        agg[abl] = {}
        for mk in metric_keys:
            vals = [qm.metrics[abl][mk] for qm in all_qm if abl in qm.metrics and mk in qm.metrics[abl]]
            agg[abl][mk] = round(float(np.mean(vals)), 4) if vals else float("nan")
        roc_vals = [qm.roc_auc[abl] for qm in all_qm if abl in qm.roc_auc and not math.isnan(qm.roc_auc[abl])]
        agg[abl]["roc_auc"] = round(float(np.mean(roc_vals)), 4) if roc_vals else float("nan")

    return agg


def efficiency_report(all_qm: list[QueryMetrics]) -> dict[str, Any]:
    stage_keys = ["expand", "filter", "cand", "struct", "ce", "llm", "total"]
    avg_timing = {}
    for k in stage_keys:
        vals = [qm.timing.get(k, 0.0) for qm in all_qm]
        avg_timing[k] = round(float(np.mean(vals)), 3)

    # API cost estimate
    total_reranker_in  = sum(qm.llm_api_tokens.get("reranker_in", 0)  for qm in all_qm)
    total_reranker_out = sum(qm.llm_api_tokens.get("reranker_out", 0) for qm in all_qm)
    total_judge_in     = sum(qm.llm_api_tokens.get("judge_in", 0)     for qm in all_qm)
    total_judge_out    = sum(qm.llm_api_tokens.get("judge_out", 0)    for qm in all_qm)

    reranker_cost = (
        total_reranker_in  / 1000 * COST_PER_1K_IN.get(LLM_MODEL, 0)
      + total_reranker_out / 1000 * COST_PER_1K_OUT.get(LLM_MODEL, 0)
    )
    judge_cost = (
        total_judge_in  / 1000 * COST_PER_1K_IN.get(JUDGE_MODEL, 0)
      + total_judge_out / 1000 * COST_PER_1K_OUT.get(JUDGE_MODEL, 0)
    )

    return {
        "n_queries": len(all_qm),
        "avg_latency_per_stage": avg_timing,
        "avg_total_latency": avg_timing.get("total", 0.0),
        "tokens": {
            "reranker_in":  total_reranker_in,
            "reranker_out": total_reranker_out,
            "judge_in":     total_judge_in,
            "judge_out":    total_judge_out,
        },
        "estimated_cost_usd": {
            "reranker": round(reranker_cost, 5),
            "judge":    round(judge_cost, 5),
            "total":    round(reranker_cost + judge_cost, 5),
            "per_query": round((reranker_cost + judge_cost) / max(len(all_qm), 1), 5),
        },
    }


def complexity_report(engine: EvalEngine) -> dict[str, Any]:
    """Static characteristics of the pipeline implementation."""
    n_records   = len(engine.records)
    emb_dim     = engine.embeddings.shape[1] if hasattr(engine, "embeddings") else 0
    index_size  = engine.embeddings.nbytes if hasattr(engine, "embeddings") else 0

    return {
        "pipeline_stages": 5,
        "stage_descriptions": [
            "Intent Parser (spaCy + regex)",
            "Hard Filter (tiered, 4 fallback levels)",
            "FAISS + BM25 → RRF (dual-signal candidate generation)",
            "Structured Scoring (6-signal weighted combination)",
            "LLM Reranker (gpt-4o-mini, batched, 1 API call)",
        ],
        "models_loaded": [
            f"Embedding: BAAI/bge-small-en-v1.5 (dim={emb_dim})",
            "Cross-encoder: ms-marco-MiniLM-L-6-v2 (fallback reranker)",
            f"LLM: {LLM_MODEL} (primary reranker + query expansion)",
            f"Judge: {JUDGE_MODEL} (evaluation only)",
        ],
        "index": {
            "type": "FAISS IndexFlatIP",
            "n_vectors": n_records,
            "embedding_dim": emb_dim,
            "index_size_mb": round(index_size / 1024 / 1024, 2),
        },
        "api_calls_per_query": {
            "query_expansion": "1 (cached on repeat)",
            "llm_reranker": "1 (batched, constant regardless of corpus size)",
            "total_new_query": 2,
            "total_cached_query": 1,
        },
        "scoring_weights": {
            "cosine_faiss": 0.40,
            "bm25_keyword": 0.20,
            "industry_naics": 0.15,
            "location": 0.10,
            "size": 0.10,
            "recency": 0.05,
        },
        "data": {
            "corpus_size": n_records,
            "candidate_k_rrf": 50,
            "rerank_top": 20,
            "final_top": 10,
        },
    }


def overhead_report(engine_load_time: float, tracemalloc_peak_mb: float) -> dict[str, Any]:
    return {
        "engine_load_time_s": round(engine_load_time, 2),
        "peak_memory_mb": round(tracemalloc_peak_mb, 1),
        "implementation_notes": [
            "Embeddings cached to disk (SHA-256 keyed) — rebuild only on data change",
            "Query expansions cached in memory per session — 0 cost on repeat queries",
            "BM25 is in-memory (rank_bm25); must be rebuilt on restart (O(n) at 457 records)",
            "FAISS IndexFlatIP: exact search, O(n*d), no approximation overhead",
            "Cross-encoder runs locally — no API cost, ~100ms on CPU for 20 pairs",
            "LLM reranker: 1 API call per query, ~1-2s network latency",
        ],
    }


# ── Pretty printer ─────────────────────────────────────────────────────────────

def print_report(
    all_qm: list[QueryMetrics],
    agg: dict,
    eff: dict,
    cplx: dict,
    ovhd: dict,
) -> None:
    W = 70
    def hr(): print("─" * W)
    def hdr(title): hr(); print(f"  {title}"); hr()

    hdr("AGGREGATE METRICS  (mean over all evaluated queries)")

    ablation_labels = {"llm": "LLM reranker", "ce": "Cross-encoder", "stage3": "Stage3 score only"}
    metric_display = [
        ("NDCG@5",       "ndcg@5"),
        ("NDCG@10",      "ndcg@10"),
        ("MAP@10",       "map@10"),
        ("MRR",          "mrr"),
        ("P@1",          "p@1"),
        ("P@3",          "p@3"),
        ("P@5",          "p@5"),
        ("P@10",         "p@10"),
        ("P@1 (strict)", "p@1_strict"),
        ("P@3 (strict)", "p@3_strict"),
        ("ROC-AUC",      "roc_auc"),
        ("Avg judge",    "mean_judge_score"),
    ]

    header = f"{'Metric':<16}" + "".join(f"{ablation_labels[a]:>18}" for a in ["llm", "ce", "stage3"])
    print(header)
    print("·" * len(header))
    for label, key in metric_display:
        row = f"{label:<16}"
        for abl in ["llm", "ce", "stage3"]:
            v = agg[abl].get(key, float("nan"))
            row += f"{v:>18.4f}"
        print(row)

    hdr("PER-QUERY RESULTS  (LLM reranker | judge scores at each rank)")
    for qm in all_qm:
        print(f"\n  Q: {qm.query[:60]}")
        print(f"     Filter: {qm.n_filtered} → Candidates: {qm.n_candidates}")
        graded = qm.ranked_judge_scores.get("llm", [])
        scores_str = " ".join(
            f"[{'■' if s >= BINARY_THRESHOLD else '·'}{int(s)}]"
            for s in graded[:10]
        )
        print(f"     Judge scores (ranks 1-10): {scores_str}")
        m = qm.metrics.get("llm", {})
        print(
            f"     NDCG@10={m.get('ndcg@10',0):.3f}  "
            f"MAP={m.get('map@10',0):.3f}  "
            f"MRR={m.get('mrr',0):.3f}  "
            f"ROC-AUC={qm.roc_auc.get('llm',float('nan')):.3f}"
        )

    hdr("EFFICIENCY")
    t = eff["avg_latency_per_stage"]
    print(f"  Average latency per query: {eff['avg_total_latency']:.2f}s")
    print(f"    Query expansion : {t.get('expand', 0):.2f}s (1 LLM call, cached on repeat)")
    print(f"    Hard filter     : {t.get('filter', 0):.3f}s (linear scan, O(n))")
    print(f"    FAISS + BM25    : {t.get('cand',   0):.3f}s (exact search + BM25 scoring)")
    print(f"    Structured score: {t.get('struct',  0):.3f}s (6-signal weighted combination)")
    print(f"    Cross-encoder   : {t.get('ce',     0):.2f}s (20 pairs, local inference)")
    print(f"    LLM reranker    : {t.get('llm',    0):.2f}s (1 batched API call)")
    print()
    c = eff["estimated_cost_usd"]
    print(f"  API cost over {eff['n_queries']} queries:")
    print(f"    Pipeline reranker ({LLM_MODEL}): ${c['reranker']:.5f}")
    print(f"    Eval judge ({JUDGE_MODEL}):  ${c['judge']:.5f}")
    print(f"    Total                                : ${c['total']:.5f}")
    print(f"    Per query                            : ${c['per_query']:.5f}")
    tok = eff["tokens"]
    print(f"  Tokens used: {tok['reranker_in']+tok['reranker_out']} reranker, {tok['judge_in']+tok['judge_out']} judge")

    hdr("COMPLEXITY")
    print(f"  Pipeline stages : {cplx['pipeline_stages']}")
    for i, s in enumerate(cplx["stage_descriptions"], start=1):
        print(f"    {i}. {s}")
    print()
    print(f"  Models loaded:")
    for m in cplx["models_loaded"]:
        print(f"    - {m}")
    print()
    idx = cplx["index"]
    print(f"  Vector index    : {idx['type']}, {idx['n_vectors']} vectors, dim={idx['embedding_dim']}, {idx['index_size_mb']} MB")
    d = cplx["data"]
    print(f"  Corpus          : {d['corpus_size']} companies → RRF top-{d['candidate_k_rrf']} → rerank top-{d['rerank_top']} → return top-{d['final_top']}")
    api = cplx["api_calls_per_query"]
    print(f"  API calls/query : {api['total_new_query']} (new) / {api['total_cached_query']} (cached expansion)")
    print()
    print(f"  Scoring weights :")
    for sig, w in cplx["scoring_weights"].items():
        bar = "█" * int(w * 40)
        print(f"    {sig:<18} {w:.2f}  {bar}")

    hdr("IMPLEMENTATION OVERHEAD")
    print(f"  Engine load time : {ovhd['engine_load_time_s']:.1f}s")
    print(f"  Peak memory      : {ovhd['peak_memory_mb']:.0f} MB")
    print()
    for note in ovhd["implementation_notes"]:
        print(f"  • {note}")

    hr()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the ranking pipeline")
    parser.add_argument("--queries", type=int, default=len(EVAL_QUERIES),
                        help=f"Number of queries to evaluate (default: {len(EVAL_QUERIES)})")
    parser.add_argument("--out", type=str, default=None,
                        help="Path to write JSON results (optional)")
    args = parser.parse_args()

    queries = EVAL_QUERIES[: args.queries]

    print(f"\nEvaluating {len(queries)} queries")
    print(f"  Pipeline reranker : {LLM_MODEL}")
    print(f"  Evaluation judge  : {JUDGE_MODEL}  (independent)")
    print(f"  Ablations         : LLM reranker | Cross-encoder | Stage3 score\n")

    # Load engine (measure overhead)
    tracemalloc.start()
    t_load = time.perf_counter()
    engine = EvalEngine()
    load_time = time.perf_counter() - t_load
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak_mem / 1024 / 1024

    print(f"Engine loaded in {load_time:.1f}s  |  peak memory {peak_mb:.0f} MB\n")

    # Run evaluation
    all_qm: list[QueryMetrics] = []
    for q in queries:
        qm = evaluate_query(engine, q)
        all_qm.append(qm)

    # Aggregate
    agg  = aggregate_metrics(all_qm)
    eff  = efficiency_report(all_qm)
    cplx = complexity_report(engine)
    ovhd = overhead_report(load_time, peak_mb)

    # Print report
    print("\n\n")
    print_report(all_qm, agg, eff, cplx, ovhd)

    # Write JSON output
    if args.out:
        output = {
            "queries_evaluated": len(all_qm),
            "aggregate_metrics": agg,
            "efficiency": eff,
            "complexity": cplx,
            "overhead": ovhd,
            "per_query": [
                {
                    "query": qm.query,
                    "n_filtered": qm.n_filtered,
                    "n_candidates": qm.n_candidates,
                    "timing": qm.timing,
                    "metrics": qm.metrics,
                    "roc_auc": qm.roc_auc,
                    "judge_scores_ranked": qm.ranked_judge_scores,
                    "llm_api_tokens": qm.llm_api_tokens,
                }
                for qm in all_qm
            ],
        }
        with open(args.out, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
