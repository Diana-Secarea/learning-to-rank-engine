"""
rank_phase2.py — LTR-powered ranking pipeline (Phase 2)

Replaces the cross-encoder re-ranking step with the trained LightGBM
LambdaMART model.

Pipeline:
  query
    → parse intent
    → hard filter
    → FAISS + BM25 + RRF  (top CANDIDATE_K)
    → feature engineering  (same 11 features used during training)
    → LightGBM LambdaMART  →  final top-N

Usage:
  python rank_phase2.py                  # interactive CLI
  python rank_phase2.py "some query"     # single query from argv
"""

from __future__ import annotations

import json
import pathlib
import sys
import textwrap

import lightgbm as lgb
import numpy as np

from solution import (
    RankingEngine,
    parse_intent,
    _passes_hard_filter,
    _compute_structured_score,
    _score_industry,
    _score_location,
    _score_size,
    _score_recency,
    CANDIDATE_K,
    FINAL_TOP,
)
from training_phase2 import build_feature_row, keyword_overlap

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH  = pathlib.Path("ltr_model.txt")
RRF_K_VAL   = 60

# ── Phase-2 Ranker ────────────────────────────────────────────────────────────

class Phase2Ranker:
    def __init__(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"{MODEL_PATH} not found — run training_phase2.py first."
            )
        print("Loading ranking engine...", flush=True)
        self.engine = RankingEngine()

        print(f"Loading LTR model from {MODEL_PATH}...", flush=True)
        self.model = lgb.Booster(model_file=str(MODEL_PATH))
        print("Ready.\n", flush=True)

    def rank(self, query: str, top_n: int = FINAL_TOP) -> list[dict]:
        q = parse_intent(query)

        # ── Stage 1: hard filter ──
        filtered = [
            i for i, r in enumerate(self.engine.records)
            if _passes_hard_filter(r, q)
        ]
        if not filtered:
            print("[WARN] Hard filter removed all companies — using full set.", flush=True)
            filtered = list(range(len(self.engine.records)))
        filtered_set = set(filtered)

        # ── Stage 2: FAISS + BM25 → RRF ──
        q_emb = self.engine.embed_model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)
        scores_raw, indices_raw = self.engine.faiss_index.search(
            q_emb, len(self.engine.records)
        )
        faiss_ranked = [
            (int(idx), float(sc))
            for idx, sc in zip(indices_raw[0], scores_raw[0])
            if int(idx) in filtered_set
        ]
        bm25_all = self.engine.bm25.get_scores(query.lower().split())
        bm25_ranked = sorted(
            [(i, float(bm25_all[i])) for i in filtered],
            key=lambda x: x[1], reverse=True,
        )
        rrf: dict[int, float] = {}
        for rank, (idx, _) in enumerate(faiss_ranked, start=1):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (RRF_K_VAL + rank)
        for rank, (idx, _) in enumerate(bm25_ranked, start=1):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (RRF_K_VAL + rank)
        candidate_ids = sorted(rrf, key=lambda x: rrf[x], reverse=True)[:CANDIDATE_K]

        if not candidate_ids:
            return []

        # ── Stage 3: feature engineering ──
        faiss_map = dict(faiss_ranked)
        bm25_max  = max((bm25_all[i] for i in candidate_ids), default=1.0)
        bm25_max  = float(bm25_max) if bm25_max > 0 else 1.0

        rows = []
        for idx in candidate_ids:
            feats = build_feature_row(
                self.engine,
                idx,
                query,
                q_emb[0],
                bm25_max,
            )
            rows.append(feats)

        X = np.array(rows, dtype=np.float32)

        # ── Stage 4: LightGBM re-rank ──
        ltr_scores = self.model.predict(X)

        ranked = sorted(
            zip(candidate_ids, ltr_scores.tolist()),
            key=lambda x: x[1], reverse=True,
        )[:top_n]

        # ── Build result dicts ──
        results = []
        for rank_pos, (idx, ltr_score) in enumerate(ranked, start=1):
            r     = self.engine.records[idx]
            addr  = r.get("address") or {}
            naics = r.get("primary_naics") or {}
            results.append({
                "rank":           rank_pos,
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
                "ltr_score":      round(float(ltr_score), 4),
            })
        return results


# ── Formatting ────────────────────────────────────────────────────────────────

def fmt_result(r: dict) -> str:
    emp   = f"emp={r['employees']:>7,}" if r["employees"] else "emp=      ?"
    rev   = f"rev=${r['revenue']/1e6:>7.1f}M" if r["revenue"] else "rev=         ?"
    pub   = "PUBLIC" if r["is_public"] else "private"
    naics = (r["naics_label"] or "")[:30]
    return (
        f"  {r['rank']:>2}. {(r['name'] or '?'):<36} [{r['country']:>2}] "
        f"{emp}  {rev}  {pub:<7}  ltr={r['ltr_score']:+.4f}\n"
        f"      {naics}\n"
        f"      {textwrap.shorten(r['description'] or '', 100)}"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ranker = Phase2Ranker()

    queries = sys.argv[1:] if len(sys.argv) > 1 else None

    if queries:
        for query in queries:
            _run_query(ranker, query)
    else:
        # Interactive loop
        while True:
            try:
                query = input("Search for companies (or 'exit'): ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not query or query.lower() in ("exit", "quit", "q"):
                break
            _run_query(ranker, query)


def _run_query(ranker: Phase2Ranker, query: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"Query : {query}")
    print("=" * 70)
    results = ranker.rank(query)
    if not results:
        print("  (no results)")
    else:
        for r in results:
            print(fmt_result(r))
    print()


if __name__ == "__main__":
    main()
