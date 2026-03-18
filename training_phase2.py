"""
training_phase2.py — Learning-to-Rank Training (Phase 2)

Reads labels.jsonl produced by label_generation_phase2.py, builds a feature
matrix, and trains a LightGBM LambdaMART model.

Features:
  cosine_sim      — semantic similarity (query embedding · company embedding)
  bm25_norm       — BM25 score normalised per query
  industry_score  — NAICS-prefix + keyword match (0-1)
  location_score  — country / region match (0-1)
  size_score      — employee count proximity (0-1)
  recency         — normalised founding year (0-1)
  structured_score— weighted sum of the above (existing pipeline score)
  keyword_overlap — fraction of query tokens found in company text
  is_public       — 0/1
  has_employees   — data-completeness flag
  has_revenue     — data-completeness flag

Target : relevance_label  (0-3, weak labels from LLM)
Metric : NDCG@10

Output : ltr_model.txt  (LightGBM booster)
         feature_importance.json
"""

from __future__ import annotations

import json
import pathlib
import random
from collections import defaultdict

import lightgbm as lgb
import numpy as np
from sklearn.metrics import ndcg_score

from solution import (
    RankingEngine,
    parse_intent,
    _passes_hard_filter,
    _compute_structured_score,
    _score_industry,
    _score_location,
    _score_size,
    _score_recency,
)

# ── Config ────────────────────────────────────────────────────────────────────

LABELS_PATH   = pathlib.Path("labels.jsonl")
MODEL_PATH    = pathlib.Path("ltr_model.txt")
FI_PATH       = pathlib.Path("feature_importance.json")

TRAIN_FRAC    = 0.80     # fraction of queries used for training
RANDOM_SEED   = 42

FEATURE_NAMES = [
    "cosine_sim",
    "bm25_norm",
    "industry_score",
    "location_score",
    "size_score",
    "recency",
    "structured_score",
    "keyword_overlap",
    "is_public",
    "has_employees",
    "has_revenue",
]

# ── Feature Engineering ───────────────────────────────────────────────────────

def keyword_overlap(query: str, company_text: str) -> float:
    """Fraction of unique query tokens present in lowercased company text."""
    tokens = set(query.lower().split())
    if not tokens:
        return 0.0
    ct = company_text.lower()
    return sum(1 for t in tokens if t in ct) / len(tokens)


def build_feature_row(
    engine: RankingEngine,
    company_idx: int,
    query: str,
    query_embedding: np.ndarray,
    bm25_max: float,
) -> list[float]:
    """Return one feature vector (list of floats) for a (query, company) pair."""
    record = engine.records[company_idx]
    q      = parse_intent(query)

    # Cosine similarity (embeddings are L2-normalised → inner product == cosine)
    company_emb = engine.embeddings[company_idx]          # shape (dim,)
    cosine_raw  = float(np.dot(query_embedding, company_emb))
    cosine_01   = (cosine_raw + 1.0) / 2.0               # [-1,1] → [0,1]

    # BM25 (raw score, normalised by per-query max passed in)
    bm25_raw  = float(engine.bm25.get_scores(query.lower().split())[company_idx])
    bm25_norm = bm25_raw / bm25_max if bm25_max > 0 else 0.0

    # Structured sub-scores
    ind_score = _score_industry(record, q)
    loc_score = _score_location(record, q)
    siz_score = _score_size(record, q)
    rec_score = _score_recency(record)

    # Full structured score
    struct = _compute_structured_score(record, cosine_raw, bm25_norm, q)

    # Keyword overlap
    kw_ov = keyword_overlap(query, engine.texts[company_idx])

    # Data completeness / categorical
    pub       = 1.0 if record.get("is_public") else 0.0
    has_emp   = 1.0 if record.get("employee_count") is not None else 0.0
    has_rev   = 1.0 if record.get("revenue")        is not None else 0.0

    return [
        cosine_01,
        bm25_norm,
        ind_score,
        loc_score,
        siz_score,
        rec_score,
        struct,
        kw_ov,
        pub,
        has_emp,
        has_rev,
    ]


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_labels(path: pathlib.Path) -> dict[str, list[dict]]:
    """Load labels grouped by query_id → list of {company_id, query, relevance_label}."""
    groups: dict[str, list[dict]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            groups[rec["query_id"]].append(rec)
    return dict(groups)


# ── NDCG Evaluation ───────────────────────────────────────────────────────────

def evaluate_ndcg(
    model: lgb.Booster,
    X: np.ndarray,
    y: np.ndarray,
    groups: list[int],
    k: int = 10,
) -> float:
    """Compute mean NDCG@k across all queries."""
    preds  = model.predict(X)
    offset = 0
    ndcgs  = []
    for g in groups:
        true_rel = y[offset : offset + g]
        pred_rel = preds[offset : offset + g]
        if true_rel.max() == 0:
            offset += g
            continue
        # sklearn expects shape (n_samples, n_queries)
        ndcgs.append(ndcg_score([true_rel], [pred_rel], k=k))
        offset += g
    return float(np.mean(ndcgs)) if ndcgs else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not LABELS_PATH.exists():
        print(f"ERROR: {LABELS_PATH} not found. Run label_generation_phase2.py first.")
        return

    # ── Load engine (embeddings + BM25 + records) ──
    engine = RankingEngine()

    # ── Load labels ──
    print("Loading labels...", flush=True)
    groups_data = load_labels(LABELS_PATH)
    query_ids   = sorted(groups_data.keys())
    print(f"  {len(query_ids)} queries, "
          f"{sum(len(v) for v in groups_data.values())} total pairs", flush=True)

    # ── Train / val split (by query so no leakage) ──
    random.seed(RANDOM_SEED)
    random.shuffle(query_ids)
    split       = int(len(query_ids) * TRAIN_FRAC)
    train_qids  = set(query_ids[:split])
    val_qids    = set(query_ids[split:])

    # ── Build feature matrices ──
    print("Building feature matrix...", flush=True)

    def build_split(qids: set[str]) -> tuple[np.ndarray, np.ndarray, list[int]]:
        X_rows, y_rows, group_sizes = [], [], []
        for qid in sorted(qids):
            recs  = groups_data[qid]
            query = recs[0]["query"]

            # Encode query once per group
            q_emb = engine.embed_model.encode(
                [query], normalize_embeddings=True, convert_to_numpy=True
            ).astype(np.float32)[0]

            # BM25 max for this query (for normalisation)
            bm25_scores = engine.bm25.get_scores(query.lower().split())
            bm25_max    = float(bm25_scores.max()) if bm25_scores.max() > 0 else 1.0

            for item in recs:
                cid   = int(item["company_id"])
                label = int(item["relevance_label"])
                feats = build_feature_row(engine, cid, query, q_emb, bm25_max)
                X_rows.append(feats)
                y_rows.append(label)

            group_sizes.append(len(recs))

        X = np.array(X_rows,  dtype=np.float32)
        y = np.array(y_rows,  dtype=np.float32)
        return X, y, group_sizes

    X_train, y_train, g_train = build_split(train_qids)
    X_val,   y_val,   g_val   = build_split(val_qids)
    print(f"  Train: {X_train.shape}  Val: {X_val.shape}", flush=True)

    # ── LightGBM LambdaMART ──
    train_ds = lgb.Dataset(X_train, label=y_train, group=g_train,
                           feature_name=FEATURE_NAMES, free_raw_data=False)
    val_ds   = lgb.Dataset(X_val,   label=y_val,   group=g_val,
                           feature_name=FEATURE_NAMES, free_raw_data=False,
                           reference=train_ds)

    params = {
        "objective":        "lambdarank",
        "metric":           "ndcg",
        "ndcg_eval_at":     [1, 5, 10],
        "learning_rate":    0.05,
        "num_leaves":       63,
        "min_data_in_leaf": 5,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "lambda_l1":        0.1,
        "lambda_l2":        0.1,
        "label_gain":       [0, 1, 3, 7],  # gain for relevance levels 0-3
        "verbosity":        -1,
        "seed":             RANDOM_SEED,
    }

    print("Training LambdaMART...", flush=True)
    callbacks = [lgb.early_stopping(50, verbose=True), lgb.log_evaluation(25)]
    model = lgb.train(
        params,
        train_ds,
        num_boost_round=500,
        valid_sets=[train_ds, val_ds],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    # ── Evaluate ──
    ndcg_train = evaluate_ndcg(model, X_train, y_train, g_train, k=10)
    ndcg_val   = evaluate_ndcg(model, X_val,   y_val,   g_val,   k=10)
    print(f"\nNDCG@10  train={ndcg_train:.4f}  val={ndcg_val:.4f}", flush=True)

    # ── Feature importance ──
    fi = dict(zip(FEATURE_NAMES, model.feature_importance("gain").tolist()))
    fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
    print("\nFeature importance (gain):")
    for name, imp in fi_sorted.items():
        bar = "█" * int(imp / max(fi_sorted.values()) * 30)
        print(f"  {name:<20} {imp:>10.1f}  {bar}")

    FI_PATH.write_text(json.dumps(fi_sorted, indent=2))
    print(f"\nFeature importance saved to {FI_PATH}", flush=True)

    # ── Save model ──
    model.save_model(str(MODEL_PATH))
    print(f"Model saved to {MODEL_PATH}", flush=True)


if __name__ == "__main__":
    main()
