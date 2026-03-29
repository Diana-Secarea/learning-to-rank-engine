"""
label_generation_phase2.py — LLM Weak Label Generation (Phase 2)

Pipeline:
  1. Diverse query set  →  top-LABEL_K candidates per query
                           (hard filter + RRF fusion + structured score)
  2. (query, company) pairs  →  OpenAI Batch API  →  relevance label 0-3
  3. labels.jsonl  :  query_id | company_id | query | relevance_label

Notes:
  - Uses gpt-4o-mini (cheapest) for bulk classification.
  - OpenAI Batch API gives 50% cost reduction vs. real-time calls.
  - Resumes gracefully if a batch_id file is found on disk.
"""

from __future__ import annotations

import json
import time
import pathlib
import io

import numpy as np
import openai
from dotenv import load_dotenv
load_dotenv()

from solution import (
    RankingEngine,
    parse_intent,
    _passes_hard_filter,
    _compute_structured_score,
)

# ── Config ────────────────────────────────────────────────────────────────────

LABEL_MODEL  = "gpt-4o-mini"        # cheap 0-3 classification
MAX_EXAMPLES = 3_000                # cap total (query, company) pairs
LABEL_K      = 200                  # candidates fetched per query
RRF_K_VAL    = 60
LABELS_PATH  = pathlib.Path("labels.jsonl")
BATCH_ID_FILE = pathlib.Path(".label_batch_id.txt")

# ── Diverse Query Set ─────────────────────────────────────────────────────────

QUERIES: list[str] = [
    # Logistics & Supply Chain
    "Find logistics companies in Germany",
    "Trucking companies in the US with over 500 employees",
    "Supply chain management companies in Europe",
    "Freight forwarding companies in the Netherlands",
    "Last-mile delivery startups founded after 2015",
    "Warehousing and distribution companies in the DACH region",
    # Software & SaaS
    "B2B SaaS companies with annual revenue over $10M",
    "Public software companies with more than 1,000 employees",
    "Cloud infrastructure companies in the US",
    "Cybersecurity startups in Israel",
    "AI and machine learning companies in Europe",
    "Enterprise software companies founded before 2005",
    "Developer tools companies with fewer than 200 employees",
    # Fintech & Finance
    "Fintech companies competing with traditional banks in Europe",
    "Payment processing companies with over $50M revenue",
    "Insurance technology companies in the UK",
    "Wealth management software companies in the US",
    "Banking software providers in Germany",
    "Accounting software companies for small businesses",
    # Healthcare & Pharma
    "Healthcare IT companies in the United States",
    "Pharmaceutical companies with more than 5,000 employees",
    "Biotech startups founded after 2010",
    "Medical device manufacturers in Germany",
    "Digital health platforms in Europe",
    "Clinical data management software companies",
    # Manufacturing
    "Manufacturing companies in the DACH region founded before 2000",
    "Automotive parts manufacturers in Germany",
    "Aerospace companies with over 1,000 employees",
    "Food and beverage manufacturers in France",
    "Chemical companies in the Netherlands",
    "Industrial equipment manufacturers in Italy",
    # E-commerce & Retail
    "E-commerce companies with revenue over $100M",
    "B2C retail companies in the UK",
    "Online marketplace platforms in Europe",
    "Fashion e-commerce companies founded after 2010",
    "Subscription box companies in North America",
    # Energy
    "Renewable energy companies in Scandinavia",
    "Oil and gas companies in Norway",
    "Energy management software companies in Europe",
    "Clean technology startups in the US",
    "Solar panel manufacturers with over 500 employees",
    # Consulting & Professional Services
    "Management consulting firms with over 500 employees",
    "IT consulting companies in Germany",
    "Legal technology companies in the US",
    "HR software and recruiting companies in Europe",
    "Accounting firms with international presence",
    # Marketing & Media
    "Digital marketing agencies in the UK",
    "AdTech companies with more than 200 employees",
    "Media and publishing companies in France",
    "Content management software companies",
    # Real Estate & Construction
    "Real estate technology companies in the US",
    "Construction companies in Germany with over 1,000 employees",
    "Property management software companies",
    "Architecture and engineering firms in Europe",
    # Education
    "EdTech companies in Europe",
    "Online learning platforms founded after 2012",
    "Corporate training software companies in the US",
    # Complex / Multi-criteria
    "Enterprise companies in North America with revenue over $500M",
    "Startups in the DACH region with fewer than 100 employees",
    "Public technology companies with over 10,000 employees",
    "Private B2B companies in France with revenue between $10M and $100M",
    "Manufacturing companies founded between 1950 and 1980",
    "B2B software companies with annual revenue under $5M founded after 2018",
    "Large logistics companies publicly traded in Europe",
]

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a relevance-labeling assistant for a company search engine. "
    "Rate how well a company profile matches the user's search request. "
    "Reply with ONLY a single digit: 0, 1, 2, or 3. No explanation."
)


def _user_prompt(query: str, company_text: str) -> str:
    return (
        f"Search request: {query}\n\n"
        f"Company profile:\n{company_text}\n\n"
        "Relevance scale:\n"
        "  3 = Highly relevant   — strong match on industry, location, size\n"
        "  2 = Moderately relevant — matches most key criteria\n"
        "  1 = Slightly relevant  — partial match or tangential\n"
        "  0 = Not relevant       — wrong industry, location, or profile\n\n"
        "Reply with a single digit (0, 1, 2, or 3):"
    )


# ── Candidate Generation (RRF + structured score, variable k) ─────────────────

def get_top_candidates(
    engine: RankingEngine, query: str, k: int = LABEL_K
) -> list[tuple[int, float]]:
    """
    Hard filter → FAISS+BM25 RRF → structured score.
    Returns up to k (record_idx, structured_score) pairs.
    """
    q = parse_intent(query)
    filtered = [i for i, r in enumerate(engine.records) if _passes_hard_filter(r, q)]
    if not filtered:
        filtered = list(range(len(engine.records)))
    filtered_set = set(filtered)

    # FAISS
    q_emb = engine.embed_model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    scores_raw, indices_raw = engine.faiss_index.search(q_emb, len(engine.records))
    faiss_ranked = [
        (int(idx), float(sc))
        for idx, sc in zip(indices_raw[0], scores_raw[0])
        if int(idx) in filtered_set
    ]

    # BM25
    bm25_all = engine.bm25.get_scores(query.lower().split())
    bm25_ranked = sorted(
        [(i, float(bm25_all[i])) for i in filtered],
        key=lambda x: x[1], reverse=True
    )

    # RRF
    rrf: dict[int, float] = {}
    for rank, (idx, _) in enumerate(faiss_ranked, start=1):
        rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (RRF_K_VAL + rank)
    for rank, (idx, _) in enumerate(bm25_ranked, start=1):
        rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (RRF_K_VAL + rank)
    top_ids = sorted(rrf, key=lambda x: rrf[x], reverse=True)[:k]

    faiss_map = dict(faiss_ranked)
    bm25_map  = {idx: sc for idx, sc in bm25_ranked}
    candidates = [(idx, faiss_map.get(idx, 0.0), bm25_map.get(idx, 0.0)) for idx in top_ids]

    # Structured score
    bm25_vals = np.array([c[2] for c in candidates], dtype=np.float64)
    bm25_max  = bm25_vals.max()
    bm25_norm = bm25_vals / bm25_max if bm25_max > 0 else bm25_vals

    scored = [
        (idx, _compute_structured_score(engine.records[idx], fs, float(bn), q))
        for (idx, fs, _), bn in zip(candidates, bm25_norm)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


# ── OpenAI Batch API helpers ──────────────────────────────────────────────────

def build_batch_jsonl(
    pairs: list[tuple[str, int, str, str]]
) -> str:
    """Build JSONL string for OpenAI Batch API. pairs: [(query_id, company_id, query_text, company_text)]"""
    lines = []
    for qid, cid, qt, ct in pairs:
        lines.append(json.dumps({
            "custom_id": f"{qid}__{cid}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": LABEL_MODEL,
                "max_tokens": 4,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _user_prompt(qt, ct)},
                ],
            },
        }))
    return "\n".join(lines)


def submit_and_poll(
    client: openai.OpenAI, pairs: list[tuple[str, int, str, str]]
) -> dict[str, int]:
    """Submit batch, poll to completion, return {custom_id: label} dict."""

    # Resume if a previous batch is in progress
    if BATCH_ID_FILE.exists():
        batch_id = BATCH_ID_FILE.read_text().strip()
        print(f"Resuming batch {batch_id} ...", flush=True)
    else:
        jsonl_content = build_batch_jsonl(pairs)
        print(f"Uploading {len(pairs)} requests to OpenAI Batch API...", flush=True)
        file_obj = client.files.create(
            file=("batch_input.jsonl", io.BytesIO(jsonl_content.encode())),
            purpose="batch",
        )
        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        batch_id = batch.id
        BATCH_ID_FILE.write_text(batch_id)
        print(f"Batch ID: {batch_id}  (saved to {BATCH_ID_FILE})", flush=True)

    while True:
        try:
            batch = client.batches.retrieve(batch_id)
        except Exception as e:
            print(f"  [WARN] Poll error ({e}), retrying in 30s...", flush=True)
            time.sleep(30)
            continue
        c = batch.request_counts
        print(
            f"  {batch.status}  "
            f"total={c.total}  completed={c.completed}  failed={c.failed}",
            flush=True,
        )
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            break
        time.sleep(30)

    if batch.status != "completed":
        raise RuntimeError(f"Batch ended with status: {batch.status}")

    # Download results
    result_file = client.files.content(batch.output_file_id)
    labels: dict[str, int] = {}
    for line in result_file.text.splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        custom_id = item.get("custom_id", "")
        try:
            text = item["response"]["body"]["choices"][0]["message"]["content"].strip()
            if text in ("0", "1", "2", "3"):
                labels[custom_id] = int(text)
        except (KeyError, IndexError):
            continue

    BATCH_ID_FILE.unlink(missing_ok=True)
    print(f"Received {len(labels)} valid labels.", flush=True)
    return labels


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    engine = RankingEngine()
    client = openai.OpenAI()

    # ── Build candidate pairs ──
    print("Building candidate pairs...", flush=True)
    pairs: list[tuple[str, int, str, str]] = []
    seen: set[tuple[str, int]] = set()

    for q_idx, query in enumerate(QUERIES):
        if len(pairs) >= MAX_EXAMPLES:
            break
        qid = f"q{q_idx:04d}"
        top = get_top_candidates(engine, query, LABEL_K)
        added = 0
        for company_idx, _ in top:
            key = (qid, company_idx)
            if key in seen:
                continue
            seen.add(key)
            pairs.append((qid, company_idx, query, engine.texts[company_idx]))
            added += 1
            if len(pairs) >= MAX_EXAMPLES:
                break
        print(
            f"  [{q_idx+1:>2}/{len(QUERIES)}] {query[:55]!r:<57}  "
            f"+{added:>3} candidates  total={len(pairs):>6}",
            flush=True,
        )

    print(f"\nTotal pairs to label: {len(pairs)}", flush=True)

    # ── Submit in chunks (OpenAI Batch API max = 50k requests per batch) ──
    BATCH_CHUNK = 10_000
    all_labels: dict[str, int] = {}
    for i in range(0, len(pairs), BATCH_CHUNK):
        chunk = pairs[i : i + BATCH_CHUNK]
        print(f"\nChunk {i // BATCH_CHUNK + 1}: {len(chunk)} pairs", flush=True)
        labels = submit_and_poll(client, chunk)
        all_labels.update(labels)

    # ── Write labels.jsonl ──
    written = 0
    with LABELS_PATH.open("w") as f:
        for qid, cid, query, _ in pairs:
            custom_id = f"{qid}__{cid}"
            if custom_id not in all_labels:
                continue
            f.write(json.dumps({
                "query_id":        qid,
                "company_id":      cid,
                "query":           query,
                "relevance_label": all_labels[custom_id],
            }) + "\n")
            written += 1

    print(f"\nDone. {written} labels written to {LABELS_PATH}", flush=True)

    # Label distribution
    dist = {0: 0, 1: 0, 2: 0, 3: 0}
    for v in all_labels.values():
        dist[v] = dist.get(v, 0) + 1
    print("Label distribution:", dist, flush=True)


if __name__ == "__main__":
    main()
