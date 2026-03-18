# learning-to-rank-engine
Classifier + ranker of companies

The first part that I find the most important is to take a look on the dataset that was offered. I can see the dataset is big, so we need to analyze it and pre-process it in order to be as accurate as possible for the process of ranking and re-ranking that we will do further. Therefore, I took a look on the dataset and made this observations:

DATA CLEANING
1.	Invalid JSON structure
Looking at the first line of the JSON file , I have identified that we are passing the address like this: "year_founded":null,"address":"{'country_code': 'ro', 'latitude': 44.4792186, 'longitude': 26.1045773, 'region_name': 'Bucharest', 'town': 'Bucharest'}
  Two things immediately looked wrong:
  - The value is a string (wrapped in "), not an object.
  - Inside that string, the quotes are single quotes ('), not double quotes. Valid JSON only uses double quotes — so
  this is Python repr() output, not JSON.
To JSON, this is perfectly valid — it's just a string value that happens to look like a dict. No error is raised. The bug is invisible unless you actually check isinstance(value, dict) after parsing. So if we write code like record['address']['country_code'] it would crash with TypeError: string indices must be integers. We need to change it .
Fix that I used  
‘ ‘ ‘ ‘ ‘ ‘ ‘ ‘ ‘ ‘ ‘ ‘ ‘ ‘
import ast
  def parse_dict_field(value):
      if isinstance(value, dict):
          return value
      if isinstance(value, str) and value.startswith('{'):
          return ast.literal_eval(value)  # handles Python single-quote dicts
      return value

  record['address'] = parse_dict_field(record['address'])
  record['primary_naics'] = parse_dict_field(record['primary_naics'])
‘ ‘ ‘ ‘ ‘ ‘ ‘ ‘ ‘ ‘ ‘ ‘ ‘ ‘
2.Duplicates (25 name duplicates, 13 website duplicates)
  Problem: Three distinct duplicate patterns were found:
  Proposed fix: Use website as the primary deduplication key (more stable and unique than name), with a tiered strategy.
Basically I have checked the data with this code:
“
import json
      from collections import defaultdict

      records = []
      with open('companies_clean.jsonl') as f:
          for line in f:
              records.append(json.loads(line))

      website_map = defaultdict(list)
      for r in records:
          w = (r.get('website') or '').lower().strip()
          if w:
              website_map[w].append(r)

      for website, recs in website_map.items():
          if len(recs) > 1:
              print(f'\n=== {website} ===')
              if recs[0] == recs[1]:
                  print('IDENTICAL records')
              else:
                  for key in recs[0]:
                      v1 = recs[0].get(key)
                      v2 = recs[1].get(key)
                      if v1 != v2:
                          print(f'  DIFF {key}: {v1} vs {v2}')
      "
And I realized that there any many companies that share the same website and the same name.
Some of them being:
 wavedragon.com: ['Wave Dragon', 'Wave Dragon']
     norhybrid.com: ['Norhybrid Renewables', 'Norhybrid Renewables']
     teslaoceanturbine.com: ['Tesla Ocean Turbine', 'Tesla Ocean Turbine']
     windainergy.com: ['windainergy', 'windainergy']
     cktl.se: ['Carita och Torbjörn', 'Carita och Torbjörn']
     nextfabrication.com: ['Next Fabrication', 'Next Fabrication']
     worldwidewind.no: ['World Wide Wind', 'World Wide Wind']
     cirkelenergi.dk: ['CIRKEL Energi', 'CIRKEL Energi']
     windspider.com: ['WindSpider', 'WindSpider']
     sol-ind.com: ['Sol Industries', 'Sol Industries']
     vindparken.se: ['Erikshester Vindpark', 'Erikshester Vindpark']
     teunhulst.nl: ['Logic-Energy', 'Logic-Energy']
     fredolsen1848.com: ['Fred. Olsen 1848', 'Fred. Olsen 1848']
Those are 100% identical records — every single field is the same. Pure copy-paste duplicates in the dataset. Safe to just drop one of each, no merging needed.
Therefore I have added this code :
def count_fields(record: dict) -> int:
    """Count non-null fields in a record."""
    return sum(1 for v in record.values() if v is not None)

def dedup_by_website(records: list) -> list:
    seen = {}
    no_website = []
    for r in records:
        key = (r.get("website") or "").lower().strip()
        if not key:
            no_website.append(r)
            continue
        if key not in seen:
            seen[key] = r
        else:
            if count_fields(r) > count_fields(seen[key]):
                seen[key] = r
    return list(seen.values()) + no_website

Then I have discovered that: Duplicate descriptions (7 pairs) — these are the same no-website records from earlier. Rögle Vindkraftpark, Enercon, MantaWind, etc. — identical descriptions, no website to dedup on. Your
   dedup_by_website skipped them because they have no website key. These are true duplicates that slipped through. Worth fixing. -> I added a fix in data_cleaning.py
cleaned = dedup_by_description(cleaned)

3.Missing Numerical Fields — employee_count (39.4%) and revenue (19.5%)
I have ran on the cleaned dataset this:

  missing_emp = sum(1 for r in records if r.get('employee_count') is None)
  print(f'Missing employee_count: {missing_emp} ({missing_emp/n*100:.1f}%)')

Just counted how many records have employee_count equal to None (null in JSON), divided by total records

And I found out that: 39% of companies have no employee_count, 19% have no revenue.
Cause of missing:
  Problem: These are potentially strong ranking signals but have significant missingness. Dropping records or using a global mean would distort rankings.
Now we will ask ourselves, do we need to guess a value based on the industry or do we treat them as unknown and leave them as such?
  Proposed strategies (in preference order):
  1. NAICS-group median imputation — impute missing employee_count and revenue with the median of companies sharing the same primary_naics.code. This is sector-aware and more accurate than a global median.
  2. Size-bucket encoding — encode employee_count as an ordinal bucket (micro/small/medium/large/enterprise) based on common thresholds (1–10, 11–50, 51–200, 201–1000, 1000+). Treat missing as its own unknown category — this avoids fabricating a value.
  3. Missingness indicator — always add a boolean employee_count_known / revenue_known feature alongside the imputed value, so the model can learn that imputed values are less reliable.
Basically, the problem will be for  - "fast-growing companies" — revenue is needed to infer growth. We will ask ourselves if it is relevant for a query that requires employee count or revenue.
We will leave it as such for the moment
4. secondary_naics — 97.7% Missing
  Problem: Only 11 of 477 records have a non-null secondary_naics. It cannot be used as a reliable signal.
  Proposed fix: Drop this field from any feature vector. It should not be imputed (no basis for doing so). Keep it in the raw record for display only.
The risk of keeping it: you might accidentally uprank or downrank those 11 companies based on a field that 97.7% of companies don't even have, which creates inconsistency.
For the moment we will drop it and we will see.

5.Noisy data treatment 
- Deduplication within lists: lowercased, strip whitespace, drop near-duplicates.
  - Length cap: cap core_offerings at ~10 items, target_markets at ~8, keeping the most specific/distinguishing ones
  (shorter, more concrete terms).
  - Normalization: lowercase and strip for consistent matching.

  def clean_list_field(items, max_items=10):
      if not items:
          return []
      seen, result = set(), []
      for item in items:
          norm = item.lower().strip()
          if norm not in seen:
              seen.add(norm)	
              result.append(item.strip())
      return result[:max_items]
FOR LATER: The length cap — this one is worth doing. 31 target markets is noise, not signal. When you embed a company's text, a wall of 31 generic market labels dilutes the meaningful ones. Capping keeps the embedding focused.
- core_offerings goes up to 18 items, target_markets up to 31 items
  - Zero actual duplicates within any single record's list
  So:
  - The deduplication part — solves a problem that doesn't exist in this dataset. Skip it.
  - The length cap — this one is worth doing. 31 target markets is noise, not signal. When you embed a company's text, a wall of 31 generic market labels dilutes the meaningful ones. Capping keeps the embedding focused.
  - The normalization (lowercase/strip) — only matters if you're doing exact string matching. If
  you're embedding, the model handles casing internally. Low value.
  My opinion: Only implement the length cap, skip the rest. And the cap only matters at embedding time  — you could apply it in solution.py right before building the embedding, rather than baking it into the cleaned data permanently. That way companies_clean.jsonl stays as complete as possible.
SOLUTION.py:
Now we can analyze the data and we see this as conclusion;
Reliable signals (always present):
  - description, primary_naics, business_model, target_markets, core_offerings — 100% coverage → your main text signals
  - address, is_public — 100% → hard filters that always work

Partially reliable:
  - revenue (82%), year_founded (73%) → usable as filters but need null fallback
  - employee_count (62%) → ~38% missing, soft filter only
We need to see how we will build the company text and how we will embed it. company_to_text() function — the function that converts a company record into a single rich string for embedding. This is the most important function in the whole pipeline because everything downstream depends on it.
The first step is to make the text embeddings. HOW?
Concatenate only the 100% reliable fields as the backbone, append partials if present:
  def company_to_text(record: dict) -> str:
      parts = []
      # Always present — backbone
      if record.get("operational_name"):
          parts.append(record["operational_name"])
      if isinstance(record.get("primary_naics"), dict):
          parts.append(f"Industry: {record['primary_naics']['label']}")
      if record.get("description"):
          parts.append(record["description"])  # richest signal
      if record.get("core_offerings"):
          parts.append(f"Offerings: {', '.join(record['core_offerings'])}")

      if record.get("target_markets"):
          parts.append(f"Markets: {', '.join(record['target_markets'])}")
      if record.get("business_model"):
          parts.append(f"Business model: {', '.join(record['business_model'])}")
      # Address — always present but structured
      addr = record.get("address", {})
      if isinstance(addr, dict):
          loc = ", ".join(filter(None, [addr.get("town"), addr.get("country_code","").upper()]))
          if loc:
              parts.append(f"Location: {loc}")

      # Partials — only if present, never placeholder text
      if record.get("employee_count"):
          parts.append(f"Employees: {int(record['employee_count'])}")
      if record.get("revenue"):
          parts.append(f"Revenue: ${record['revenue']:,.0f}")
      if record.get("year_founded"):
          parts.append(f"Founded: {int(record['year_founded'])}")
      if record.get("is_public") is True:
          parts.append("Public company")
      return ". ".join(parts)
  We will implement this rule: never write "employee_count: unknown" or similar. Missing = absent from text, not filled with noise.
---
Each line in companies_text.jsonl will be {"website": ..., "text": ...} — website as the key to join back to the original records later.
And yes, everything semantically useful is captured. The only fields skipped are:
  - latitude / longitude — coordinates mean nothing to a text embedding model
  - website — URLs carry no semantic signal
  - secondary_naics — dropped in cleaning (only 11/477 records had it)


The companies_text.jsonl file that text_to_embed.py can write via its __main__ block is not used by solution.py — it's a standalone inspection tool. solution.py imports company_to_text as a function and calls it inline, which is cleaner and keeps a single source of truth for how companies are represented as text.
1.	parse_intent(query)
The embedding model can't run hard filters — it just finds semantically similar text. So before any retrieval, we need to extract structured signals from the query to use in Stage 1 (hard filter) and Stage 3 (scoring).
  parse_intent reads the query and fills a Query dataclass:
  Query(
      country_codes    = {"de"},
      business_models  = {"Business-to-Business", "Software-as-a-Service"},
      naics_prefixes   = ["5112", "511210"],
      employee_min     = 500,
      employee_max     = None,
      is_public        = None,
      ...
  )
  That structured object is then used by:
  - Hard filter — "does this company have country_code = DE?" — boolean, no score
  - _score_industry() — "does the company's NAICS code start with 5112?"
  - _score_location() — "exact country match or just same region?"
  - _score_size() — "is employee_count >= 500? how far outside the range?"
  Without this step, you'd have to rely entirely on the embedding similarity to capture "Germany" or "500 employees" — but those are exact structured facts, not semantic concepts. Embeddings are great at meaning, bad at numbers and geography. parse_intent bridges the gap.
We will create some HARD FILTERS for Hybrid Retrival :
2.	Hard Filters(Qualitfication ) the filters will be country, inudstry NAICS< B2B, BW@C , employeee range, public/private. propose more filters. 
Why? To eliminate irrelevant companies before doing any expensive computation.
Hard filters are boolean strict. They do not count as score. 
The embedding + BM25 + cross-encoder are all blind to hard constraints. If the query says:

  ▎ "Public companies with more than 1,000 employees"

  The embedding might still return a 10-person private startup with a similar description — because semantically the text is close. The embedding doesn't know the difference between a company talking about enterprise clients vs actually being an enterprise.
  Hard filters cut those out for free (just Python comparisons, no model inference) before FAISS and BM25 ever run.
  Two benefits:
  1. Speed — less to search and score
  2. Accuracy — cross-encoder slots are precious (only 20). You don't want to waste them on a private company when the query explicitly said "public".
Boolean elimination over all 457 records. Missing field = benefit of doubt = pass
2.Candidate generation(semantic) precompute embeddings for company, embedding for query, FAISS/vector search, cosine similarity + B25 keyword -> top k candidates
FAISS (full index, filter after) + BM25 → RRF fusion → top 50
solution.py calls company_to_text directly on every record:
# solution.py line ~130
  self.texts: list[str] = [company_to_text(r) for r in self.records]
Then those strings are what gets embedded:
raw = self.embed_model.encode(self.texts, ...)
So the flow is:
  companies_clean.jsonl
          ↓
    company_to_text()          ← from text_to_embed.py
          ↓
    list of text strings
          ↓
    BAAI/bge-small-en-v1.5    ← SentenceTransformer
          ↓
    embeddings (cached .npy + .faiss)

3. Scoring layer give a structured scoring layer. score = 0.40·cosine + 0.20·bm25 + 0.15·industry + 0.10·location + 0.10·size + 0.05·recency -> gives the top 20
 important
is to normalize everything 0-1 and keep this as baseline. 
4.Re -ranking with a cross encoder :  cross-encoder/ms-marco-MiniLM-L-6-v2 on top 20 → final 10
- Embeddings cached to .cache/ keyed by SHA-256 of the JSONL + model name — auto-invalidates on data change
  - All 12+ NAICS industry keywords mapped to 2-digit sector prefixes for fuzzy matching
  - Full business_model vocabulary from the actual data
  - is_public=None / missing data always passes filters
The cross-encoder (ms-marco-MiniLM-L-6-v2) takes the top 20 candidates from Stage 3 (structured scoring), scores each (query, company_text) pair, and re-sorts them. The final output is sorted by ce_score, not the structured score — so the cross-encoder has the final say on ordering. You can see both stage3_score and ce_score in the returned result dicts.
Now make another file called main.py where it asks the user in the CLI for input for what he wants and then applies solution.py and then receives only the first company from the top offered by the solution.py. Run it with python main.py — it prompts for input, runs the full pipeline, and prints only the top result.
TRAINING DATA
Now we will do another script called label_generation_phase2.py wherw we will ask the llm for every (query, company of top 200 to generate query_id | company_id | relevancce_label we limit it at max 5k-20k examples, prompt: give this user request and company profile rate relevance fom 0 not relevant to 3 highly relevant . i want to use LLM to generate weak labels and boostrapping. then we will do  training_phase2.py we will make features: cosine similarity, industry match, size match, location match, keyword overlap etc and target: relevance_label
We will use claude-api
label_generation_phase2.py

  What it does:
  1. Runs 65 diverse queries covering all major industries, locations, and size profiles
  2. For each query: hard filter → FAISS+BM25 RRF → structured score → top-200 candidates
  3. Sends all (query, company) pairs to Claude via the Batches API (50% cheaper than real-time)
  4. Labels each pair 0–3 and writes labels.jsonl

  Key decisions:
  - Uses claude-haiku-4-5 (fast, cheap) since 0–3 classification is simple - Opus for higher quality
  - Caps at MAX_EXAMPLES = 10_000 (tune to taste, up to 20k)
  - Polls every 30s, saves batch ID to disk for recovery

  ---
  training_phase2.py

  Features engineered per (query, company) pair:

  ┌──────────────────┬──────────────────────────────────────────┐
  │     Feature      │               Description                │
  ├──────────────────┼──────────────────────────────────────────┤
  │ cosine_sim       │ Query · company embedding dot product    │
  ├──────────────────┼──────────────────────────────────────────┤
  │ bm25_norm        │ BM25 score normalised by per-query max   │
  ├──────────────────┼──────────────────────────────────────────┤
  │ industry_score   │ NAICS prefix + keyword match (0–1)       │
  ├──────────────────┼──────────────────────────────────────────┤
  │ location_score   │ Country/region match (0–1)               │
  ├──────────────────┼──────────────────────────────────────────┤
  │ size_score       │ Employee count proximity (0–1)           │
  ├──────────────────┼──────────────────────────────────────────┤
  │ recency          │ Normalised founding year                 │
  ├──────────────────┼──────────────────────────────────────────┤
  │ structured_score │ Weighted combination (existing pipeline) │
  ├──────────────────┼──────────────────────────────────────────┤
  │ keyword_overlap  │ Fraction of query tokens in company text │
  ├──────────────────┼──────────────────────────────────────────┤
  │ is_public        │ Binary flag                              │
  ├──────────────────┼──────────────────────────────────────────┤
  │ has_employees    │ Data completeness flag                   │
  ├──────────────────┼──────────────────────────────────────────┤
  │ has_revenue      │ Data completeness flag                   │
  └──────────────────┴──────────────────────────────────────────┘

  Model: LightGBM LambdaMART with label_gain=[0,1,3,7] (exponential gain matching 0–3 relevance scale), 80/20 query-wise
   train/val split, early stopping on NDCG@10. Outputs ltr_model.txt + feature_importance.json.

  Run order:
  pip install -r requirements.txt
  python label_generation_phase2.py   # generates labels.jsonl
  python training_phase2.py           # trains + saves ltr_model.txt 
  Pipeline — same first two stages as solution.py, then diverges at re-ranking:

  query
    → parse intent
    → hard filter                       (same as solution.py)
    → FAISS + BM25 + RRF → top-50       (same as solution.py)
    → feature engineering (11 features) ← training_phase2.py logic
    → LightGBM LambdaMART predict
    → top-N sorted by ltr_score

  Key design decisions:
  - Imports build_feature_row and keyword_overlap directly from training_phase2.py — no duplication of feature logic
  - Drops the cross-encoder (ms-marco-MiniLM) entirely and replaces it with the trained LTR model — much faster at
  inference
  - Result dicts have ltr_score instead of ce_score so it's clear which model scored them

  Run order:
  python label_generation_phase2.py   # 1. generate labels
  python training_phase2.py           # 2. train ltr_model.txt
  python rank_phase2.py               # 3. use it interactively
- main.py — uses solution.py's full pipeline: FAISS+BM25+RRF → structured score → cross-encoder re-rank → top-1 result
  - rank_phase2.py — uses the same FAISS+BM25+RRF → then LightGBM LambdaMART re-rank → top-10 results (interactive loop)

  The only difference is the final re-ranking step. If you want rank_phase2.py to also show just the top-1 result like mian.py

