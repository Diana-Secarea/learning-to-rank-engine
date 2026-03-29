# WRITEUP — Company Ranking & Qualification System


The first part that I find the most important is to take a look on the dataset that was offered. I can see the dataset is big, so we need to analyze it and pre-process it in order to be as accurate as possible for the process of ranking and re-ranking that we will do further. Therefore, I took a look on the dataset and made this observations:

## data_cleaning.py

The first part that I find the most important is to take a look on the dataset that was offered. I can see the dataset is big, so we need to analyze it and pre-process it in order to be as accurate as possible for the process of ranking and re-ranking that we will do further. Therefore, I took a look on the dataset and made this observations:

Before the pipeline can run, `companies.jsonl` needs to be cleaned. This script takes the raw file and produces `companies_clean.jsonl` in four passes:

**Pass 1 — Fix invalid JSON structure (`parse_dict_field`)**

Fields like `address` and `primary_naics` are stored as Python `repr()` output instead of real JSON objects:

```
"address": "{'country_code': 'ro', 'latitude': 44.47, ...}"
```

JSON parses this as a plain string, so `record['address']['country_code']` raises `TypeError: string indices must be integers`. The fix uses `ast.literal_eval()` to safely convert these Python-style dict strings back to real dicts.
Looking at the first line of the JSON file , I have identified that we are passing the address like this: "year_founded":null,"address":"{'country_code': 'ro', 'latitude': 44.4792186, 'longitude': 26.1045773, 'region_name': 'Bucharest', 'town': 'Bucharest'}
Two things immediately looked wrong:
  - The value is a string (wrapped in "), not an object.
  - Inside that string, the quotes are single quotes ('), not double quotes. Valid JSON only uses double quotes.
To JSON, this is perfectly valid — it's just a string value that happens to look like a dict. No error is raised. The bug is invisible unless you actually check isinstance(value, dict) after parsing. So if we write code like record['address']['country_code'] it would crash with TypeError: string indices must be integers. We need to change it .


**Pass 2 — Deduplicate by website (`dedup_by_website`)**

13 websites appear exactly twice with byte-for-byte identical records. Website is used as the dedup key — it's more stable and unique than company name. If two records share a website, the one with more non-null fields is kept. Records with no website are passed through unchanged.
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
Those are 100% identical records — every single field is the same. Pure copy-paste duplicates in the dataset. Safe to just drop one of each, no merging needed.

**Pass 3 — Deduplicate by description (`dedup_by_description`)**

7 companies have no website and identical descriptions — Rögle Vindkraftpark, Enercon, MantaWind, etc. These slipped through the website dedup. A second pass uses description as the fallback key to catch them.

**Pass 4 — Drop `secondary_naics` (`drop_feature_fields`)**

Only 11 of 464 records have a non-null `secondary_naics` (~2.3%). Keeping it in ranking logic would create inconsistency — it would influence only those 11 companies. The field is removed from every record entirely.
The risk of keeping it: we might accidentally uprank or downrank those 11 companies based on a field that 97.7% of companies don't even have, which creates inconsistency.

**Running it:**

```bash
python data_cleaning.py companies.jsonl companies_clean.jsonl
# Output: Cleaned 477 records, removed 20 duplicates → 457 records saved to companies_clean.jsonl
```

**Pass 5 : Missing Numerical Field - employee_count and revenue**
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
We will leave it as such for the moment.

---

## Development Notes

The first part that I find the most important is to take a look on the dataset that was offered. I can see the dataset is big, so we need to analyze it and pre-process it in order to be as accurate as possible for the process of ranking and re-ranking that we will do further. Therefore, I took a look on the dataset and made this observations:

### DATA CLEANING

**1. Invalid JSON structure**

Looking at the first line of the JSON file, I have identified that we are passing the address like this:

```
"year_founded":null,"address":"{'country_code': 'ro', 'latitude': 44.4792186, 'longitude': 26.1045773, 'region_name': 'Bucharest', 'town': 'Bucharest'}"
```

Two things immediately looked wrong:
- The value is a string (wrapped in `"`), not an object.
- Inside that string, the quotes are single quotes (`'`), not double quotes. Valid JSON only uses double quotes — so this is Python `repr()` output, not JSON.

To JSON, this is perfectly valid — it's just a string value that happens to look like a dict. No error is raised. The bug is invisible unless you actually check `isinstance(value, dict)` after parsing. So if we write code like `record['address']['country_code']` it would crash with `TypeError: string indices must be integers`. We need to change it.

Fix that I used:

```python
import ast

def parse_dict_field(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.startswith('{'):
        return ast.literal_eval(value)  # handles Python single-quote dicts
    return value

record['address'] = parse_dict_field(record['address'])
record['primary_naics'] = parse_dict_field(record['primary_naics'])
```

**2. Duplicates (25 name duplicates, 13 website duplicates)**

I checked the data by grouping records by website and comparing fields. I realized there are many companies that share the same website and the same name:

```
wavedragon.com:       ['Wave Dragon', 'Wave Dragon']
norhybrid.com:        ['Norhybrid Renewables', 'Norhybrid Renewables']
teslaoceanturbine.com:['Tesla Ocean Turbine', 'Tesla Ocean Turbine']
windainergy.com:      ['windainergy', 'windainergy']
cktl.se:              ['Carita och Torbjörn', 'Carita och Torbjörn']
nextfabrication.com:  ['Next Fabrication', 'Next Fabrication']
worldwidewind.no:     ['World Wide Wind', 'World Wide Wind']
cirkelenergi.dk:      ['CIRKEL Energi', 'CIRKEL Energi']
windspider.com:       ['WindSpider', 'WindSpider']
sol-ind.com:          ['Sol Industries', 'Sol Industries']
vindparken.se:        ['Erikshester Vindpark', 'Erikshester Vindpark']
teunhulst.nl:         ['Logic-Energy', 'Logic-Energy']
fredolsen1848.com:    ['Fred. Olsen 1848', 'Fred. Olsen 1848']
```

Those are 100% identical records — every single field is the same. Pure copy-paste duplicates in the dataset. Safe to just drop one of each, no merging needed. I used website as the primary dedup key, keeping the record with more non-null fields:

```python
def count_fields(record: dict) -> int:
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
```

Then I discovered a second problem: 7 pairs of duplicate descriptions — Rögle Vindkraftpark, Enercon, MantaWind, etc. — identical descriptions, no website to dedup on. `dedup_by_website` skipped them because they have no website key. Added `dedup_by_description` as a second pass in `data_cleaning.py`.

**3. Missing Numerical Fields — employee_count (39.4%) and revenue (19.5%)**

```python
missing_emp = sum(1 for r in records if r.get('employee_count') is None)
print(f'Missing employee_count: {missing_emp} ({missing_emp/n*100:.1f}%)')
```

39% of companies have no employee_count, 19% have no revenue. These are potentially strong ranking signals but have significant missingness. Dropping records or using a global mean would distort rankings.

Strategies I considered:
1. NAICS-group median imputation — impute missing values with the median of companies sharing the same `primary_naics.code`. Sector-aware and more accurate than a global median.
2. Size-bucket encoding — encode employee_count as an ordinal bucket (micro/small/medium/large/enterprise). Treat missing as its own unknown category.
3. Missingness indicator — add a boolean `employee_count_known` alongside the imputed value so the model can learn that imputed values are less reliable.

Decision: leave as-is for now. The hard filter already handles missing = benefit of doubt, and the LLM reranker can reason about company size from descriptions.

**4. secondary_naics — 97.7% Missing**

Only 11 of 477 records have a non-null `secondary_naics`. It cannot be used as a reliable signal. Dropped it from any feature vector — keeping it would accidentally uprank or downrank those 11 companies based on a field 97.7% of companies don't have.

**5. Noisy list fields**

`core_offerings` goes up to 18 items, `target_markets` up to 31 items. A wall of 31 generic market labels dilutes the meaningful ones when embedding. I only implemented the length cap (applied at embedding time in `solution.py`, not baked into `companies_clean.jsonl`), skipped deduplication within lists (zero actual duplicates existed) and normalization (the embedding model handles casing internally).

---

### SOLUTION.py — Field coverage analysis

After cleaning, the field reliability picture is:

**Reliable signals (always present):**
- `description`, `primary_naics`, `business_model`, `target_markets`, `core_offerings` — 100% coverage → main text signals
- `address`, `is_public` — 100% → hard filters that always work

**Partially reliable:**
- `revenue` (82%), `year_founded` (73%) → usable as filters but need null fallback
- `employee_count` (62%) → ~38% missing, soft filter only

The `company_to_text()` function is the most important function in the whole pipeline because everything downstream depends on it. Rule: never write `"employee_count: unknown"` or similar. Missing = absent from text, not filled with noise. The only fields skipped entirely are latitude/longitude (coordinates mean nothing to a text model), website (URLs carry no semantic signal), and secondary_naics (dropped in cleaning).

---

### What actually works in the pipeline

After running the first version I verified what was solid before touching anything:

- Pipeline flow (filter → FAISS+BM25+RRF → structured score → reranker) is sound. Stages feed each other correctly.
- Hard filter applies benefit-of-doubt correctly — missing data always passes, only contradicting data is rejected.
- Weights sum to exactly 1.0. Double checked.
- RRF fusion formula is correctly implemented: `score = 1/(60 + rank_faiss) + 1/(60 + rank_bm25)`
- FAISS using IndexFlatIP with normalized embeddings is correct for cosine similarity.
- Cache invalidation via SHA-256 of the data file is solid — changing the data automatically triggers re-embedding.
- BM25 normalization handles the edge case where max score is 0.

---

### Problems I found after first run

Running it against all 12 test queries made the issues obvious:

**1. The dataset is the main bottleneck.** 457 companies, heavily biased toward HR/software. Queries like "logistics companies in Germany" simply don't have good answers in the data — the system can't return what isn't there.

**2. Romanian company at rank 1 for a Germany query.** CFR Romania kept appearing as the top result for "Find logistics companies in Germany". Technically the closest match by NAICS code, but it's a Romanian company. Any evaluator would flag this.

**3. Hard filter had no fallback.** When no companies passed the full filter, the old code returned the full unfiltered corpus — 457 random companies dumped as results. That's worse than returning nothing.

**4. Duplicates in results.** Rompetrol appeared twice in the Romania logistics query because it had two records in the dataset. The dedup only happened at data cleaning time, not at result time.

**5. Regex intent parsing is brittle.** The query "fast-growing fintech companies competing with traditional banks" was setting `year_founded_max = 2000` because it found the word "traditional". It didn't understand that "traditional" was describing the *banks*, not the companies being searched for.

**6. Cross-encoder doesn't understand industry semantics.** The ms-marco model was trained on web passage retrieval. It has no concept of "is pharma a SaaS company?" For the B2B SaaS query it was ranking pharma companies at positions 2–10 because their descriptions scored high on cosine similarity.

---

### Fix 1 — Deduplication at result time

Simple fix, worked perfectly. Q6 (Logistics companies in Romania): Rompetrol now appears once. Result count dropped from 10 to 9, ranks re-numbered correctly.

---

### Fix 2 — Tiered fallback for hard filter

Instead of dumping all 457 records when nothing passes the filter, I added 4 tiers:
1. Full filter (geography + industry + size + year)
2. Relax geography → show industry matches globally
3. Relax geography + industry → show by size/type only
4. Full unfiltered corpus (last resort, with a clear warning printed)

Q1 (Find logistics companies in Germany): now prints `[NOTE] No companies found in the specified region — showing industry matches globally` and returns only 1 result — CFR Romania. Which is honestly correct: there is only 1 logistics company globally in this dataset. The message is honest instead of silently returning junk.

---

### Fix 3 — spaCy for intent parsing

The regex parser sees a flat list of words with no idea of grammatical context. For "fast-growing fintech companies competing with traditional banks in Europe", regex finds "traditional" and fires `year_founded_max = 2000`. It doesn't know "traditional" belongs to "banks", not to "companies".

I switched to spaCy dependency parsing. spaCy builds a parse tree for the sentence:

```
Fast-growing    head=companies     dep=amod        ← modifier of the TARGET
fintech         head=companies     dep=compound
companies       head=companies     dep=ROOT
competing       head=companies     dep=acl
with            head=competing     dep=prep
traditional     head=banks         dep=amod        ← modifier of BANKS, not target
banks           head=with          dep=pobj
```

The fix: find any "context verb" (competing, disrupting, replacing, vs) and exclude its entire subtree from filter extraction. `traditional` and `banks` are 3 levels deep in the `competing` subtree → excluded. `Fast-growing` modifies `companies` directly → kept.

This also fixed the contradictory case: "startups disrupting traditional banks" was setting both `year_founded_min=2010` (from "startups") AND `year_founded_max=2000` (from "traditional") at the same time. With dependency parsing, only "startups" fires.

---

### Fix 4 — Clean energy NAICS codes

The phrase "clean energy" was falling through to the generic `energy` keyword which mapped to oil/gas NAICS codes. There were also no renewable energy NAICS codes in the map at all.

I added explicit entries for `clean energy`, `renewable energy`, `wind`, `solar` mapped to the correct NAICS codes (221114–221118 for electric power, 333611 for wind turbines, 334413 for solar cells).

Result for Q12 (Clean energy startups): Energy-Technik (Egyptian engineering, wrong) dropped out, VASOL TECH (Swedish solar) appeared. Q15 (Renewable energy Scandinavia): Home Energy [SE] appeared at rank 5 — a Swedish renewable energy equipment company that wasn't in the top 10 before.

---

### Fix 5 — LLM reranker replacing the cross-encoder

The cross-encoder problem was clear from the B2B SaaS query: pharma companies at ranks 2–10. The ms-marco model doesn't understand "SaaS" as a business model — it just sees dense text that's semantically close to the query.

I replaced the cross-encoder with a single gpt-4o-mini call. All 20 candidates go into one prompt, the LLM scores them 0–3, done. Two API calls total per query (one for expansion, one for reranking), regardless of dataset size. The ms-marco cross-encoder is kept as fallback.

**Before (cross-encoder):** Q3 B2B SaaS results 2–10 were Acino, OM Pharma, CordenPharma, Ferring — all pharma.

**After (LLM reranker):** Personio, NOVRH, quarksUp, Lucca, BambooHR appear at ranks 1–5. All score 3.

---

### Fix 6 — Query expansion

The raw query "companies supplying packaging for cosmetics brands" embeds near *cosmetics companies* — because they share the most vocabulary. FAISS was finding L'Oréal and similar brands instead of packaging suppliers.

The fix: before FAISS search, an LLM rewrites the query as a description of the ideal company. The rewritten query embeds near packaging manufacturers instead of cosmetics brands. BM25 still uses the original raw query because keyword matching benefits from the user's exact terms.

Results where expansion helped:
- **Q3 (B2B SaaS)**: BambooHR surfaced at rank 6 — it was in the dataset but the raw query didn't embed near it.
- **Q4 (Fintech Europe)**: Aircash (Croatian mobile wallet) appeared at rank 8 — previously invisible to FAISS.
- **Q5 (DACH manufacturing)**: The expansion correctly described automotive/aerospace manufacturers — but FAISS still returned Swiss pharma because that's all DACH manufacturing there is in the dataset. The expansion did its job, the data just doesn't have the right companies.

Query expansion is one extra API call per query, cached on repeat. It surfaces 1–2 new correct companies per query on average without hurting anything.


## 3.1 Approach

### Architecture overview

The system is a five-stage pipeline. Each stage is cheaper and faster than the one after it, so the expensive work is only done on a small, already-filtered candidate set. I created this method by experimenting different days with cross-encoders, LLms, hard filters and other methods that made sense once I combined all together. 

```
User query
    │
    ▼
[1] Intent Parser          — extract structured signals (country, size, industry, public/private, year)
    │
    ▼
[2] Hard Filter            — boolean elimination; missing data = benefit of the doubt = pass
    │                        tiered fallback: relax geography first, then NAICS
    ▼
[3] FAISS + BM25 → RRF     — top-50 candidates by semantic + keyword recall
    │
    ▼
[4] Query Expansion        — LLM rewrites query as a company description for FAISS
    │                        (runs before stage 3; cached per query)
    ▼
[5] LLM Reranker           — gpt-4o-mini scores all 50 candidates in one API call (0–3)
    │                        fallback to cross-encoder (ms-marco-MiniLM) on API failure
    ▼
Top-10 results, deduplicated
```

### Stage 1 — Intent Parser

Regex + spaCy extracts structured constraints from the natural language query:

- **Geography**: country names and region names (DACH, Europe, Nordics, etc.) → ISO-2 country codes
- **Size**: employee count ranges, qualitative hints ("startup" → max 200, "enterprise" → min 1000)
- **Revenue**: min/max thresholds from phrases like "revenue over $50M"
- **Public/private** flag
- **Business model**: B2B, SaaS, fintech, etc. mapped to canonical vocabulary
- **Industry**: query keywords mapped to NAICS prefix lists
- **Year founded**: explicit ranges or qualitative hints ("founded before 2000")

The spaCy dependency parser identifies context verbs ("competing with", "disrupting", "replacing") and excludes tokens in their subtrees from industry and year matching. This prevents "fast-growing fintech companies competing with traditional banks" from triggering banking NAICS codes or "traditional" from setting a year_founded_max of 2000.

Firstly I was using only rgeex but then I have realised that   Regex sees a flat list of words: [fast-growing, fintech, companies, competing, with, traditional, banks, in, europe]. It has no idea that traditional belongs to banks not to companies.
spaCy sees a tree and knows the grammatical ownership of every word. That's why it can correctly separate "what the user is searching for" from "what they mentioned as context".


### Stage 2 — Hard Filter

Why? To eliminate irrelevant companies before doing any expensive computation.
Hard filters are boolean strict. They do not count as score. 

Boolean elimination using the structured signals from stage 1. The rule is: **only reject a company if it has data AND that data contradicts the query**. Missing data always passes — a company without a recorded country code is not excluded from a Germany query.

A tiered fallback prevents the pipeline from returning garbage when no companies survive the full filter:
1. Full filter (geography + industry + size + year)
2. Relax geography → show industry matches globally
3. Relax geography + industry → show matches by size/type only
4. Full unfiltered corpus (last resort, labelled clearly)

NAICS is a conditional hard filter: it only applies when the query specifies both an industry keyword and a business model (e.g. "B2B SaaS"). A lone "manufacturing" keyword uses NAICS as a soft scoring signal rather than a hard gate, because stacking multiple hard filters on a 457-record dataset eliminates all candidates.

What we analysed with the hard filter:
  - Geography: rejects only if company has a country code AND it's not in the query's set
  - Employee count: rejects if outside [min, max] range
  - Revenue: same range check
  - Public/private: rejects only if field exists and doesn't match
  - Business model: rejects if company has model data and none match
  - NAICS: rejects if company has a code and it doesn't start with any queried prefix
  - Year founded: range check

I hva experimented a lot with the hard filter with many runs of solution.py, making me come up with a hard filter that is combined with a soft approach. SO even tho some criterias are hard, we do not want to eliminate companies just based on that and to loose possible good candidates.
Check the structure of them both:
 Hard Filter

Binary pass/fail — runs before any ML. It removes companies that definitely don't match. A company passes unless it has data that explicitly contradicts the query:

  - Geography: company's country code not in the query's target countries → rejected
  - Employee count: outside the query's min/max range → rejected
  - Revenue: outside the query's range → rejected
  - Public/private: company is public but query wants private → rejected
  - Business model: company's model (B2B, B2C...) has no overlap with query → rejected
  - NAICS industry: only applied as hard when industry + business model both signal it and no country or numeric
  constraints are already narrowing the pool (to avoid over-filtering the small dataset)
  - Founded year: outside min/max → rejected

  Key rule: missing data = pass. If a company has no employee_count, it won't be rejected for employee filters.

  Tiered fallback

  If the hard filter kills too many companies, it progressively relaxes:
  1. Full filter → if 0 results...
  2. Drop geography → if 0 results...
  3. Drop geography + industry → if 0 results...
  4. Return everything (last resort)

  This is why you saw [NOTE] No companies found in the specified region — showing industry matches globally. for the Germany logistics query — only 1 company survived tier 1.

  ---
  "Soft" Filter (Structured Score)

  There's no explicit "soft filter" — what acts as soft scoring is the Stage 3 structured score, a weighted sum of 6
  signals computed for each candidate that survived the hard filter:

  ┌────────────────┬────────┬───────────────────────────────────────┐
  │     Signal     │ Weight │           What it measures            │
  ├────────────────┼────────┼───────────────────────────────────────┤
  │ FAISS cosine   │ 0.40   │ Semantic similarity (embedding match) │
  ├────────────────┼────────┼───────────────────────────────────────┤
  │ BM25 keyword   │ 0.20   │ Keyword overlap                       │
  ├────────────────┼────────┼───────────────────────────────────────┤
  │ NAICS industry │ 0.15   │ Industry code match                   │
  ├────────────────┼────────┼───────────────────────────────────────┤
  │ Location       │ 0.10   │ Geographic proximity                  │
  ├────────────────┼────────┼───────────────────────────────────────┤
  │ Size           │ 0.10   │ Employee/revenue fit                  │
  ├────────────────┼────────┼───────────────────────────────────────┤
  │ Recency        │ 0.05   │ How recently founded                  │
  └────────────────┴────────┴───────────────────────────────────────┘

  So hard filter is a gate (in or out), and the structured score is a ranking signal — companies that partially match on
   geography or industry still get through but score lower.


### Stage 3 — FAISS + BM25 → RRF

Two complementary retrieval signals are fused using Reciprocal Rank Fusion (RRF):

- **FAISS** (IndexFlatIP on `bge-small-en-v1.5` embeddings) captures semantic similarity. Company texts are enriched with semantic size labels ("large enterprise"), revenue bands ("mid-market revenue"), and recency labels ("young startup") so the embedding model can match qualitative queries.
- **BM25** provides exact keyword recall that semantic search misses. A query containing "Shopify" will score companies mentioning it via BM25 even if the semantic embedding doesn't align.

RRF (with k=60) combines the two ranked lists without requiring score normalization. The top-50 candidates by RRF score proceed to reranking.
Runs two retrieval methods in parallel and fuses them:
FAISS (semantic): Encodes the raw query, does dot-product search over all normalized embeddings, filters to hard-filtered indices.
BM25 (keyword): Scores all filtered companies using BM25Okapi on tokenized text.

Reciprocal Rank Fusion (RRF): Combines both ranked lists:
  score(doc) = 1/(60 + rank_faiss) + 1/(60 + rank_bm25)
Returns top 50 candidates with their FAISS and BM25 raw scores.


### Stage 4 — Query Expansion

The raw query embeds near companies that *mention* the query topic, not necessarily companies that *are* what the query describes. The cosmetics packaging failure is the canonical example: "companies supplying packaging for cosmetics brands" embeds near cosmetics companies because they share the most vocabulary.

Before FAISS search, an LLM rewrites the query as a company description:

> *"Companies supplying packaging for cosmetics brands"*
> → *"A packaging manufacturer producing bottles, jars, tubes, and labels supplied to cosmetics and personal care brands. B2B supplier in the packaging and containers industry."*

This shifts the query embedding from the cosmetics space into the packaging/containers space. BM25 continues to use the original query (keyword matching benefits from the user's exact terms). Expansions are cached per query — repeated runs are free.
So NO, I am not taking all the database and parse every company to the LLM, I am just parsing the query so it can be better understood.


### Stage 5 — LLM Reranker

All 50 candidates are sent to `gpt-4o-mini` in a **single API call**. The model scores each company 0–3:

- **3** — clearly and directly satisfies the query intent
- **2** — relevant, matches main criteria with minor gaps
- **1** — tangential or borderline
- **0** — not relevant

Results are sorted by LLM score descending, with the structured score as tiebreaker within each score tier. A cross-encoder (`ms-marco-MiniLM-L-6-v2`) acts as fallback on API failure.

This is the key architectural decision. The LLM sees 50 companies in one call — not 457 sequential calls. Cost per query: ~$0.002 (query expansion) + ~$0.004 (rerank) = ~$0.006 total. Two API calls regardless of dataset size.

Before this I have tried with re-ranking with a cross-encoder ms-marco-MiniLM-L-6-v2 on top 20 → final 10. The cross-encoder (ms-marco-MiniLM-L-6-v2) was taking the top 20 candidates from Stage 3 (structured scoring), scores each (query, company_text) pair, and re-sorts them. The final output was sorted by ce_score, not the structured score — so the cross-encoder has the final say on ordering. You can see both stage3_score and ce_score in the returned result dicts.

---

## 3.2 Tradeoffs

### What was optimised for

**Accuracy over latency.** At 457 companies and 12 queries, query latency of 2–4 seconds is acceptable. The LLM reranker was chosen over a trained cross-encoder (ms-marco-MiniLM) because the ms-marco model was trained on web passage retrieval — it measures text similarity, not company qualification. It has no concept of "pharma ≠ SaaS" or "cosmetics brand ≠ packaging supplier." The LLM understands industry semantics and business model distinctions that a retrieval model cannot.
I honestly opted for having more acccurancy than having more latency. Because I want the best comapnies to be selected, not to get random results as it happened with the cross-encoder. And hnestly, the LLM is not so slow, it does not add up much time(just some seconds) but the benefits are immense.

**Two LLM calls per query (not zero, not 457).** Query expansion and reranking each require one API call. This is a deliberate tradeoff: expanding a cached query costs nothing on repeated runs, and the reranker call is batched so cost is independent of dataset size.

**Benefit of the doubt on missing data.** The dataset has significant missing fields (employee count, revenue, year founded). Hard-filtering on missing data would eliminate many real companies. The choice to treat missing = pass means some irrelevant companies survive the filter, but the LLM reranker catches them.
I consider it is the best approach in order to get the wanted results. Eliminate what seems to not fit the criteria, let the LLM fill it with relevant companies.

**Hard filter aggressiveness calibrated for 457 records.** On a 500-company dataset, stacking three hard filters (geography + NAICS + revenue) frequently eliminates all candidates. The conditional NAICS filter and tiered fallback exist specifically because of this. At 100K+ companies these constraints would be less aggressive, not more.

### What was sacrificed

**Determinism.** LLM scores vary slightly across runs. A company that scores 2 might score 1 in a different run, changing its rank relative to another 2. For borderline cases this is real inconsistency. This is because LLM are impredictable and it can get varied results. This can be handled by reducing the temperature, as low temperatues mean more results that are fixed, deterministic.

**Explainability of the reranker.** The cross-encoder produced a numeric score with a clear (if imperfect) interpretation. The LLM score of 0–3 is more semantically meaningful but less auditable — you cannot inspect the model's internal reasoning for a given score without requesting chain-of-thought output.

**Inference speed.** The cross-encoder ran locally in ~100ms. The LLM reranker adds ~1-2 seconds of network latency per query.

---

## 3.3 Error Analysis

### Query: "Find logistics companies in Germany" / "Logistic companies in Romania"

**Germany:** The dataset contains no German logistics companies. The tiered fallback drops geography and finds 1 company globally with logistics NAICS codes (CFR Romania, a rail infrastructure company). The LLM correctly scores it 0 — it is a railway operator, not a freight logistics company. Result: 1 result, borderline relevant.

**Root cause:** Pure data gap. The pipeline handles it correctly — the tiered fallback and honest LLM scoring are better than returning Romanian warehousing companies as "German logistics."

**Romania:** Bunge Romania (oilseed commodity trader) scores 3 from the LLM, ranking above Brasov Industrial Portfolio (actual warehousing company). Bunge's description mentions "wholesale distribution of agricultural commodities" — the LLM interpreted distribution operations as logistics. This is a genuine false positive at rank 1.

**Root cause:** The LLM is interpreting "distribution" in Bunge's description too broadly. A stricter qualification prompt, or a second-pass binary qualification step ("is this primarily a logistics company, yes/no"), would catch this.

### Query: "Manufacturing companies in the DACH region founded before 2000"

All 10 results are Swiss pharmaceutical companies, scored 0 by the LLM. The results are correct in their scores — the LLM agrees none of them match "manufacturing" in the general industrial sense. But they appear because:

1. NAICS "32" (Manufacturing) includes pharmaceutical manufacturing (NAICS 3254). Swiss pharma companies correctly pass the NAICS soft score.
2. The dataset has almost no non-pharma DACH manufacturers. After geography + year filter, the candidate pool is dominated by Swiss pharma.

**Root cause:** Combined data gap (dataset biased toward pharma/software/HR) and NAICS taxonomy ambiguity (pharma is manufacturing by classification). The query expansion correctly described "automotive, aerospace, machinery" manufacturing, but no such companies existed to retrieve.

### Query: "B2B SaaS companies with annual revenue over $10M" (before LLM reranker)

With only the cross-encoder, results 2–10 were pharmaceutical companies: Acino, OM Pharma, CordenPharma, Ferring. All had revenue > $10M (passing the hard filter) and dense text descriptions that scored high on cosine similarity. The cross-encoder, trained on web passage retrieval, could not distinguish "has revenue over $10M" from "is a SaaS company."

**After LLM reranker:** Pharma companies score 0. Personio, NOVRH, quarksUp, Lucca appear at ranks 1–4. This is the clearest demonstration of why the LLM reranker was chosen over the cross-encoder.

### Query: "E-commerce companies using Shopify or similar platforms"

**Structural impossibility.** No company in the dataset has a tech-stack field. No company description mentions Shopify, WooCommerce, or any e-commerce platform. The LLM scores Harbor Freight, Home Depot, and Walmart as 3 — correctly identifying them as companies with e-commerce operations, but incorrectly classifying them as "Shopify-type companies." They are large physical retailers with e-commerce channels, not DTC brands built on Shopify.

**Root cause:** Data limitation. This query cannot be answered from the available fields. The honest response would be: "No companies in the dataset match this specific technology criterion."


---

## 3.4 Scaling to 100,000 Companies

The pipeline has four components with different scaling characteristics.

### FAISS — scales well

`IndexFlatIP` does exact inner-product search. At 100K companies:
- Index size: ~100K × 384 floats × 4 bytes ≈ 150MB — fits in RAM
- Search time: ~5ms (FAISS is highly optimised for this)
- No changes needed until ~10M companies, at which point approximate search (`IndexIVFFlat`, `IndexHNSW`) trades 0.5% recall for 10–100× speed

### BM25 — needs replacement

`rank_bm25` is a pure-Python in-memory implementation. At 100K documents it becomes slow (~500ms) and the index won't fit comfortably in RAM.

**Replacement:** Elasticsearch or OpenSearch with BM25 as native scoring. Queries in <10ms, horizontally scalable, supports filtered BM25 (geography/NAICS pre-filter before scoring).

### Hard Filter — needs indexing

The current implementation scans all records linearly (`O(n)`). At 100K companies this is 100× slower.

**Replacement:** A relational index (PostgreSQL or DuckDB) with columns for `country_code`, `employee_count`, `revenue`, `is_public`, `year_founded`, `naics_code`. The hard filter becomes a SQL WHERE clause executing in milliseconds regardless of dataset size.

### LLM Reranker — scales perfectly

The LLM always sees the top-50 candidates, not the full corpus. Cost and latency are constant regardless of whether the corpus is 500 or 100 million companies. This is the core architectural advantage of the pipeline.

### Query Expansion — scales perfectly

One LLM call per unique query. Cached indefinitely. At any scale: one call per new query.

### Summary

| Component | Current | At 100K |
|---|---|---|
| Hard filter | Linear scan O(n), ~5ms | SQL index, ~1ms |
| BM25 | In-memory, ~50ms | Elasticsearch, ~10ms |
| FAISS | IndexFlatIP, ~1ms | IndexIVFFlat, ~2ms |
| Query expansion | 1 LLM call, cached | unchanged |
| LLM reranker | 1 LLM call, top-50 | unchanged |

---

## 3.5 Failure Modes

### Confident scores on a bad candidate pool

The LLM scores companies relative to the query, not relative to an absolute "good match" standard. When all 50 candidates are genuinely wrong — because no relevant company exists in the dataset — the model scores them all 0 or 1, but the top-ranked 0s still appear in the output as if they were results.

**Example:** "Find logistics companies in Germany" returns CFR Romania (rail infrastructure) at rank 1 with score 0. The score correctly signals a bad match, but the company still surfaces.

**Detection in production:** Monitor the distribution of LLM scores per query. A query where all top-10 results score ≤ 1 should trigger a "no confident matches found" response rather than showing low-quality results. An alert threshold of `max_score < 2` for all top-10 results would catch this pattern.

### Query expansion hallucinating the wrong industry

The expansion LLM occasionally misidentifies the target industry. For "Manufacturing companies in the DACH region founded before 2000," the expansion correctly described automotive/aerospace manufacturing — but if the query were more ambiguous, the LLM might expand toward the wrong sector (e.g. expanding "energy companies in Norway" toward oil rather than renewables). The FAISS search then retrieves candidates from the wrong industry, and the reranker has a bad pool to work with.

**Detection in production:** Log every expansion alongside query + top-10 results. Human review of expansions that produce all-0 reranker scores would catch systematic misdirections.

### NAICS taxonomy traps

NAICS "32" (Manufacturing) includes pharmaceutical manufacturing (3254), chemical manufacturing (325), and plastics (326). A query for "manufacturing companies" that should return industrial manufacturers will score pharma companies as perfect NAICS matches. The LLM reranker largely corrects this, but it requires the LLM to understand industrial vs pharmaceutical manufacturing context from the query — which it usually does, but not always for ambiguous queries.

**Detection in production:** Spot-check queries with `industry_keywords = ["manufacturing"]` where top results are pharma companies. This specific pattern is a known failure mode.

### Missing data exploited by adversarial records

The "missing = pass" rule on hard filters is correct for real-world sparse data but exploitable if the dataset contains low-quality records with no structured fields. A company with no country code, no NAICS code, and no employee count passes every hard filter. It then competes purely on semantic similarity and LLM scoring.

**Detection in production:** Track `has_employees`, `has_revenue`, `has_naics` completeness rates per query result set. A result set dominated by data-sparse companies signals either a data quality issue or a query that nothing in the corpus genuinely matches.

### LLM inconsistency on borderline cases

A company that is a genuine 1.5 on the 0–3 scale will receive either 1 or 2 depending on the run, changing its rank relative to other borderline companies. For queries with many borderline candidates (e.g. the fintech query, where forex companies are debatable matches for "competing with traditional banks"), rank positions 5–10 are effectively random within the tied-score band.

**Mitigation:** Use temperature=0 for the reranker call (deterministic greedy decoding). The current implementation does not set temperature explicitly — adding `temperature=0` to the API call would eliminate this source of variance.

---


---

### Phase 2 — LightGBM LambdaMART (explored, as a next phase to use : this is more of a scaling consideration, you can not use it with such a small dataset but it might be really useful in the future)

Three scripts (`label_generation_phase2.py`, `training_phase2.py`, `rank_phase2.py`) implement a full learning-to-rank pipeline using LightGBM LambdaMART as the final reranker instead of the LLM.

The approach:
1. `label_generation_phase2.py` — runs 65 diverse queries, sends top-200 candidates per query to an LLM, gets 0–3 relevance labels → writes `labels.jsonl`
2. `training_phase2.py` — engineers 11 features per (query, company) pair and trains a LambdaMART model optimised on NDCG@10
3. `rank_phase2.py` — uses the trained model at inference time instead of the cross-encoder

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

Model: LightGBM LambdaMART with label_gain=[0,1,3,7] (exponential gain matching 0–3 relevance scale), 80/20 query-wise train/val split, early stopping on NDCG@10. Outputs ltr_model.txt + feature_importance.json.

 3000 labels written. Label distribution:
  - 0 (not relevant): 2168 (72%)
  - 1 (slightly relevant): 423 (14%)
  - 2 (moderately relevant): 171 (6%)
  - 3 (highly relevant): 238 (8%)

 Phase 2 LTR Model — Observations

  Training results:
  - Train NDCG@10: 0.865, Val NDCG@10: 0.527
  - Early stopping at iteration 3 (out of 50+) — the model is heavily overfitting on very few data points
  - structured_score dominates feature importance (99.1 gain) — the LTR model is essentially learning to trust the
  existing pipeline score, not adding much new signal

  Feature importance:
  - structured_score: 99.1 — overwhelmingly dominant
  - bm25_norm, keyword_overlap, cosine_sim: ~42-50 — moderate
  - industry_score, has_revenue: 0.0 — completely unused

   Root causes of problems:
  1. Label imbalance — 72% of labels are 0, only 8% are 3. The model learns mostly "what's not relevant"
  2. Too few training examples — only 3000 pairs across 20 unique queries; LambdaMART needs more diversity
  3. Dataset sparsity — very few German logistics companies in the corpus, so geo filtering fails entirely

**Why it was not used in the final pipeline:**

The fundamental problem is circular: the training labels are generated by an LLM, and the model learns to predict those labels. At inference time you still need the LLM to score new queries — unless you retrain. You end up with a slower, less flexible system that approximates what the LLM already does directly.

This approach makes sense at scale (100K+ companies, latency-critical, LLM inference too expensive) — the LightGBM model runs in microseconds locally with no API cost. At 457 companies and 12 queries, the LLM reranker is strictly better. The scripts are kept here as a reference for how the system would evolve under production constraints.

The dataset is too small now to use this model but when I designed the arhitecture, I considered it a good option for later use.
---

## Evaluation (`evaluate.py`)

### Metrics computed

| Category | Metrics |
|---|---|
| Retrieval quality | NDCG@5, NDCG@10, MAP@10, MRR, P@1/3/5/10 |
| Binary (score ≥ 2) | precision, ROC-AUC |
| Strict (score = 3) | P@1\_strict, P@3\_strict |
| Per-signal | mean judge score |

### Label types

- **graded (0–3)** — raw LLM judge scores, used for NDCG
- **binary (≥ 2)** — relevant / not-relevant, used for MAP / P@K / ROC-AUC
- **strict (= 3)** — highly relevant only, used for P@1\_strict / P@3\_strict

### LLM judge

Uses OpenAi mini -4 independently from the pipeline's own reranker (OpenAI `gpt-4o-mini`), so the evaluation is not self-referential. The judge sees the same company summaries but scores them without knowledge of how the pipeline ranked them. Since we have no grounding true, at least we have something that has the mind of a human that can properly review.

### Three ablations per query

1. **llm** — full pipeline with LLM reranker (gpt-4o-mini, batched)
2. **ce** — cross-encoder only (ms-marco-MiniLM, no LLM call)
3. **stage3** — structured 6-signal score only, no neural reranker

This directly answers the question of how much each reranking stage contributes to result quality — and whether the LLM reranker justifies its API cost over the free local cross-encoder.

### Efficiency & overhead

Per-stage wall time is measured for every query:

| Stage | What it measures |
|---|---|
| `expand` | Query expansion LLM call (cached on repeat) |
| `filter` | Hard filter linear scan |
| `cand` | FAISS + BM25 + RRF fusion |
| `struct` | Structured 6-signal scoring |
| `ce` | Cross-encoder inference (local, 20 pairs) |
| `llm` | LLM reranker API call (1 batched call) |

Token counts and estimated USD cost are tracked per query for both the pipeline reranker and the evaluation judge.

### Complexity report

Static characteristics logged at evaluation time: number of pipeline stages, models loaded, FAISS index type and size, scoring weight breakdown, corpus size, and candidate funnel (50 → 20 → 10).

### Usage

```bash
python evaluate.py                     # all 12 queries
python evaluate.py --queries 3         # quick smoke test
python evaluate.py --out results.json  # also write full JSON output
```
After running for the first time evaluation.py I have received the following score:

   Aggregate Metrics (with real gpt-4o judge)

  ┌─────────┬──────────────┬───────────────┬─────────────┐
  │ Metric  │ LLM reranker │ Cross-encoder │ Stage3 only │
  ├─────────┼──────────────┼───────────────┼─────────────┤
  │ NDCG@5  │ 0.662        │ 0.539         │ 0.626       │
  ├─────────┼──────────────┼───────────────┼─────────────┤
  │ NDCG@10 │ 0.738        │ 0.680         │ 0.728       │
  ├─────────┼──────────────┼───────────────┼─────────────┤
  │ MAP@10  │ 0.709        │ 0.571         │ 0.682       │
  ├─────────┼──────────────┼───────────────┼─────────────┤
  │ MRR     │ 0.760        │ 0.658         │ 0.679       │
  ├─────────┼──────────────┼───────────────┼─────────────┤
  │ P@1     │ 0.750        │ 0.583         │ 0.583       │
  ├─────────┼──────────────┼───────────────┼─────────────┤
  │ P@3     │ 0.694        │ 0.417         │ 0.667       │
  ├─────────┼──────────────┼───────────────┼─────────────┤
  │ ROC-AUC │ 0.834        │ 0.648         │ 0.574       │
  └─────────┴──────────────┴───────────────┴─────────────┘

The reality is that this cost is less than 1 euro but it improves my pipeline so much.
Here are the main findings:
  ---
  Key Observations

  1. LLM reranker is now clearly the winner — unlike before (when Stage3 was "perfect" due to the self-referential bug),
   with a real independent judge the LLM reranker leads on every metric, especially ROC-AUC (0.834 vs 0.574 for Stage3).

  2. Stage3 alone is still a strong baseline — it beats the cross-encoder on almost every metric, meaning the
  handcrafted signals are very effective for this dataset.

  3. Cross-encoder is the weakest — consistently the lowest scores. It's a general-purpose MS-MARCO model not tuned for
  company search, which limits its value here.

  4. Two queries score 0.000 — data coverage issue:
  - "Find logistics companies in Germany" → only 1 candidate, judged score=0 (it's not actually a logistics company)
  - "Manufacturing companies in DACH before 2000" → 22 candidates but all scored 0 by gpt-4o — the pipeline is returning
   companies that don't match the manufacturing criteria strictly

  5. "Logistic companies in Romania" — very poor (MRR=0.125). The hard filter fallback returned global results instead
  of Romania-specific ones, and the semantic search isn't finding relevant matches.

  6. Best performing queries — Pharma in Switzerland, Packaging suppliers, B2B SaaS HR: all 3 ablations score perfectly
  or near-perfectly. These are queries where industry semantics are clear and the dataset has good coverage.

  7. Cost: judge (gpt-4o) costs ~10x more than the pipeline reranker (gpt-4o-mini) — $0.04 vs $0.004 for 12 queries.
  Fine for evaluation but confirms gpt-4o-mini is the right choice for the pipeline itself.

  I consider a ROC-AUC above 8 is a good score for the model. It shows that the model is performing well and can be included in production. This evaluation does not address the task itself, but better how the model is behaving and if it respects what it was coded to respect. This shows us again that the LLM reranker was a better approach than the cross-encoder, as passing some companies that are in top to the LLM adds clarity to the solution and can give us better results at really low costs.


TESTING TO SEE IF THE REQUIREMENT IS RESPECTED:

In solution.py as mentioned earlier I have included the following queries.

 5 "Original" queries (self-generated for testing):
  1. Find logistics companies in Germany
  2. Public software companies with more than 1,000 employees
  3. B2B SaaS companies with annual revenue over $10M
  4. Fast-growing fintech companies competing with traditional banks in Europe
  5. Manufacturing companies in the DACH region founded before 2000

  11 "Actual evaluation queries" (labeled as such in the code):
  6. Logistic companies in Romania
  7. Food and beverage manufacturers in France
  8. Companies that could supply packaging materials for a direct-to-consumer cosmetics brand
  9. Construction companies in the United States with revenue over $50 million
  10. Pharmaceutical companies in Switzerland
  11. B2B SaaS companies providing HR solutions in Europe
  12. Clean energy startups founded after 2018 with fewer than 200 employees
  13. Fast-growing fintech companies competing with traditional banks in Europe ← duplicate of #4
  14. E-commerce companies using Shopify or similar platforms
  15. Renewable energy equipment manufacturers in Scandinavia
  16. Companies that manufacture or supply critical components for electric vehicle battery production

I will send below a query to query analysis:

  Query-by-Query Analysis

  Q1: "Find logistics companies in Germany" — FAIL

  - Problem: Zero German logistics companies in dataset. Pipeline falls back to global industry matches and returns CFR
  (Romanian railway). Score is very negative (-9.4).
  - Root cause: Pure data gap — no German logistics companies exist in the 457-company dataset.

  ---
  Q2: "Public software companies with more than 1,000 employees" — EXCELLENT

  - CGI, TCS, ASGN, Concentrix, EPAM — all correct, all public, all large.
  - Top 10 is nearly perfect. Minor noise: Concentrix is more of a BPO than software, but defensible.

  ---
  Q3: "B2B SaaS companies with annual revenue over $10M" — POOR

  - NOVRH (#1) is correct. But positions 2–10 are Swiss pharma companies.
  - Root cause: Dataset has almost no SaaS companies with revenue data. The cross-encoder is promoting pharma because
  they do have revenue > $10M, even though they're not SaaS.
  - Fix candidate: Stricter industry NAICS gate before cross-encoder.

  ---
  Q4: "Fast-growing fintech companies competing with traditional banks in Europe" — MEDIOCRE

  - European Pay (#1, +0.16) is correct. Phin (#2) is borderline. But SkyHarvest (#3) is an agri-finance company, not
  fintech.
  - CordenPharma (#9) is clearly wrong — pharma leak into fintech results.
  - The spaCy fix works (no banking companies), but the dataset has very few actual fintechs.

  ---
  Q5: "Manufacturing companies in the DACH region founded before 2000" — FAIL

  - All results are Swiss pharma, no DE/AT manufacturers.
  - Root cause: Dataset is heavily pharma/software biased; no German/Austrian manufacturers present.

  ---
  Q6: "Logistic companies in Romania" — GOOD

  - Brasov Industrial Portfolio (#1, warehousing/storage) and STILL (#2, material handling) are reasonable.
  - CBRE (#3, real estate) and METRO (#4, grocery) are questionable but Romanian.
  - Rompetrol (#8) and Unilever (#9) are clearly wrong — no logistics relation.

  ---
  Q7: "Food and beverage manufacturers in France" — EXCELLENT

  - Top 10 is entirely French food/beverage companies. French wineries, soft drink manufacturers.
  - Clean, relevant, high-confidence scores (+5 to +8).

  ---
  Q8: "Companies that could supply packaging materials for a direct-to-consumer cosmetics brand" — EXCELLENT

  - Top 10 is packaging companies — plastics bottles, glass containers, cosmetic packaging. Highly relevant.
  - Good international coverage (VN, AU, US, KR, CN). Scores solid (+3.4 to +5.7).

  ---
  Q9: "Construction companies in the United States with revenue over $50 million" — VERY GOOD

  - Focus Civil, Tutor Perini, DPR, Kiewit, Quanta, Skanska — all US construction with large revenue.
  - Colaska (#6, $38B revenue) looks like a data quality issue (likely rev in thousands).
  - Otherwise strong results.

  ---
  Q10: "Pharmaceutical companies in Switzerland" — EXCELLENT

  - Sumitomo Pharma, Acino, Idorsia, TRB Chemedica, Helsinn, CARBOGEN — all Swiss pharma.
  - Perfect precision, high scores (+6 to +8.7). Best-performing query.

  ---
  Q11: "B2B SaaS companies providing HR solutions in Europe" — GOOD

  - Paychex, Sincron HR, Foederis, NOVRH, Bizneo HR — all HR software in Europe.
  - Pipeline triggered [NOTE] (no exact industry+region match) but still returned relevant results.
  - Jooma (#4, 4 employees, $0.3M rev) is marginal but technically correct.

  ---
  Q12: "Clean energy startups founded after 2018 with fewer than 200 employees" — MEDIOCRE

  - MantaWind and Verta are good. But You Shan Technology (#3, Chinese chemicals) and Norhybrid Renewables (#5, $193M
  rev) don't fit "startup <200 employees."
  - The year_founded and employee_count filters don't appear to be cutting hard enough here.

  ---
  Q13 (duplicate of Q4): "Fast-growing fintech companies..." — same results

  ---
  Q14: "E-commerce companies using Shopify or similar platforms" — POOR

  - Harbor Freight, Home Depot, Walmart — these are traditional retail giants, not Shopify e-commerce brands.
  - Root cause: The dataset has no native e-commerce/DTC brands. Pipeline falls back to "retail" companies. The
  "Shopify" specificity can't be matched.

  ---
  Q15: "Renewable energy equipment manufacturers in Scandinavia" — VERY GOOD

  - BayWa r.e. Scandinavia (#1), Ecopwr, Home Energy, Alotta, Moventas (wind gearboxes), WindSpider — all relevant.
  - Varmalämmitys (#2, Finnish heating equipment) is borderline but defensible.

  ---
  Q16: "Companies that manufacture or supply critical components for electric vehicle battery production" — EXCELLENT

  - QuantumScape, Nascent Materials, Greenbatt, Altmin, Mitra Chem, Sepion — all battery/EV component makers.
  - Strong semantic reasoning: the query is complex inference and the pipeline handles it well.

  ---
  Summary

  ┌────────┬──────────────────────────────┬───────┐
  │   #    │            Query             │ Grade │
  ├────────┼──────────────────────────────┼───────┤
  │ Q2     │ Public software >1k emp      │ A     │
  ├────────┼──────────────────────────────┼───────┤
  │ Q7     │ Food & bev in France         │ A     │
  ├────────┼──────────────────────────────┼───────┤
  │ Q8     │ Packaging for cosmetics      │ A     │
  ├────────┼──────────────────────────────┼───────┤
  │ Q10    │ Pharma in Switzerland        │ A     │
  ├────────┼──────────────────────────────┼───────┤
  │ Q16    │ EV battery components        │ A     │
  ├────────┼──────────────────────────────┼───────┤
  │ Q9     │ US construction >$50M rev    │ B+    │
  ├────────┼──────────────────────────────┼───────┤
  │ Q15    │ Renewable energy Scandinavia │ B+    │
  ├────────┼──────────────────────────────┼───────┤
  │ Q6     │ Logistics Romania            │ B     │
  ├────────┼──────────────────────────────┼───────┤
  │ Q11    │ HR SaaS Europe               │ B     │
  ├────────┼──────────────────────────────┼───────┤
  │ Q4/Q13 │ Fintech Europe               │ C     │
  ├────────┼──────────────────────────────┼───────┤
  │ Q12    │ Clean energy startups        │ C     │
  ├────────┼──────────────────────────────┼───────┤
  │ Q3     │ B2B SaaS >$10M               │ D     │
  ├────────┼──────────────────────────────┼───────┤
  │ Q14    │ E-commerce Shopify           │ D     │
  ├────────┼──────────────────────────────┼───────┤
  │ Q1     │ Logistics Germany            │ F     │
  ├────────┼──────────────────────────────┼───────┤
  │ Q5     │ DACH manufacturing <2000     │ F     │
  └────────┴──────────────────────────────┴───────┘

  The two F grades and the D grades are all data gaps — the dataset simply doesn't contain those companies. The pipeline logic is sound; the dataset is the limiting factor. The notable fixable issue is Q3 (pharma leaking into SaaS results) which a stronger NAICS gate could address.


## Cross-Validation Module — cross_validation.py
The pipeline's LLM reranker and cross-encoder are probabilistic — they can surface plausible-sounding but structurally wrong results. This validator adds a deterministic safety net that catches obvious mismatches cheaply, before they reach the user.

  A post-retrieval consistency checker that runs after the pipeline returns results. For every (query, company) pair, it
   validates the company's full attribute profile against:
  1. Its own taxonomy label (internal consistency)
  2. The query's intent (query plausibility)

  No LLM call needed — fully deterministic and rule-based.

  ---
  Check 1 — Internal Consistency

  ▎ Does this company's own data make sense together?

  ┌───────────────────────────────────────────────────────────────────┬─────────────────────────────────────────────┐
  │                         Example violation                         │                What's wrong                 │
  ├───────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────┤
  │ NAICS=5112 (Software) + business_model=["Manufacturing"] only     │ Software companies don't manufacture        │
  ├───────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────┤
  │ NAICS=484 (Logistics) + zero logistics words in description       │ Label contradicts the text                  │
  ├───────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────┤
  │ NAICS=31xx (Manufacturing) + business_model=["Retail", "Service   │ Manufacturing NAICS but no manufacturing    │
  │ Provider"] only                                                   │ business model                              │
  └───────────────────────────────────────────────────────────────────┴─────────────────────────────────────────────┘

  These checks are query-independent — they catch bad data regardless of what was searched.

  ---
  Check 2 — Query Plausibility

  ▎ Is this company a reasonable answer to this specific query?

  ┌────────────────────────────────────┬─────────────────────────────────────────┬──────────────────────────────────┐
  │            Query intent            │                 Company                 │             Verdict              │
  ├────────────────────────────────────┼─────────────────────────────────────────┼──────────────────────────────────┤
  │ "manufacturing companies"          │ NAICS=325412 Pharma                     │ ✓ allowed (exception)            │
  ├────────────────────────────────────┼─────────────────────────────────────────┼──────────────────────────────────┤
  │ "SaaS companies"                   │ NAICS=333414 Heating equipment          │ ✗ flagged                        │
  ├────────────────────────────────────┼─────────────────────────────────────────┼──────────────────────────────────┤
  │ "packaging suppliers for cosmetics │ Company's offerings mention "cosmetics  │ ✗ flagged — it's a buyer, not a  │
  │  brand"                            │ products"                               │ supplier                         │
  ├────────────────────────────────────┼─────────────────────────────────────────┼──────────────────────────────────┤
  │ "fintech companies"                │ NAICS=522320 Financial services         │ ✓ passes                         │
  └────────────────────────────────────┴─────────────────────────────────────────┴──────────────────────────────────┘

  ---
  Scoring

  Each company gets a confidence score from 0.0 to 1.0:

  confidence = 1.0
             - 0.15 × (number of internal consistency flags)
             - 0.35 × (number of query plausibility flags)

  A company is marked is_plausible = False if:
  - confidence < 0.5, or
  - any query plausibility flag was raised (strict — wrong industry = demote)

  I used it more to evaluate and as a demonstration but I think it can further be used in one of these scenarios, with option 2 feeling th emost natural.
 
  Option 1 — Post-retrieval filter (in the pipeline)
  We can plug it into solution.py after the LLM reranker, before returning results. Any company with is_plausible=False gets demoted to the bottom or removed entirely.
  # In RankingEngine.rank()
  results = self._rerank_llm(q, top_stage3)
  results = [r for r in results if validate(query, r).is_plausible]
  Use case: production — stops bad results from ever reaching the user.

  ---
  Option 2 — Evaluation signal (in evaluate.py)
  We can add the cross-validator as an additional metric column in the eval report. For each query, report what % of the top-10
   results passed the consistency check — per ablation.

  Metric                LLM reranker  Cross-encoder  Stage3 only
  NDCG@10                     0.738          0.680        0.728
  Cross-val pass rate         0.80           0.60         0.70   ← new

  This gives us a cheap proxy for precision that doesn't require LLM judge calls — you can run it on every query for free.