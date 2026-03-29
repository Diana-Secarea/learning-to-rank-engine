"""
solution.py — Hybrid Retrieval + Structured Scoring + Cross-Encoder Re-ranking

Pipeline:
  1. Intent Parsing   — extract structured signals from the natural language query
  2. Hard Filter      — boolean elimination (missing data = benefit of doubt = pass)
  3. Candidate Gen    — FAISS (semantic) + BM25 (keyword) fused with RRF → top 50
  4. Structured Score — weighted 0-1 signals → top 20
  5. Cross-Encoder    — ms-marco-MiniLM re-ranks top 20 → final top 10
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import re
import sys
import textwrap
from dataclasses import dataclass, field, replace

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from text_to_embed import company_to_text

# ── Constants ──────────────────────────────────────────────────────────────────

EMBED_MODEL  = "BAAI/bge-small-en-v1.5"
CROSS_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL    = "gpt-4o-mini"   # fast + cheap for batch qualification
DATA_PATH    = pathlib.Path("companies_clean.jsonl")
CACHE_DIR    = pathlib.Path(".cache")

CANDIDATE_K  = 50   # FAISS + BM25 → RRF candidates
RERANK_TOP   = 20   # fed to LLM / cross-encoder reranker
FINAL_TOP    = 10   # returned to caller

YEAR_MIN, YEAR_MAX = 1900, 2024

# Scoring weights (must sum to 1.0)
W_COSINE   = 0.40
W_BM25     = 0.20
W_INDUSTRY = 0.15
W_LOCATION = 0.10
W_SIZE     = 0.10
W_RECENCY  = 0.05

# ── Geography ──────────────────────────────────────────────────────────────────

COUNTRY_MAP: dict[str, str] = {
    "germany": "de", "german": "de",
    "france": "fr", "french": "fr",
    "united states": "us", "usa": "us", "america": "us",
    "united kingdom": "gb", "britain": "gb", "uk": "gb", "england": "gb",
    "romania": "ro", "netherlands": "nl", "dutch": "nl", "holland": "nl",
    "sweden": "se", "norway": "no", "denmark": "dk", "finland": "fi",
    "switzerland": "ch", "austria": "at", "belgium": "be",
    "spain": "es", "spanish": "es", "italy": "it", "italian": "it",
    "poland": "pl", "czech": "cz", "czechia": "cz", "hungary": "hu",
    "portugal": "pt", "greece": "gr", "ireland": "ie",
    "canada": "ca", "australia": "au", "india": "in",
    "china": "cn", "japan": "jp", "brazil": "br", "mexico": "mx",
    "singapore": "sg", "israel": "il", "south korea": "kr", "korea": "kr",
    "south africa": "za", "uae": "ae", "turkey": "tr",
    "ukraine": "ua", "russia": "ru",
}

REGION_MAP: dict[str, set[str]] = {
    "europe": {
        "de","fr","gb","it","es","nl","pl","ro","se","no","dk","fi",
        "ch","at","be","pt","gr","ie","hu","cz","sk","si","hr","bg",
        "rs","ua","tr","is","lt","lv","ee","lu","mt","cy",
    },
    "dach":    {"de","at","ch"},
    "nordics": {"se","no","dk","fi","is"},
    "benelux": {"be","nl","lu"},
    "north america": {"us","ca","mx"},
    "latin america": {"br","mx","ar","cl","co","pe","ve","ec","uy","py","bo"},
    "latam":   {"br","mx","ar","cl","co","pe","ve","ec","uy","py","bo"},
    "asia":    {"cn","jp","in","sg","kr","tw","hk","th","id","my","ph","vn"},
    "apac":    {"cn","jp","in","sg","kr","tw","hk","th","id","my","ph","vn","au","nz"},
    "middle east": {"il","ae","sa","qa","kw","bh","om","jo","lb","eg"},
    "africa":  {"za","ng","ke","gh","et","tz","eg","ma","tn"},
    "oceania": {"au","nz"},
    "emea":    {
        "de","fr","gb","it","es","nl","pl","ro","se","no","dk","fi",
        "ch","at","be","pt","gr","ie","hu","cz","za","ng","ke","ae","sa","il","eg","ma",
    },
}

# ── Industry / NAICS ───────────────────────────────────────────────────────────

# keyword → list of NAICS prefixes (2 or more digits)
NAICS_MAP: dict[str, list[str]] = {
    "logistics":        ["484","488"],
    "trucking":         ["484"],
    "freight":          ["484","4841","4842"],
    "shipping":         ["483","484"],
    "transportation":   ["484","485","487"],
    "warehousing":      ["493"],
    "supply chain":     ["484","493"],
    "software":         ["5112","511210"],
    "saas":             ["5112","511210"],
    "tech":             ["51","5112","518"],
    "technology":       ["51","5112"],
    "fintech":          ["522","5221","5222","5223"],
    "finance":          ["52"],
    "banking":          ["5221","52211"],
    "financial":        ["52"],
    "insurance":        ["524"],
    "real estate":      ["531"],
    "manufacturing":    ["31","32","33"],
    "retail":           ["44","45"],
    "ecommerce":        ["454","45411"],
    "e-commerce":       ["454","45411"],
    "healthcare":       ["62"],
    "pharma":           ["3254"],
    "pharmaceutical":   ["3254"],
    "biotech":          ["5417"],
    "energy":           ["211","324","325","221"],
    "oil":              ["211"],
    "gas":              ["211","213"],
    "agriculture":      ["11"],
    "farming":          ["111","112"],
    "construction":     ["23"],
    "consulting":       ["5416"],
    "marketing":        ["5418"],
    "advertising":      ["5418"],
    "media":            ["515","516","519"],
    "publishing":       ["511"],
    "telecom":          ["517"],
    "telecommunications":["517"],
    "hospitality":      ["721","722"],
    "travel":           ["561510","56151"],
    "aerospace":        ["3364"],
    "automotive":       ["3361","3362","3363"],
    "food":             ["311","722"],
    "cybersecurity":    ["5112","5415"],
    "cloud":            ["518","5112"],
    "data":             ["518","519"],
    "ai":               ["5112","5415"],
    "education":        ["61"],
    "edtech":           ["61","5112"],
    "hr":               ["5613"],
    "recruiting":       ["5613"],
    "legal":            ["5411"],
    "accounting":       ["5412"],
    # Renewable / clean energy — specific codes to avoid matching oil (211/324)
    "clean energy":     ["221114","221115","221116","221117","221118","333611","334413"],
    "renewable energy": ["221114","221115","221116","221117","221118"],
    "wind energy":      ["221115","333611"],
    "solar energy":     ["221114","334413"],
    "wind":             ["221115","333611"],
    "solar":            ["221114","334413"],
}

# Inverse: naics_prefix → list of keywords (for soft label matching)
_NAICS_INVERSE: dict[str, list[str]] = {}
for _kw, _prefixes in NAICS_MAP.items():
    for _p in _prefixes:
        _NAICS_INVERSE.setdefault(_p, []).append(_kw)

# ── Business model vocabulary (from data) ─────────────────────────────────────

BM_QUERY_TO_CANONICAL: dict[str, str] = {
    "b2b":      "Business-to-Business",
    "b2c":      "Business-to-Consumer",
    "b2g":      "Government/Public Sector",
    "saas":     "Software-as-a-Service",
    "wholesale": "Wholesale",
    "retail":   "Retail",
    "ecommerce": "E-commerce",
    "e-commerce": "E-commerce",
    "subscription": "Subscription-Based",
    "nonprofit": "Non-Profit/Non-Governmental Organization",
    "non-profit": "Non-Profit/Non-Governmental Organization",
    "manufacturing": "Manufacturing",
    "logistics": "Logistics/Transportation",
}

# ── Query dataclass ────────────────────────────────────────────────────────────

@dataclass
class Query:
    raw:              str
    country_codes:    set[str]        = field(default_factory=set)   # ISO-2 lowercase
    region_countries: set[str]        = field(default_factory=set)   # expanded from region names
    employee_min:     int   | None    = None
    employee_max:     int   | None    = None
    revenue_min:      float | None    = None
    revenue_max:      float | None    = None
    is_public:        bool  | None    = None
    business_models:  set[str]        = field(default_factory=set)   # canonical strings
    naics_prefixes:   list[str]       = field(default_factory=list)
    year_founded_min: int   | None    = None
    year_founded_max: int   | None    = None
    industry_keywords: list[str]      = field(default_factory=list)  # for soft scoring
    expanded:         str   | None    = None                         # LLM-rewritten query for FAISS

# ── spaCy target-token extraction ─────────────────────────────────────────────

_nlp = None

def _load_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    return _nlp

# Verbs that introduce a competitor/context clause, not the target company
CONTEXT_VERBS = {
    "compete", "competing", "disrupt", "disrupting",
    "replace", "replacing", "challenge", "challenging",
    "vs", "versus",
}

def _get_target_tokens(query: str) -> tuple[set[str], str]:
    """Return (token_set, ordered_text) for tokens that describe the search target.

    Excludes tokens inside subtrees of context verbs like "competing with",
    "disrupting", "replacing" — those describe the context, not the target.

    Returns both a set (for fast membership checks) and an order-preserving string
    (for multi-word phrase matching like "clean energy", "supply chain").

    Example:
      "fast-growing fintech companies competing with traditional banks in Europe"
      → excludes "competing", "with", "traditional", "banks"
      → "traditional" does NOT trigger year_founded_max=2000
    """
    doc = _load_nlp()(query)
    excluded: set[int] = set()
    for token in doc:
        if token.lemma_.lower() in CONTEXT_VERBS:
            for child in token.subtree:
                excluded.add(child.i)
    kept = [token for token in doc if token.i not in excluded]
    token_set  = {t.text.lower() for t in kept}
    token_text = " ".join(t.text.lower() for t in kept)
    return token_set, token_text

# ── Intent Parser ──────────────────────────────────────────────────────────────

def _parse_number(s: str) -> int:
    return int(s.replace(",", "").replace(".", "").strip())

def _parse_money(s: str) -> float:
    s = s.replace(",", "").strip()
    m = re.match(r"([\d.]+)\s*(b|billion|bn|m|million|mn|k|thousand)?", s, re.I)
    if not m:
        return 0.0
    val = float(m.group(1))
    suffix = (m.group(2) or "").lower()
    if suffix in ("b", "billion", "bn"):
        val *= 1e9
    elif suffix in ("m", "million", "mn"):
        val *= 1e6
    elif suffix in ("k", "thousand"):
        val *= 1e3
    return val

def parse_intent(query: str) -> Query:
    q  = Query(raw=query)
    ql = query.lower()

    # ── Country ──
    for name, code in COUNTRY_MAP.items():
        if re.search(rf"\b{re.escape(name)}\b", ql):
            q.country_codes.add(code)

    # ── Region (expands into country_codes for hard filter, region_countries for soft) ──
    for region, codes in REGION_MAP.items():
        if re.search(rf"\b{re.escape(region)}\b", ql):
            q.country_codes.update(codes)
            q.region_countries.update(codes)

    # ── Employee count ──
    m = re.search(r"(\d[\d,]*)\s*\+\s*(?:employees|people|staff)", ql)
    if m:
        q.employee_min = _parse_number(m.group(1))

    m = re.search(r"(?:more than|over|at least|>\s*)[\$\s]*([\d,]+)\s*(?:employees|people|staff)", ql)
    if m:
        q.employee_min = max(q.employee_min or 0, _parse_number(m.group(1)))

    m = re.search(r"(?:fewer than|under|less than|<\s*)([\d,]+)\s*(?:employees|people|staff)", ql)
    if m:
        q.employee_max = _parse_number(m.group(1))

    m = re.search(r"between\s+([\d,]+)\s+and\s+([\d,]+)\s+(?:employees|people|staff)", ql)
    if m:
        q.employee_min = _parse_number(m.group(1))
        q.employee_max = _parse_number(m.group(2))

    # Qualitative size hints (only set if not already specified by numbers)
    if re.search(r"\b(enterprise|fortune\s*500|large\s+compan)", ql):
        if not q.employee_min:
            q.employee_min = 1000
    elif re.search(r"\b(startup|startups|early-stage|small\s+business|sme)\b", ql):
        if not q.employee_max:
            q.employee_max = 200

    # ── Revenue ──
    m = re.search(
        r"(?:revenue|sales|arr|turnover)\s*(?:over|above|more than|exceeding|of at least|>\s*)"
        r"\s*\$?([\d,.]+\s*(?:billion|million|bn|mn|[bmkBMK])?)",
        ql,
    )
    if m:
        q.revenue_min = _parse_money(m.group(1))

    m = re.search(
        r"(?:revenue|sales|arr|turnover)\s*(?:under|below|less than|<\s*)"
        r"\s*\$?([\d,.]+\s*(?:billion|million|bn|mn|[bmkBMK])?)",
        ql,
    )
    if m:
        q.revenue_max = _parse_money(m.group(1))

    # ── Public / Private ──
    if re.search(r"\bpublic(?:ly)?\s*(?:traded|listed|compan)?", ql):
        q.is_public = True
    if re.search(r"\bprivate(?:ly)?\s*(?:held|owned|compan)?", ql):
        q.is_public = False

    # ── Business model ──
    for keyword, canonical in BM_QUERY_TO_CANONICAL.items():
        if re.search(rf"\b{re.escape(keyword)}\b", ql):
            q.business_models.add(canonical)

    # ── Industry / NAICS ──
    # Use only target tokens (spaCy) to avoid picking up industry keywords from
    # context clauses like "competing with traditional banks" → skip "banking".
    # token_text preserves word order so multi-word phrases ("clean energy",
    # "supply chain") match correctly; sorted() was breaking adjacency.
    target_tokens, target_text = _get_target_tokens(query)
    prefixes: list[str] = []
    kws: list[str] = []
    for kw, codes in NAICS_MAP.items():
        if re.search(rf"\b{re.escape(kw)}\b", target_text):
            prefixes.extend(codes)
            kws.append(kw)
    q.naics_prefixes = list(dict.fromkeys(prefixes))   # deduplicate, preserve order
    q.industry_keywords = list(dict.fromkeys(kws))

    # ── Year founded ──
    m = re.search(r"founded\s+(?:after|since)\s+(\d{4})", ql)
    if m:
        q.year_founded_min = int(m.group(1))
    m = re.search(r"founded\s+(?:before|prior\s+to)\s+(\d{4})", ql)
    if m:
        q.year_founded_max = int(m.group(1))
    m = re.search(r"founded\s+in\s+(\d{4})", ql)
    if m:
        q.year_founded_min = q.year_founded_max = int(m.group(1))

    # Qualitative year hints: only fire if the word describes the TARGET, not a competitor
    if not q.year_founded_min and any(
        w in target_tokens for w in ("startup", "startups", "early-stage")
    ):
        q.year_founded_min = 2010
    if not q.year_founded_max and any(
        w in target_tokens for w in ("established", "legacy", "traditional", "long-standing")
    ):
        q.year_founded_max = 2000

    return q

# ── Hard Filter ────────────────────────────────────────────────────────────────

def _passes_hard_filter(r: dict, q: Query) -> bool:
    """Return True if company passes all hard filters. Missing data = pass."""
    addr = r.get("address") or {}
    cc   = (addr.get("country_code") or "").lower()

    # Geography: only reject if company has data AND it doesn't match
    if q.country_codes and cc and cc not in q.country_codes:
        return False

    emp = r.get("employee_count")
    if emp is not None:
        if q.employee_min is not None and emp < q.employee_min:
            return False
        if q.employee_max is not None and emp > q.employee_max:
            return False

    rev = r.get("revenue")
    if rev is not None:
        if q.revenue_min is not None and rev < q.revenue_min:
            return False
        if q.revenue_max is not None and rev > q.revenue_max:
            return False

    if q.is_public is not None:
        pub = r.get("is_public")
        if pub is not None and pub != q.is_public:
            return False

    if q.business_models:
        bm_vals = {v for v in (r.get("business_model") or [])}
        if bm_vals and not q.business_models.intersection(bm_vals):
            return False

    # NAICS hard filter: only when industry is the PRIMARY signal and no other
    # numeric constraint (revenue, employees) is already narrowing the pool.
    # Conditions to enforce as hard:
    #   - no specific country named
    #   - AND business_model also confirms the industry ("B2B SaaS")
    #   - AND no revenue_min/employee_min already constraining (stacking too many
    #     hard filters on a small dataset kills all candidates)
    # Otherwise NAICS is soft — scored in _score_industry(), not filtered here.
    if q.naics_prefixes and q.business_models:
        specific_countries = q.country_codes - q.region_countries
        no_numeric_constraint = q.revenue_min is None and q.employee_min is None
        if not specific_countries and no_numeric_constraint:
            naics = r.get("primary_naics")
            code  = (naics.get("code", "") if isinstance(naics, dict) else "")
            if code and not any(code.startswith(p) for p in q.naics_prefixes):
                return False

    yr = r.get("year_founded")
    if yr is not None:
        if q.year_founded_min is not None and yr < q.year_founded_min:
            return False
        if q.year_founded_max is not None and yr > q.year_founded_max:
            return False

    return True

# ── Structured Scoring — sub-scorers ──────────────────────────────────────────

def _score_industry(r: dict, q: Query) -> float:
    if not q.naics_prefixes and not q.industry_keywords:
        return 0.5   # neutral: query has no industry signal

    naics = r.get("primary_naics")
    code  = (naics.get("code", "") if isinstance(naics, dict) else "")
    label = (naics.get("label", "") if isinstance(naics, dict) else "").lower()
    desc  = (r.get("description") or "").lower()
    offerings = " ".join(r.get("core_offerings") or []).lower()
    full_text = f"{label} {desc} {offerings}"

    # Exact NAICS prefix match → perfect score
    if code and any(code.startswith(p) for p in q.naics_prefixes):
        return 1.0

    # Keyword match in company text → partial
    hits = sum(1 for kw in q.industry_keywords if kw in full_text)
    if hits:
        return min(1.0, 0.4 + 0.3 * hits)

    return 0.0


def _score_location(r: dict, q: Query) -> float:
    if not q.country_codes:
        return 0.5   # neutral: no geographic intent
    addr = r.get("address") or {}
    cc   = (addr.get("country_code") or "").lower()
    if not cc:
        return 0.5   # no data → neutral

    if cc in q.country_codes:
        # Exact country match
        if cc not in q.region_countries:
            return 1.0   # explicitly named country
        else:
            return 0.85  # country matched via region expansion
    return 0.0


def _score_size(r: dict, q: Query) -> float:
    emp = r.get("employee_count")
    if emp is None:
        return 0.5   # neutral
    lo = q.employee_min
    hi = q.employee_max
    if lo is None and hi is None:
        return 0.5   # neutral
    if (lo is None or emp >= lo) and (hi is None or emp <= hi):
        return 1.0
    # Exponential decay proportional to how far outside the range
    if lo is not None and emp < lo:
        dist = lo - emp
        scale = max(lo, 1.0)
    else:
        dist = emp - (hi or emp)
        scale = max(hi or emp, 1.0)
    return math.exp(-dist / scale)


def _score_recency(r: dict) -> float:
    yr = r.get("year_founded")
    if yr is None:
        return 0.5
    return max(0.0, min(1.0, (float(yr) - YEAR_MIN) / (YEAR_MAX - YEAR_MIN)))


def _compute_structured_score(
    r: dict,
    cosine_raw: float,
    bm25_norm: float,
    q: Query,
) -> float:
    cosine_01 = (cosine_raw + 1.0) / 2.0   # inner product on normalized vecs: [-1,1] → [0,1]
    return (
        W_COSINE   * cosine_01
      + W_BM25     * bm25_norm
      + W_INDUSTRY * _score_industry(r, q)
      + W_LOCATION * _score_location(r, q)
      + W_SIZE     * _score_size(r, q)
      + W_RECENCY  * _score_recency(r)
    )

# ── Ranking Engine ─────────────────────────────────────────────────────────────

class RankingEngine:
    def __init__(self, data_path: pathlib.Path = DATA_PATH) -> None:
        print("Loading data...", flush=True)
        self.records: list[dict] = [
            json.loads(line) for line in data_path.read_text().splitlines() if line.strip()
        ]
        self.texts: list[str] = [company_to_text(r) for r in self.records]

        print("Loading models...", flush=True)
        self.embed_model   = SentenceTransformer(EMBED_MODEL)
        self.cross_encoder = CrossEncoder(CROSS_MODEL)

        self.embeddings, self.faiss_index = self._get_or_build_cache(data_path)
        self._expansion_cache: dict[str, str] = {}

        print("Building BM25 index...", flush=True)
        self.bm25 = BM25Okapi([t.lower().split() for t in self.texts])

        print("Ready.\n", flush=True)

    # ── Cache ──────────────────────────────────────────────────────────────

    def _cache_key(self, data_path: pathlib.Path) -> str:
        sha = hashlib.sha256(data_path.read_bytes()).hexdigest()[:16]
        model_slug = EMBED_MODEL.replace("/", "_")
        return f"{model_slug}_{sha}"

    def _get_or_build_cache(
        self, data_path: pathlib.Path
    ) -> tuple[np.ndarray, faiss.IndexFlatIP]:
        CACHE_DIR.mkdir(exist_ok=True)
        key        = self._cache_key(data_path)
        emb_path   = CACHE_DIR / f"{key}.npy"
        index_path = CACHE_DIR / f"{key}.faiss"

        if emb_path.exists() and index_path.exists():
            print("Loading cached embeddings...", flush=True)
            embeddings = np.load(str(emb_path))
            index      = faiss.read_index(str(index_path))
            return embeddings, index

        print(f"Computing embeddings for {len(self.records)} companies...", flush=True)
        raw = self.embed_model.encode(
            self.texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        embeddings = raw.astype(np.float32)
        np.save(str(emb_path), embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(index_path))
        return embeddings, index

    # ── Query Expansion ────────────────────────────────────────────────────

    def _expand_query(self, query: str) -> str:
        """
        Rewrite the query as a company description before FAISS embedding.

        Raw query text embeds near companies that *mention* the query topic.
        A rewritten description embeds near companies that *are* what the query asks for.

        Example:
          "companies supplying packaging for cosmetics brands"
          → "A packaging manufacturer producing bottles, jars, tubes, and labels
             supplied to cosmetics and personal care brands. B2B supplier in
             the packaging and containers industry."

        Falls back to the original query on any API error.
        Result is cached so repeated calls for the same query are free.
        """
        if query in self._expansion_cache:
            return self._expansion_cache[query]

        try:
            import openai
            client = openai.OpenAI()
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                max_tokens=120,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Rewrite search queries as company descriptions. "
                            "Describe the ideal matching company — its role, industry, activities, and business model. "
                            "Write as if describing the company itself, not a search. "
                            "Return only the description, no preamble."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Query: {query}",
                    },
                ],
            )
            expanded = resp.choices[0].message.content.strip()
            self._expansion_cache[query] = expanded
            print(f"  [QE] → {expanded}", flush=True)
            return expanded

        except Exception as exc:
            print(f"[WARN] Query expansion failed ({exc}), using original query.", flush=True)
            self._expansion_cache[query] = query
            return query

    # ── Stage 1: Hard Filter ───────────────────────────────────────────────

    def _hard_filter(self, q: Query) -> list[int]:
        """Return integer indices of companies that pass all hard filters."""
        return [i for i, r in enumerate(self.records) if _passes_hard_filter(r, q)]

    def _hard_filter_tiered(self, q: Query) -> list[int]:
        """
        Tiered fallback: progressively relax constraints rather than dumping the
        full unfiltered corpus (which produces garbage from semantic search).

          Tier 1 — full hard filter (exact match)
          Tier 2 — relax geography; keep NAICS, size, public, year filters
          Tier 3 — relax geography + NAICS; keep size, public, year
          Tier 4 — full unfiltered corpus (last resort)
        """
        result = self._hard_filter(q)
        if result:
            return result

        if q.country_codes:
            q2 = replace(q, country_codes=set(), region_countries=set())
            result = self._hard_filter(q2)
            if result:
                print(
                    "[NOTE] No companies found in the specified region — "
                    "showing industry matches globally.",
                    flush=True,
                )
                return result

        if q.naics_prefixes or q.business_models:
            q3 = replace(
                q,
                country_codes=set(),
                region_countries=set(),
                naics_prefixes=[],
                business_models=set(),
            )
            result = self._hard_filter(q3)
            if result:
                print(
                    "[NOTE] No exact industry+region match — "
                    "showing closest matches by size and other criteria.",
                    flush=True,
                )
                return result

        print("[WARN] Hard filter eliminated all companies even after relaxation. Returning unfiltered.", flush=True)
        return list(range(len(self.records)))

    # ── Stage 2: Candidate Generation ─────────────────────────────────────

    def _candidate_gen(
        self, q: Query, filtered_indices: list[int]
    ) -> list[tuple[int, float, float]]:
        """
        FAISS + BM25 → Reciprocal Rank Fusion.
        Returns (record_idx, faiss_score, bm25_raw_score) for top CANDIDATE_K.
        """
        filtered_set = set(filtered_indices)

        # -- Semantic (FAISS) --
        # Use the LLM-expanded description if available: it embeds near companies
        # that *are* what the query asks for, not just companies that mention the topic.
        faiss_query = q.expanded if q.expanded else q.raw
        q_emb = self.embed_model.encode(
            [faiss_query], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)
        # Retrieve everything, then filter — simple and correct at this scale
        scores_raw, indices_raw = self.faiss_index.search(q_emb, len(self.records))
        faiss_ranked = [
            (int(idx), float(score))
            for idx, score in zip(indices_raw[0], scores_raw[0])
            if int(idx) in filtered_set
        ]

        # -- Keyword (BM25) --
        # BM25 uses the original query — keyword matching benefits from exact user terms.
        bm25_all = self.bm25.get_scores(q.raw.lower().split())
        bm25_ranked = sorted(
            [(i, float(bm25_all[i])) for i in filtered_indices],
            key=lambda x: x[1],
            reverse=True,
        )

        # -- Reciprocal Rank Fusion --
        RRF_K = 60
        rrf: dict[int, float] = {}
        for rank, (idx, _) in enumerate(faiss_ranked, start=1):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (RRF_K + rank)
        for rank, (idx, _) in enumerate(bm25_ranked, start=1):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (RRF_K + rank)

        top_ids = sorted(rrf, key=lambda x: rrf[x], reverse=True)[:CANDIDATE_K]

        faiss_map = dict(faiss_ranked)
        bm25_map  = {idx: score for idx, score in bm25_ranked}
        return [(idx, faiss_map.get(idx, 0.0), bm25_map.get(idx, 0.0)) for idx in top_ids]

    # ── Stage 3: Structured Scoring ───────────────────────────────────────

    def _structured_score(
        self, q: Query, candidates: list[tuple[int, float, float]]
    ) -> list[tuple[int, float]]:
        """Score candidates, return top RERANK_TOP sorted descending."""
        if not candidates:
            return []

        # Normalize BM25 across this candidate set only
        bm25_vals = np.array([c[2] for c in candidates], dtype=np.float64)
        bm25_max  = bm25_vals.max()
        bm25_norm = bm25_vals / bm25_max if bm25_max > 0 else bm25_vals

        scored = [
            (
                idx,
                _compute_structured_score(
                    self.records[idx], faiss_score, float(bm_n), q
                ),
            )
            for (idx, faiss_score, _), bm_n in zip(candidates, bm25_norm)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:RERANK_TOP]

    # ── Stage 4: Cross-Encoder Re-ranking ─────────────────────────────────

    def _rerank(
        self, q: Query, top_stage3: list[tuple[int, float]]
    ) -> list[dict]:
        """Re-rank with cross-encoder, return top FINAL_TOP enriched result dicts."""
        if not top_stage3:
            return []

        pairs     = [(q.raw, self.texts[idx]) for idx, _ in top_stage3]
        ce_scores = self.cross_encoder.predict(pairs)

        merged = sorted(
            zip([idx for idx, _ in top_stage3], [s for _, s in top_stage3], ce_scores),
            key=lambda x: x[2],
            reverse=True,
        )

        results = []
        for rank, (idx, s3, ce) in enumerate(merged[:FINAL_TOP], start=1):
            r     = self.records[idx]
            addr  = r.get("address") or {}
            naics = r.get("primary_naics") or {}
            results.append({
                "rank":          rank,
                "name":          r.get("operational_name"),
                "website":       r.get("website"),
                "country":       (addr.get("country_code") or "").upper(),
                "region":        addr.get("region_name"),
                "employees":     int(r["employee_count"]) if r.get("employee_count") else None,
                "revenue":       r.get("revenue"),
                "naics_code":    naics.get("code") if isinstance(naics, dict) else None,
                "naics_label":   naics.get("label") if isinstance(naics, dict) else None,
                "year_founded":  int(r["year_founded"]) if r.get("year_founded") else None,
                "is_public":     r.get("is_public"),
                "business_model": r.get("business_model"),
                "description":   r.get("description"),
                "stage3_score":  round(float(s3), 4),
                "ce_score":      round(float(ce), 4),
            })
        return results

    # ── Stage 4b: LLM Re-ranking ───────────────────────────────────────────

    def _rerank_llm(
        self, q: Query, top_stage3: list[tuple[int, float]]
    ) -> list[dict]:
        """
        Re-rank top candidates using Claude as a qualification judge.

        All RERANK_TOP candidates are scored in a single API call (0-3).
        Falls back to cross-encoder if the API call fails for any reason.

        Score meaning:
          3 = Highly relevant — clearly satisfies the query intent
          2 = Relevant — matches the main criteria with minor gaps
          1 = Tangential — partial or borderline match
          0 = Not relevant
        """
        if not top_stage3:
            return []

        try:
            import openai
            client = openai.OpenAI()

            # Build condensed company summaries — enough context without blowing token budget
            lines: list[str] = []
            for i, (idx, _) in enumerate(top_stage3, start=1):
                r     = self.records[idx]
                name  = r.get("operational_name") or "?"
                addr  = r.get("address") or {}
                cc    = (addr.get("country_code") or "").upper()
                naics = r.get("primary_naics") or {}
                ind   = naics.get("label", "") if isinstance(naics, dict) else ""
                bm    = ", ".join(r.get("business_model") or [])
                desc  = (r.get("description") or "")[:200]
                lines.append(f"{i}. {name} [{cc}] | {ind} | {bm}\n   {desc}")

            companies_block = "\n\n".join(lines)

            prompt = (
                f'Query: "{q.raw}"\n\n'
                f"Score each company's relevance to the query.\n"
                f"0=not relevant, 1=tangential, 2=relevant, 3=highly relevant.\n\n"
                f"Be strict:\n"
                f"- A pharma company is NOT a generic 'manufacturing company' unless pharma is requested.\n"
                f"- A cosmetics company is NOT a packaging supplier.\n"
                f"- Match the specific industry, geography, size, and business model implied by the query.\n\n"
                f"Companies:\n{companies_block}\n\n"
                f"Return ONLY a JSON array (no markdown, no explanation):\n"
                f'[{{"id": 1, "score": 0}}, {{"id": 2, "score": 3}}, ...]'
            )

            resp = client.chat.completions.create(
                model=LLM_MODEL,
                max_tokens=512,
                temperature=0,   # deterministic — eliminates score variance on borderline companies
                messages=[
                    {"role": "system", "content": "You are a precise company qualification system. Return only valid JSON arrays."},
                    {"role": "user",   "content": prompt},
                ],
            )

            raw = resp.choices[0].message.content.strip()
            # Tolerate markdown fences the model might add
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            if not m:
                raise ValueError(f"No JSON array in LLM response: {raw[:200]}")
            scores_list = json.loads(m.group(0))
            scores: dict[int, float] = {item["id"]: float(item["score"]) for item in scores_list}
            print(
                f"  [LLM] scored {len(scores)}/{len(top_stage3)} candidates  "
                f"(model={LLM_MODEL})",
                flush=True,
            )

        except Exception as exc:
            print(f"[WARN] LLM rerank failed ({exc}), falling back to cross-encoder.", flush=True)
            return self._rerank(q, top_stage3)

        # Sort by LLM score descending; use stage3 structured score as tiebreaker
        merged = sorted(
            [
                (idx, s3, scores.get(i + 1, 0.0))
                for i, (idx, s3) in enumerate(top_stage3)
            ],
            key=lambda x: (x[2], x[1]),
            reverse=True,
        )

        results: list[dict] = []
        for rank, (idx, s3, llm_s) in enumerate(merged[:FINAL_TOP], start=1):
            r     = self.records[idx]
            addr  = r.get("address") or {}
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
                "ce_score":       round(llm_s, 4),   # reuses field; 0-3 scale
            })
        return results

    # ── Public entry point ─────────────────────────────────────────────────

    def rank(self, query: str, top_n: int = FINAL_TOP) -> list[dict]:
        """
        Full pipeline: parse → hard filter → FAISS+BM25+RRF → structured score → LLM rerank.
        Falls back to cross-encoder if the LLM call fails.
        Returns up to top_n result dicts, deduplicated by operational_name.
        """
        q = parse_intent(query)
        q.expanded = self._expand_query(query)

        filtered   = self._hard_filter_tiered(q)
        candidates = self._candidate_gen(q, filtered)
        if not candidates:
            return []

        top_stage3 = self._structured_score(q, candidates)
        results    = self._rerank_llm(q, top_stage3)

        # Deduplicate by operational_name (same company can appear with multiple records)
        seen: set[str] = set()
        deduped: list[dict] = []
        for r in results:
            key = (r.get("name") or "").lower().strip()
            if key and key not in seen:
                seen.add(key)
                deduped.append(r)

        # Re-number ranks after dedup
        for i, r in enumerate(deduped[:top_n], start=1):
            r["rank"] = i
        return deduped[:top_n]

# ── CLI ────────────────────────────────────────────────────────────────────────

SAMPLE_QUERIES = [
    # Original 5
    "Find logistics companies in Germany",
    "Public software companies with more than 1,000 employees",
    "B2B SaaS companies with annual revenue over $10M",
    "Fast-growing fintech companies competing with traditional banks in Europe",
    "Manufacturing companies in the DACH region founded before 2000",
    # Actual evaluation queries
    "Logistic companies in Romania",
    "Food and beverage manufacturers in France",
    "Companies that could supply packaging materials for a direct-to-consumer cosmetics brand",
    "Construction companies in the United States with revenue over $50 million",
    "Pharmaceutical companies in Switzerland",
    "B2B SaaS companies providing HR solutions in Europe",
    "Clean energy startups founded after 2018 with fewer than 200 employees",
    "Fast-growing fintech companies competing with traditional banks in Europe",
    "E-commerce companies using Shopify or similar platforms",
    "Renewable energy equipment manufacturers in Scandinavia",
    "Companies that manufacture or supply critical components for electric vehicle battery production",
]

def _fmt_result(r: dict) -> str:
    emp = f"emp={r['employees']:>7,}" if r["employees"] else "emp=      ?"
    rev = f"rev=${r['revenue']/1e6:>7.1f}M" if r["revenue"] else "rev=         ?"
    pub = "PUBLIC" if r["is_public"] else "private"
    naics = (r["naics_label"] or "")[:30]
    return (
        f"  {r['rank']:>2}. {(r['name'] or '?'):<36} [{r['country']:>2}] "
        f"{emp}  {rev}  {pub:<7}  score={r['ce_score']:+.3f}\n"
        f"      {naics}\n"
        f"      {textwrap.shorten(r['description'] or '', 100)}"
    )

if __name__ == "__main__":
    engine  = RankingEngine()
    queries = sys.argv[1:] if len(sys.argv) > 1 else SAMPLE_QUERIES

    for query in queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print("=" * 70)
        results = engine.rank(query)
        if not results:
            print("  (no results)")
        else:
            for r in results:
                print(_fmt_result(r))
