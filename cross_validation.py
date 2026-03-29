"""
cross_validation.py — Cross-attribute consistency checker for retrieved companies.

For each (query, company) pair produced by the pipeline, this module validates
that the company's full attribute profile is internally consistent with its
assigned NAICS label, and plausible for the query's intent.

Two types of checks are performed:

  1. Internal consistency  — does the company's own data make sense?
       e.g. NAICS=5112 (Software) but business_model=["Manufacturing"] only → suspicious
       e.g. NAICS=3254 (Pharma) but core_offerings mention only "retail" → suspicious

  2. Query plausibility    — is this company a reasonable answer to this query?
       e.g. query asks for "manufacturing companies" but company is pharma → flag unless
            pharma is explicitly in scope
       e.g. query asks for "packaging suppliers" but company is a cosmetics *brand* → reject

Each check returns a ValidationResult with:
  - is_plausible  : bool — False means the company should be demoted or filtered
  - flags         : list of human-readable issues found
  - confidence    : float 0-1 — 1.0 = fully consistent, 0.0 = clearly wrong

Usage:
  from cross_validation import validate
  result = validate(query_str, company_dict)
  if not result.is_plausible:
      print(result.flags)
"""

from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass, field

# ── NAICS taxonomy consistency rules ──────────────────────────────────────────
#
# Maps a NAICS prefix to the set of business_model values that are EXPECTED
# for that sector. If a company has none of these, it's flagged.
#
# Also maps to INCOMPATIBLE business models — if a company has ALL its models
# in the incompatible set it's flagged as suspicious.

NAICS_EXPECTED_BM: dict[str, set[str]] = {
    # Software / SaaS
    "511":  {"Software-as-a-Service", "Business-to-Business", "Subscription-Based",
              "Enterprise", "Business-to-Consumer"},
    "5112": {"Software-as-a-Service", "Business-to-Business", "Subscription-Based",
              "Enterprise", "Business-to-Consumer"},
    "518":  {"Software-as-a-Service", "Business-to-Business", "Enterprise"},
    # Manufacturing (broad)
    "31":   {"Manufacturing", "Wholesale", "Business-to-Business"},
    "32":   {"Manufacturing", "Wholesale", "Business-to-Business"},
    "33":   {"Manufacturing", "Wholesale", "Business-to-Business"},
    # Pharma / biotech
    "3254": {"Manufacturing", "Wholesale", "Business-to-Business", "Research"},
    "5417": {"Research", "Business-to-Business"},
    # Logistics / transport
    "484":  {"Logistics/Transportation", "Business-to-Business", "Wholesale"},
    "488":  {"Logistics/Transportation", "Business-to-Business"},
    "493":  {"Logistics/Transportation", "Wholesale", "Business-to-Business"},
    # Finance
    "52":   {"Business-to-Business", "Business-to-Consumer", "Enterprise"},
    "522":  {"Business-to-Business", "Business-to-Consumer"},
    "524":  {"Business-to-Business", "Business-to-Consumer"},
    # Retail / e-commerce
    "44":   {"Retail", "Business-to-Consumer", "E-commerce"},
    "45":   {"Retail", "Business-to-Consumer", "E-commerce"},
    "454":  {"E-commerce", "Retail", "Business-to-Consumer"},
    # Energy
    "211":  {"Manufacturing", "Wholesale", "Business-to-Business"},
    "221":  {"Business-to-Business", "Business-to-Consumer"},
    # Consulting / professional services
    "5416": {"Business-to-Business", "Enterprise", "Service Provider"},
    "5415": {"Business-to-Business", "Enterprise", "Software-as-a-Service"},
    # Healthcare
    "62":   {"Business-to-Consumer", "Business-to-Business", "Service Provider"},
    # Construction
    "23":   {"Business-to-Business", "Business-to-Consumer", "Manufacturing"},
}

# NAICS prefixes that represent manufacturing — used for query "manufacturing" checks
MANUFACTURING_NAICS = {"31", "32", "33", "3254", "3361", "3362", "3363", "3364", "311", "312", "313"}

# NAICS prefixes for software/tech
# Note: 2022 NAICS revision renamed 511210 → 513210 (Software Publishers)
SOFTWARE_NAICS = {"511", "5112", "513", "5132", "5415", "518", "519"}

# NAICS prefixes for logistics/transport
LOGISTICS_NAICS = {"484", "488", "493", "481", "482", "483", "487"}

# NAICS prefixes for finance
FINANCE_NAICS = {"52", "521", "522", "523", "524", "525"}

# ── Query intent → plausibility rules ─────────────────────────────────────────
#
# If the query signals one of these intents, companies with certain NAICS prefixes
# are implausible unless explicitly expanded by the query.

QUERY_INTENT_RULES: list[dict] = [
    {
        "intent_keywords": ["manufacturing", "manufacturer", "manufacturers"],
        "required_naics_prefixes": MANUFACTURING_NAICS,
        "exclusion_message": (
            "Company is not a manufacturer (NAICS {code}: {label}). "
            "Manufacturing queries require NAICS 31–33 or similar."
        ),
        "exceptions": ["pharma", "pharmaceutical", "biotech", "chemical", "food", "beverage"],
    },
    {
        "intent_keywords": ["saas", "software-as-a-service"],
        "required_naics_prefixes": SOFTWARE_NAICS,
        "exclusion_message": (
            "Company NAICS {code} ({label}) is not a software company. "
            "SaaS queries expect NAICS 511x/513x/541x/518x."
        ),
        "exceptions": [],
    },
    {
        "intent_keywords": ["logistics", "trucking", "freight", "shipping", "warehousing"],
        "required_naics_prefixes": LOGISTICS_NAICS,
        "exclusion_message": (
            "Company NAICS {code} ({label}) is not a logistics/transport company. "
            "Expected NAICS 484/488/493 or similar."
        ),
        "exceptions": ["supply chain", "distribution"],
    },
    {
        "intent_keywords": ["packaging", "packaging supplier", "packaging materials"],
        "required_naics_prefixes": {"322", "326", "3221", "3222", "3261", "3262", "3089", "4239"},
        "exclusion_message": (
            "Company NAICS {code} ({label}) does not appear to be a packaging supplier. "
            "Expected NAICS 322x (paper/paperboard) or 326x (plastics)."
        ),
        "exceptions": ["cosmetics brand", "beauty brand", "consumer goods"],
        # Special rule: if company IS a cosmetics/beauty brand, reject even harder
        "reject_if_core_offerings_contain": ["cosmetics", "beauty product", "skincare", "makeup"],
    },
    {
        "intent_keywords": ["fintech", "financial technology"],
        "required_naics_prefixes": FINANCE_NAICS | SOFTWARE_NAICS,
        "exclusion_message": (
            "Company NAICS {code} ({label}) does not match fintech intent. "
            "Expected finance (52x) or software (511x) sector."
        ),
        "exceptions": [],
    },
]

# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    is_plausible: bool
    confidence:   float                  # 1.0 = fully consistent
    flags:        list[str] = field(default_factory=list)
    company_name: str = ""
    naics_code:   str = ""
    naics_label:  str = ""

    def __str__(self) -> str:
        status = "OK" if self.is_plausible else "SUSPICIOUS"
        lines  = [f"[{status}] {self.company_name}  (NAICS {self.naics_code}: {self.naics_label})  conf={self.confidence:.2f}"]
        for f in self.flags:
            lines.append(f"  ⚠  {f}")
        return "\n".join(lines)


# ── Core validation logic ──────────────────────────────────────────────────────

def _naics_prefix_match(code: str, prefixes: set[str]) -> bool:
    return any(code.startswith(p) for p in prefixes)


def _extract_naics(company: dict) -> tuple[str, str]:
    """Extract (code, label) from either pipeline output or raw record format."""
    # Pipeline output uses flat keys
    if "naics_code" in company:
        return (company.get("naics_code") or ""), (company.get("naics_label") or "")
    # Raw record uses nested dict
    naics = company.get("primary_naics") or {}
    if isinstance(naics, dict):
        return naics.get("code", ""), naics.get("label", "")
    return "", ""


def _check_internal_consistency(company: dict) -> list[str]:
    """
    Check that the company's own fields are internally consistent.
    Returns a list of flag strings (empty = no issues).
    """
    flags: list[str] = []

    code, label_raw = _extract_naics(company)
    label = label_raw.lower()
    bm_vals   = set(company.get("business_model") or [])
    desc      = (company.get("description") or "").lower()
    offerings = " ".join(company.get("core_offerings") or []).lower()
    targets   = " ".join(company.get("target_markets") or []).lower()
    full_text = f"{label} {desc} {offerings} {targets}"

    if not code:
        flags.append("Missing NAICS code — cannot verify taxonomy consistency.")
        return flags

    # Check expected business models for this NAICS prefix
    expected: set[str] = set()
    for prefix, bms in NAICS_EXPECTED_BM.items():
        if code.startswith(prefix):
            expected |= bms
            break

    if expected and bm_vals and not bm_vals.intersection(expected):
        flags.append(
            f"Business model mismatch: NAICS {code} ({label}) expects one of "
            f"{sorted(expected)} but company has {sorted(bm_vals)}."
        )

    # Software NAICS but no tech signal in text
    if _naics_prefix_match(code, SOFTWARE_NAICS):
        tech_signals = ["software", "saas", "platform", "application", "app",
                        "cloud", "api", "digital", "technology", "tech", "data"]
        if not any(sig in full_text for sig in tech_signals):
            flags.append(
                f"NAICS {code} indicates software/tech but company description "
                f"contains no tech-related terms."
            )

    # Manufacturing NAICS but business model is purely service/retail
    if _naics_prefix_match(code, MANUFACTURING_NAICS):
        service_only = {"Service Provider", "Retail", "Business-to-Consumer", "E-commerce"}
        if bm_vals and bm_vals.issubset(service_only) and "Manufacturing" not in bm_vals:
            flags.append(
                f"NAICS {code} indicates manufacturing but all business models are "
                f"service/retail: {sorted(bm_vals)}."
            )

    # Logistics NAICS but no logistics signal in text
    if _naics_prefix_match(code, LOGISTICS_NAICS):
        logistics_signals = ["logistic", "transport", "freight", "shipping", "delivery",
                             "warehousing", "cargo", "supply chain", "trucking"]
        if not any(sig in full_text for sig in logistics_signals):
            flags.append(
                f"NAICS {code} indicates logistics but no logistics terms found in company profile."
            )

    return flags


def _check_query_plausibility(query: str, company: dict) -> list[str]:
    """
    Check that the company is a plausible result for the given query.
    Returns a list of flag strings (empty = no issues).
    """
    flags:    list[str] = []
    q_lower   = query.lower()
    code, label_raw = _extract_naics(company)
    label = label_raw.lower()
    offerings = " ".join(company.get("core_offerings") or []).lower()
    desc      = (company.get("description") or "").lower()
    full_text = f"{label} {desc} {offerings}"

    for rule in QUERY_INTENT_RULES:
        # Check if query triggers this rule
        triggered = any(kw in q_lower for kw in rule["intent_keywords"])
        if not triggered:
            continue

        # Check if any exception keyword appears in the query — relaxes the rule
        exceptions = rule.get("exceptions", [])
        if any(exc in q_lower for exc in exceptions):
            continue

        # Hard reject: company's core offerings reveal it's the wrong entity type
        reject_signals = rule.get("reject_if_core_offerings_contain", [])
        if reject_signals and any(sig in offerings or sig in desc for sig in reject_signals):
            flags.append(
                f"Query '{query[:60]}' asks for {rule['intent_keywords'][0]} suppliers/companies "
                f"but this company appears to be an end-consumer of those products "
                f"(found signals: {[s for s in reject_signals if s in offerings or s in desc]})."
            )
            continue

        # Check if NAICS matches the required prefixes for this intent
        if code and not _naics_prefix_match(code, rule["required_naics_prefixes"]):
            msg = rule["exclusion_message"].format(code=code, label=label)
            flags.append(msg)

    return flags


def validate(query: str, company: dict) -> ValidationResult:
    """
    Full cross-attribute validation for a (query, company) pair.

    Returns a ValidationResult with is_plausible, confidence, and flags.
    """
    code, label = _extract_naics(company)
    name        = company.get("operational_name") or company.get("name") or "?"

    internal_flags = _check_internal_consistency(company)
    query_flags    = _check_query_plausibility(query, company)
    all_flags      = internal_flags + query_flags

    # Confidence: starts at 1.0, penalised per flag
    # Internal inconsistency = -0.15 per flag (company data is wrong regardless of query)
    # Query plausibility fail = -0.35 per flag (company is wrong for this query)
    confidence = 1.0
    confidence -= 0.15 * len(internal_flags)
    confidence -= 0.35 * len(query_flags)
    confidence = max(0.0, round(confidence, 3))

    # A company is implausible if confidence drops below 0.5
    # or if any query-plausibility flag was raised (strict mode for query mismatch)
    is_plausible = confidence >= 0.5 and len(query_flags) == 0

    return ValidationResult(
        is_plausible=is_plausible,
        confidence=confidence,
        flags=all_flags,
        company_name=name,
        naics_code=code,
        naics_label=label,
    )


# ── Demo: run against eval queries and pipeline results ───────────────────────

def run_demo(data_path: pathlib.Path = pathlib.Path("companies_clean.jsonl")) -> None:
    """
    Load the pipeline, run a set of queries, and cross-validate the top results.
    Shows which results pass/fail the consistency checks.
    """
    from dotenv import load_dotenv
    load_dotenv()

    from solution import RankingEngine

    engine = RankingEngine()

    demo_queries = [
        "Manufacturing companies in the DACH region founded before 2000",
        "B2B SaaS companies with annual revenue over $10M",
        "Companies that could supply packaging materials for a direct-to-consumer cosmetics brand",
        "Fast-growing fintech companies competing with traditional banks in Europe",
        "Find logistics companies in Germany",
        "Pharmaceutical companies in Switzerland",
    ]

    for query in demo_queries:
        print("\n" + "═" * 70)
        print(f"Query: {query}")
        print("═" * 70)

        results = engine.rank(query)
        plausible_count   = 0
        implausible_count = 0

        for i, r in enumerate(results, start=1):
            result = validate(query, r)
            marker = "✓" if result.is_plausible else "✗"
            print(f"\n  {i:>2}. [{marker}] {result.company_name:<35} "
                  f"NAICS={result.naics_code:<8} conf={result.confidence:.2f}")
            for flag in result.flags:
                print(f"       ⚠  {flag}")
            if result.is_plausible:
                plausible_count += 1
            else:
                implausible_count += 1

        total = plausible_count + implausible_count
        print(f"\n  Summary: {plausible_count}/{total} plausible  |  "
              f"{implausible_count}/{total} flagged")


if __name__ == "__main__":
    run_demo()
