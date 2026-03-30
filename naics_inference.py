"""
naics_inference.py — Conservative NAICS label inference from core_offerings.

Problem this solves
-------------------
Each company in the dataset has a single primary_naics code assigned by whoever
built the dataset. That code is what the company was *classified as*, not
necessarily everything it *does*. A fintech company whose primary NAICS is
522320 (Financial Transactions Processing) also builds software — but its NAICS
code alone would give it a score of 0.0 for a "SaaS companies" query.

Why not use NAICS_MAP from solution.py
---------------------------------------
NAICS_MAP was designed for query parsing: it maps broad single-word keywords
("transportation", "gas", "software") to NAICS prefixes so that user queries
resolve to a wide set of candidate codes. Broad recall is correct for queries.

For company labeling, broad recall is wrong. Scanning a company's offerings
for "transportation" infers logistics NAICS on any company that mentions
"Transportation Services" in a name — which is almost every company. The
result is 400+ companies getting logistics labels, making the signal useless.

This module uses a separate NAICS_INFERENCE_MAP built from specific multi-word
phrases grounded in the actual core_offerings values in the dataset. Each
entry requires a precise compound phrase that unambiguously identifies the
industry — no single generic words allowed.

Usage
-----
    from naics_inference import infer_naics_from_offerings

    # At index time, enrich each record once:
    for r in records:
        r["_inferred_naics"] = infer_naics_from_offerings(r)

    # In _score_industry, check both primary and inferred:
    all_codes = [primary_code] + r.get("_inferred_naics", [])
    if any(any(c.startswith(p) for p in q.naics_prefixes) for c in all_codes):
        return 0.85  # inferred match — slightly below primary (1.0)
"""

from __future__ import annotations

import json
import pathlib
import re
from collections import defaultdict

# ── Conservative inference map ─────────────────────────────────────────────────
#
# Rules for adding entries:
#   1. Phrase must be 2+ words — bare nouns ("software", "logistics") are banned
#   2. Phrase must unambiguously identify the industry — "product development"
#      is not allowed; "payroll software development" is
#   3. Entries are grounded in actual core_offerings strings from companies_clean.jsonl
#   4. NAICS codes here use 2022 revision (513210 for Software Publishers, etc.)

NAICS_INFERENCE_MAP: dict[str, list[str]] = {

    # ── HR / Workforce Software ────────────────────────────────────────────────
    # Most common cluster in the dataset — 17 entries for "payroll management
    # software development" alone. Treat all HR-flavored software as 513210.
    "payroll management software":      ["513210", "541214"],
    "payroll software":                 ["513210", "541214"],
    "recruitment management software":  ["513210", "5613"],
    "talent management software":       ["513210", "5613"],
    "hr management software":           ["513210"],
    "hr software":                      ["513210"],
    "performance management software":  ["513210"],
    "employee onboarding software":     ["513210"],
    "time tracking software":           ["513210"],
    "workforce management software":    ["513210"],
    "training management software":     ["513210"],
    "applicant tracking":               ["513210", "5613"],
    "learning management system":       ["513210", "61"],
    "hr consulting":                    ["541612", "5613"],

    # ── SaaS / Cloud / General Software ───────────────────────────────────────
    "saas platform":                    ["513210"],
    "cloud platform":                   ["513210", "518210"],
    "cloud software":                   ["513210"],
    "mobile application":               ["513210"],
    "machine learning model":           ["513210", "541715"],
    "deep learning model":              ["513210", "541715"],
    "ai consulting":                    ["513210", "5416"],
    "cybersecurity solutions":          ["513210", "541512"],
    "digital transformation":           ["513210", "5416"],
    "it consulting":                    ["541512", "5416"],
    "api integration":                  ["513210"],
    "data analytics platform":          ["513210", "518210"],

    # ── Fintech / Payments ─────────────────────────────────────────────────────
    "payment processing":               ["522320"],
    "payment api":                      ["522320", "513210"],
    "digital banking":                  ["522110", "513210"],
    "money transfer":                   ["522320"],
    "foreign exchange":                 ["522320"],
    "remittance service":               ["522320"],
    "cryptocurrency":                   ["523130"],
    "blockchain payment":               ["522320", "513210"],
    "insurance platform":               ["524", "513210"],
    "lending platform":                 ["522110", "513210"],

    # ── Logistics / Supply Chain ───────────────────────────────────────────────
    "freight transport":                ["484"],
    "freight forwarding":               ["484"],
    "freight rail transport":           ["482110"],
    "trucking service":                 ["484"],
    "last mile delivery":               ["492110"],
    "warehouse management":             ["493110"],
    "cold chain":                       ["493120"],
    "supply chain management":          ["484", "493"],
    "fleet management":                 ["484"],
    "route optimization":               ["484", "513210"],

    # ── Renewable / Clean Energy ───────────────────────────────────────────────
    "solar panel manufacturing":        ["334413"],
    "solar panel":                      ["334413"],
    "wind turbine manufacturing":       ["333611"],
    "wind turbine":                     ["333611"],
    "renewable energy solutions":       ["221114", "221115"],
    "renewable energy semiconductor":   ["334413"],
    "battery storage":                  ["335911"],
    "energy storage":                   ["335911"],
    "offshore wind":                    ["221115"],
    "tidal energy":                     ["221116"],
    "hydropower":                       ["221111"],

    # ── Fossil / Oil & Gas ─────────────────────────────────────────────────────
    "oil refining":                     ["324110"],
    "natural gas pipeline":             ["486210"],
    "petroleum product":                ["324110"],
    "gas storage facility":             ["493190"],

    # ── Pharma / Biotech ───────────────────────────────────────────────────────
    "drug manufacturing":               ["325412"],
    "pharmaceutical manufacturing":     ["325412"],
    "clinical trial":                   ["541714", "325412"],
    "active pharmaceutical":            ["325411"],
    "cosmetic packaging manufacturing": ["322220", "326160"],

    # ── Construction / Real Estate ─────────────────────────────────────────────
    "construction project management":  ["236", "237"],
    "real estate development":          ["531"],
    "property management":              ["531311"],

    # ── Professional Services ──────────────────────────────────────────────────
    "management consulting":            ["541611"],
    "sustainability consulting":        ["541620"],
    "environmental consulting":         ["541620"],
    "regulatory compliance consulting": ["541690"],
    "marketing automation":             ["513210", "5418"],
}


# ── Inference function ─────────────────────────────────────────────────────────

def infer_naics_from_offerings(r: dict) -> list[str]:
    """
    Return a deduplicated list of NAICS prefixes inferred from the company's
    core_offerings by matching against NAICS_INFERENCE_MAP.

    Scans each offering string individually (not concatenated) so that a phrase
    cannot be formed by accidentally bridging two separate offerings.

    Returns an empty list if core_offerings is absent or no phrases match.
    """
    offerings = r.get("core_offerings") or []
    if not offerings:
        return []

    seen: set[str] = set()
    result: list[str] = []
    for offering in offerings:
        text = offering.lower()
        for phrase, codes in NAICS_INFERENCE_MAP.items():
            if re.search(rf"\b{re.escape(phrase)}\b", text):
                for code in codes:
                    if code not in seen:
                        seen.add(code)
                        result.append(code)
    return result


# ── Demo ───────────────────────────────────────────────────────────────────────

def run_demo(data_path: pathlib.Path = pathlib.Path("companies_clean.jsonl")) -> None:
    """
    Load the dataset, infer secondary NAICS for every company, and report:
      - How many companies gained at least one new label
      - Sample companies where the inferred labels differ from primary
      - Coverage: how many unique NAICS codes are inferred vs zero
    """
    records = [json.loads(l) for l in data_path.read_text().splitlines() if l.strip()]

    gained: list[dict] = []
    no_match: int = 0

    for r in records:
        inferred = infer_naics_from_offerings(r)
        r["_inferred_naics"] = inferred

        naics = r.get("primary_naics") or {}
        primary = naics.get("code", "") if isinstance(naics, dict) else ""

        # Only count as "gained" if inferred adds something beyond the primary
        new = [p for p in inferred if not (primary.startswith(p) or (p and primary.startswith(p[:3])))]
        if new:
            gained.append({
                "name":     r.get("operational_name"),
                "primary":  primary,
                "label":    (naics.get("label", "") if isinstance(naics, dict) else "")[:40],
                "new":      new,
                "offerings": (r.get("core_offerings") or [])[:3],
            })
        elif not inferred:
            no_match += 1

    print(f"Total companies      : {len(records)}")
    print(f"Gained new labels    : {len(gained)}")
    print(f"No offerings matched : {no_match}")
    print()
    print("Sample — companies with new inferred NAICS:")
    print("─" * 70)
    for g in gained[:12]:
        print(f"  {g['name']}")
        print(f"    primary  : {g['primary']} ({g['label']})")
        print(f"    new      : {g['new']}")
        print(f"    offerings: {g['offerings']}")
        print()


if __name__ == "__main__":
    run_demo()
