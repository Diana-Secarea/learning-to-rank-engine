import ast
import json
import sys


# ---------------------------------------------------------------------------
# Fix 1 – Invalid JSON structure: Python repr() dict strings
#
# Fields like `address` and `primary_naics` are stored as Python repr()
# output (single-quoted dict strings) instead of real JSON objects, e.g.:
#
#   "address": "{'country_code': 'ro', 'latitude': 44.47, ...}"
#
# JSON sees this as a plain string, so record['address']['country_code']
# raises TypeError: string indices must be integers.
# ast.literal_eval() safely parses these Python-style dicts back to real dicts.
# ---------------------------------------------------------------------------

DICT_FIELDS = ("address", "primary_naics")


def parse_dict_field(value):
    """Convert a Python repr() dict string to a real dict, if needed."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.startswith("{"):
        return ast.literal_eval(value)  # handles single-quote Python dicts
    return value


def clean_record(record: dict) -> dict:
    for field in DICT_FIELDS:
        if field in record:
            record[field] = parse_dict_field(record[field])
    return record


# ---------------------------------------------------------------------------
# Fix 3 – Drop secondary_naics as a feature
#
# Only 11 of 464 records have a non-null secondary_naics (~2.3%).
# Including it in ranking logic would create inconsistency — it influences
# only 11 companies while 453 have null. primary_naics and text fields
# (description, core_offerings) already capture multi-industry relevance.
# The field is kept in the record for display purposes, just zeroed out.
# ---------------------------------------------------------------------------

FIELDS_TO_DROP = ("secondary_naics",)


def drop_feature_fields(record: dict) -> dict:
    for field in FIELDS_TO_DROP:
        record.pop(field, None)
    return record


# ---------------------------------------------------------------------------
# Fix 2 – Duplicate records: same website, identical content
#
# 13 websites appear exactly twice with byte-for-byte identical records.
# Strategy: use website as the dedup key; if two records share a website,
# keep the one with more non-null fields (handles edge cases where they differ).
# Records with no website are kept as-is — can't dedup without a key.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Fix 4 – Duplicate records: no website, identical description
#
# 7 companies have no website and byte-for-byte identical records.
# dedup_by_website skipped them (no key). Use description as fallback key.
# ---------------------------------------------------------------------------


def dedup_by_description(records: list) -> list:
    seen = {}
    no_desc = []
    for r in records:
        key = (r.get("description") or "").strip()
        if not key:
            no_desc.append(r)
            continue
        if key not in seen:
            seen[key] = r
        else:
            if count_fields(r) > count_fields(seen[key]):
                seen[key] = r
    return list(seen.values()) + no_desc


def clean_file(input_path: str, output_path: str) -> None:
    cleaned = []
    with open(input_path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                cleaned.append(drop_feature_fields(clean_record(record)))
            except json.JSONDecodeError as exc:
                print(f"[WARN] line {lineno}: skipping – {exc}", file=sys.stderr)

    before = len(cleaned)
    cleaned = dedup_by_website(cleaned)
    cleaned = dedup_by_description(cleaned)
    after = len(cleaned)

    with open(output_path, "w", encoding="utf-8") as fh:
        for record in cleaned:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Cleaned {before} records, removed {before - after} duplicates → {after} records saved to {output_path}")


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "companies.jsonl"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "companies_clean.jsonl"
    clean_file(input_file, output_file)
