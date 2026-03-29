def _size_label(emp: int) -> str:
    if emp < 10:    return "micro company"
    if emp < 50:    return "small company"
    if emp < 200:   return "small to medium company"
    if emp < 1000:  return "mid-market company"
    if emp < 5000:  return "large company"
    return "large enterprise"


def _revenue_label(rev: float) -> str:
    if rev < 1e6:    return "early stage revenue"
    if rev < 10e6:   return "small business revenue"
    if rev < 100e6:  return "mid-market revenue"
    if rev < 1e9:    return "large company revenue"
    return "enterprise revenue"


def _recency_label(yr: int) -> str:
    if yr >= 2015:  return "young startup"
    if yr >= 2005:  return "growth stage company"
    if yr >= 1990:  return "established company"
    return "long-standing legacy company"


def company_to_text(record: dict) -> str:
    parts = []

    # Backbone — always present
    if record.get("operational_name"):
        parts.append(record["operational_name"])
    if isinstance(record.get("primary_naics"), dict):
        parts.append(f"Industry: {record['primary_naics']['label']}")
    if record.get("description"):
        parts.append(record["description"])
    if record.get("core_offerings"):
        parts.append(f"Offerings: {', '.join(record['core_offerings'])}")
    if record.get("target_markets"):
        parts.append(f"Markets: {', '.join(record['target_markets'])}")
    if record.get("business_model"):
        parts.append(f"Business model: {', '.join(record['business_model'])}")

    # Address — town + country only (street/postcode add noise, not semantic signal)
    addr = record.get("address", {})
    if isinstance(addr, dict):
        loc = ", ".join(filter(None, [
            addr.get("town"),
            addr.get("country_name") or (addr.get("country_code") or "").upper(),
        ]))
        if loc:
            parts.append(f"Location: {loc}")

    # Partials — only if present, never placeholder text
    if record.get("employee_count"):
        emp = int(record["employee_count"])
        parts.append(f"Employees: {emp} ({_size_label(emp)})")
    if record.get("revenue"):
        rev = record["revenue"]
        parts.append(f"Revenue: ${rev:,.0f} ({_revenue_label(rev)})")
    if record.get("year_founded"):
        yr = int(record["year_founded"])
        parts.append(f"Founded: {yr} ({_recency_label(yr)})")
    if record.get("is_public") is True:
        parts.append("Public company")

    return ". ".join(parts)


if __name__ == "__main__":
    import json
    import pathlib

    input_path = pathlib.Path("companies_clean.jsonl")
    output_path = pathlib.Path("companies_text.jsonl")

    with input_path.open() as fin, output_path.open("w") as fout:
        for line in fin:
            record = json.loads(line)
            text = company_to_text(record)
            json.dump({"website": record.get("website"), "text": text}, fout)
            fout.write("\n")

    print(f"Done. Written to {output_path}")
