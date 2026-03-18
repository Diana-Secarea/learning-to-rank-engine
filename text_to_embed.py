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

    # Address — always present but structured
    addr = record.get("address", {})
    if isinstance(addr, dict):
        loc_parts = filter(None, [
            addr.get("house_number"),
            addr.get("street"),
            addr.get("suburb"),
            addr.get("town"),
            addr.get("county"),
            addr.get("region_name"),
            addr.get("postcode"),
            addr.get("country_name") or (addr.get("country_code") or "").upper(),
        ])
        loc = ", ".join(loc_parts)
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
