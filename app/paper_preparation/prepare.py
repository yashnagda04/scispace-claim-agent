"""Preserve CSV records and assign stable alphanumeric IDs once."""

import csv
import re
import secrets
import string
from pathlib import Path

ALPHABET = string.ascii_uppercase + string.digits
ID_CAPACITY = 36**4 - 26**4 - 10**4


def valid_paper_id(value):
    return bool(
        isinstance(value, str)
        and re.fullmatch(r"[A-Z0-9]{4}", value)
        and any(c.isalpha() for c in value)
        and any(c.isdigit() for c in value)
    )


def generate_paper_id(used):
    if len(used) >= ID_CAPACITY:
        raise ValueError("The four-character paper ID space is exhausted.")
    while True:
        candidate = "".join(secrets.choice(ALPHABET) for _ in range(4))
        if valid_paper_id(candidate) and candidate not in used:
            return candidate


def read_papers(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if not fields or len(fields) != len(set(fields)):
            raise ValueError("Paper CSV requires unique column names.")
        if "Paper Title" not in fields:
            raise ValueError("Paper CSV requires a Paper Title column.")
        rows = list(reader)
    if not rows or any(None in row or None in row.values() for row in rows):
        raise ValueError("Paper CSV is empty or has rows with incorrect column counts.")
    return fields, rows


def prepare_papers(source, destination):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Prepared papers must use a different path from the original input.")
    fields, original = read_papers(source)
    source_fields = [name for name in fields if name != "paper_id"]
    if destination.exists():
        saved_fields, saved = read_papers(destination)
        ids = [row.get("paper_id") for row in saved]
        if not all(valid_paper_id(pid) for pid in ids) or len(set(ids)) != len(ids):
            raise ValueError("Prepared collection has missing, invalid, or duplicate paper IDs.")
        same_content = (
            [name for name in saved_fields if name != "paper_id"] == source_fields
            and len(saved) == len(original)
            and all(
                all(old[name] == new[name] for name in source_fields)
                for old, new in zip(original, saved)
            )
        )
        if not same_content:
            raise ValueError("Input papers changed. Choose a new --prepared path for the new collection.")
        return saved
    if len(original) > ID_CAPACITY:
        raise ValueError("Too many records for four-character paper IDs.")
    used = set()
    for row in original:
        row["paper_id"] = generate_paper_id(used)
        used.add(row["paper_id"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["paper_id", *source_fields])
        writer.writeheader()
        writer.writerows(original)
    temporary.replace(destination)
    return original
