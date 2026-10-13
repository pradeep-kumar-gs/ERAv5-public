"""Regenerate the README's date-source table from site/data/mechanisms.json (never hand-typed).

    python3 s8/scripts/sources_table.py
"""
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
data = json.loads((ROOT / "site" / "data" / "mechanisms.json").read_text())

rows = ["| Date | Mechanism | Primary source | How the date was checked |", "|---|---|---|---|"]
for m in data["mechanisms"]:
    src = m["source"]
    how = f"arXiv {src['id']} v1 (arXiv API)" if src["kind"] == "arxiv" else src.get("verified_via", "Publication timestamp on the page")
    req = "" if m["required"] else " *(added)*"
    title = src["title"].replace("|", "/")
    rows.append(f"| {m['date']} | {m['name']}{req} | [{title}]({src['url']}) | {how} |")

readme = ROOT / "README.md"
text = readme.read_text()
block = "<!-- SOURCES_TABLE_START -->\n" + "\n".join(rows) + "\n<!-- SOURCES_TABLE_END -->"
readme.write_text(re.sub(r"<!-- SOURCES_TABLE_START -->.*<!-- SOURCES_TABLE_END -->", lambda _: block, text, flags=re.S))
print(f"wrote {len(rows) - 2} rows")
