"""Verify every arXiv-sourced date in site/data/mechanisms.json against the arXiv API.

The date shown on the timeline is the arXiv v1 submission date. This script
fetches the `published` field (v1 timestamp) and title for each arXiv id and
fails if the JSON disagrees. Non-arXiv sources (blog posts, GitHub releases,
Reddit) are listed for manual checking.

    python3 s8/scripts/verify_dates.py
"""
import json
import pathlib
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "site" / "data" / "mechanisms.json"
NS = {"a": "http://www.w3.org/2005/Atom"}


def fetch(ids):
    url = "https://export.arxiv.org/api/query?max_results=100&id_list=" + ",".join(ids)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                root = ET.fromstring(r.read())
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(5)
    out = {}
    for e in root.findall("a:entry", NS):
        aid = e.find("a:id", NS).text.rsplit("/", 1)[-1].split("v")[0]
        out[aid] = (
            e.find("a:published", NS).text[:10],
            " ".join(e.find("a:title", NS).text.split()),
        )
    return out


def main():
    items = json.loads(DATA.read_text())["mechanisms"]
    arxiv = [m for m in items if m["source"]["kind"] == "arxiv"]
    got = {}
    ids = [m["source"]["id"] for m in arxiv]
    for i in range(0, len(ids), 40):
        got.update(fetch(ids[i : i + 40]))
        time.sleep(3)
    bad = 0
    for m in arxiv:
        aid = m["source"]["id"]
        if aid not in got:
            print(f"MISSING  {m['id']:<22} {aid}")
            bad += 1
            continue
        date, title = got[aid]
        ok = date == m["date"]
        bad += not ok
        print(f"{'OK ' if ok else 'BAD'}  {m['id']:<22} {aid:<11} json={m['date']} arxiv={date}  {title[:70]}")
    for m in items:
        if m["source"]["kind"] != "arxiv":
            print(f"MANUAL {m['id']:<22} {m['date']}  {m['source']['url']}")
    print(f"\n{len(arxiv) - bad}/{len(arxiv)} arXiv dates match v1 submission")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
