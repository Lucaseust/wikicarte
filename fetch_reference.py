#!/usr/bin/env python3
"""
Step 1 - Download and cache the reference data used to identify cities.

Sources (all public, cached under ./cache):
  * geo.api.gouv.fr /communes  : current communes -> INSEE code, name, population,
                                  département code, région code, centroid.
  * geo.api.gouv.fr /departements, /regions : official names.
  * INSEE COG 2025 v_commune    : delegated / associated communes -> parent commune
                                  (COMPARENT), used to fold pages of former communes
                                  into the current commune that absorbed them.
  * INSEE COG 2025 region, department and arrondissement tables: seat status.
  * fr.wikipedia.org API        : every redirect pointing to a commune article, so
                                  that links such as [[Mautor]] -> Abbeville or
                                  [[Dieppe]] -> "Dieppe (Seine-Maritime)" resolve.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(__file__).resolve().parent / "cache"
CACHE.mkdir(exist_ok=True)
META_CSV = ROOT / "page view" / "tables" / "pages_meta.csv"

UA = "wiki-commune-citation-research/1.0 (contact: lucaseustache2@gmail.com)"
COG_URL = "https://www.insee.fr/fr/statistiques/fichier/8377162/v_commune_2025.csv"
MVT_URL = "https://www.insee.fr/fr/statistiques/fichier/8377162/v_mvt_commune_2025.csv"
BOUNDARY_URL = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements.geojson"


def fetch_boundaries() -> None:
    """Cache metropolitan department outlines inside the citation project."""
    path = CACHE / "departements_france_metropolitaine.geojson"
    if path.exists():
        return
    data = json.loads(http_get(BOUNDARY_URL))
    data["features"] = [f for f in data["features"] if len(str(f["properties"]["code"])) == 2]
    if len(data["features"]) != 96:
        raise ValueError("Expected 96 metropolitan departments in the boundary snapshot")
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8", newline="\n")


def http_get(url: str, retries: int = 5) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1} after error: {exc}")
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def fetch_geo_api() -> None:
    out = CACHE / "geo_communes.json"
    if not out.exists():
        url = ("https://geo.api.gouv.fr/communes?fields=nom,code,population,"
               "codeDepartement,codeRegion,centre&format=json")
        out.write_bytes(http_get(url))
    for name in ("departements", "regions"):
        path = CACHE / f"geo_{name}.json"
        if not path.exists():
            path.write_bytes(http_get(f"https://geo.api.gouv.fr/{name}"))
    print(f"geo.api communes: {len(json.loads(out.read_text(encoding='utf-8')))}")


def fetch_cog() -> None:
    files = [(COG_URL, "cog_communes_2025.csv")]
    for level in ("region", "departement", "arrondissement"):
        files.append((COG_URL.replace("v_commune_2025", f"v_{level}_2025"), f"v_{level}_2025.csv"))
    for url, name in files:
        path = CACHE / name
        if path.exists():
            continue
        path.write_bytes(http_get(url))
    print("COG files:", [p.name for p in CACHE.glob("cog_*.csv")])


def fetch_redirects() -> None:
    """All redirects (ns 0) pointing to each commune article listed in pages_meta."""
    out = CACHE / "wiki_redirects.csv"
    if out.exists():
        print("redirects cached:", sum(1 for _ in out.open(encoding="utf-8")) - 1)
        return
    titles = pd.read_csv(META_CSV, encoding="utf-8")["fr_title_canon"].astype(str).tolist()
    titles = [re.sub(r"\s+", " ", t.replace("_", " ")).strip() for t in titles]
    rows: list[tuple[str, str]] = []
    api = "https://fr.wikipedia.org/w/api.php"
    for start in range(0, len(titles), 50):
        batch = titles[start:start + 50]
        params = {
            "action": "query", "format": "json", "formatversion": "2",
            "titles": "|".join(batch), "prop": "redirects",
            "rdnamespace": "0", "rdlimit": "max", "redirects": "1",
        }
        while True:
            data = json.loads(http_get(api + "?" + urllib.parse.urlencode(params)))
            q = data.get("query", {})
            # A meta title may itself have become a redirect (page renamed): record it.
            for r in q.get("redirects", []):
                rows.append((r["from"], r["to"]))
            for page in q.get("pages", []):
                for r in page.get("redirects", []) or []:
                    rows.append((r["title"], page["title"]))
            if "continue" not in data:
                break
            params.update(data["continue"])
        if (start // 50) % 50 == 0:
            print(f"  redirects: {start + len(batch)}/{len(titles)} titles, {len(rows)} redirects")
        time.sleep(0.05)
    pd.DataFrame(rows, columns=["redirect", "target"]).drop_duplicates().to_csv(
        out, index=False, encoding="utf-8")
    print("redirects fetched:", len(rows))


if __name__ == "__main__":
    fetch_boundaries()
    fetch_geo_api()
    fetch_cog()
    fetch_redirects()
