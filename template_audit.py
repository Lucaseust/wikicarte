#!/usr/bin/env python3
"""
Step 5 - How much of the citation network comes from formulaic (largely bot-written) paragraphs?

Each paragraph of cleaned body text (same cleaning as extract_links.py) is assigned to the first
matching paragraph type below; every resolved wikilink to a commune is counted under that type.
Links only (no plain-text repeats), so counts are comparable across types.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from multiprocessing import Pool, cpu_count


import extract_links as E
import analysis as A

TITLE_CODES = {}
SLUG_CODES = {}


def init_audit(resolver, title_codes, slug_codes):
    global TITLE_CODES, SLUG_CODES
    E.init_worker(resolver)
    TITLE_CODES, SLUG_CODES = title_codes, slug_codes

TYPES = [
    ("Climate (nearest weather station)", re.compile(r"station m[ée]t[ée]orologique|M[ée]t[ée]o-France", re.I)),
    ("Distance to reference towns", re.compile(r"vol d'oiseau|orthodromie", re.I)),
    ("INSEE urban typology", re.compile(r"aire d'attraction|unit[ée] urbaine|bassin de vie|grille communale de densit", re.I)),
    ("Neighbouring communes", re.compile(r"limitrophe", re.I)),
    ("Rivers & hydrography", re.compile(r"prend sa source|se jette dans|cours d'eau|confluen|affluent", re.I)),
]
FREE = "Free prose"


def work(args):
    slug, title = args
    latest = E.latest_revision(E.CONTENT_DIR / slug)
    if latest is None:
        return slug, Counter(), Counter()
    body = E.clean_body(E.read_gz(latest[1]))
    self_title = E.RESOLVE.get(E.norm_title(title) or "", title)
    by_type, by_type_target = Counter(), Counter()
    prose = "\n\n".join(text for _, text in E.split_sections(body))
    for para in re.split(r"\n\s*\n|\n(?=[*#:;])", prose):
        kind = next((name for name, rx in TYPES if rx.search(para)), FREE)
        for m in E.LINK_RE.finditer(para):
            t = E.norm_title(m.group(1))
            canon = E.RESOLVE.get(t) if t else None
            if canon and canon != self_title and canon in TITLE_CODES and TITLE_CODES[canon] != SLUG_CODES[slug]:
                by_type[kind] += 1
                by_type_target[(kind, canon)] += 1
    return slug, by_type, by_type_target


def main():
    data = A.load_data()
    nodes, slug_codes, title_codes = A.clean_city_identifiers(data)
    meta = data["pages"].loc[data["pages"].slug.isin(slug_codes), ["slug", "title"]]
    jobs = list(meta.itertuples(index=False, name=None))
    tot, tgt = Counter(), Counter()
    per_page = {}
    with Pool(min(8, max(1, cpu_count() - 1)), initializer=init_audit,
              initargs=(str(E.CACHE / "title_resolver.json"), title_codes, slug_codes)) as pool:
        for slug, a, b in pool.imap_unordered(work, jobs, chunksize=40):
            tot.update(a)
            tgt.update(b)
            per_page[slug] = sum(v for k, v in a.items() if k != FREE)
    # The same canonical metropolitan IDs and self-link exclusions as analysis.py.
    tot = Counter()
    for (kind, _), v in tgt.items():
        tot[kind] += v
    n = sum(tot.values())
    out = {"total_links": n, "by_type": {k: v / n for k, v in tot.most_common()},
           "by_type_count": dict(tot),
           "pages_with_any_template_link": sum(1 for v in per_page.values() if v > 0) / len(per_page)}
    top = {}
    for kind, _ in TYPES:
        items = sorted(((c, k[1]) for k, c in tgt.items() if k[0] == kind), reverse=True)[:8]
        top[kind] = [(name, c) for c, name in items]
    out["top_targets"] = top
    (E.HERE / "report").mkdir(exist_ok=True)
    (E.HERE / "report" / "template_audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
