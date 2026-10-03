#!/usr/bin/env python3
"""
Step 2 - Parse every commune article (latest revision) and extract body-text
citations of other communes.

How a body-text citation is identified
--------------------------------------
For each article we keep only the *prose* of the article:

  1. HTML comments and <nowiki> blocks are deleted.
  2. The article is truncated at the first end-matter heading (any level):
     Voir aussi, Notes et références, Notes, Références, Sources, Bibliographie,
     Liens externes, Articles connexes, Annexes, Pour approfondir, Galerie...
  3. <ref>…</ref>, <ref/>, <references/>, <gallery>, <timeline>, <imagemap>,
     <math>, <graph>, <mapframe>, <score> blocks are deleted.
  4. Every template {{…}} is deleted with balanced matching. This removes the
     infobox, navigation boxes / palettes, {{Communes limitrophes}},
     {{Carte communes limitrophes}}, portals, coordinates, citation templates…
  5. Every wiki table {| … |} is deleted (balanced).
  6. File / image links [[Fichier:…]] are deleted *including* their caption
     (which may itself contain nested links).
  7. Links to non-article namespaces (Catégorie, Portail, Wikipédia, interwiki…)
     are ignored.

In the remaining prose a citation of commune B is:
  * every wikilink [[B]] / [[B|label]] / [[redirect-to-B|label]]   -> n_links
  * plus every additional plain-text occurrence (outside links) of B's name,
    counted ONLY if the article links B at least once (so a plain-text match is
    disambiguated by the article's own link), the name is >= 3 characters, is
    not shared by two different linked communes in the same article, is not part
    of the citing commune's own name (e.g. "Cormeilles" in Cormeilles-en-Parisis),
    and is not in a small stoplist of names that are also ordinary capitalised words.
                                                                       -> n_text
Every count is tagged with the level-2 section in which it occurs.
"""

from __future__ import annotations

import gzip
import html
import json
import os
import re
import sys
import time
import unicodedata
import zlib
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
CONTENT_DIR = ROOT / "content_wikitext"
META_CSV = ROOT / "page view" / "tables" / "pages_meta.csv"
CACHE = HERE / "cache"

END_SECTIONS = {
    "voir aussi", "notes et references", "notes", "references", "sources",
    "bibliographie", "liens externes", "lien externe", "articles connexes",
    "annexes", "annexe", "pour approfondir", "galerie", "galerie d'images",
    "notes et sources", "references et notes", "sources et references",
    "liens", "webographie", "source",
}
SKIP_NS = {
    "aide", "annexe", "categorie", "category", "commons", "fichier", "file",
    "help", "image", "media", "modele", "module", "portal", "portail", "project",
    "projet", "special", "template", "utilisateur", "user", "wikipedia", "wp",
    "wikidata", "d", "wiktionnaire", "wikt", "s", "wikisource", "n", "q", "v",
    "b", "voy", "mw", "w", "discussion", "spécial", "media", "species", "c",
}
FILE_NS = {"fichier", "file", "image", "media"}
# Commune names that are also ordinary capitalised words (sentence starts, months,
# common nouns, saints...). They count only through explicit wikilinks.
TEXT_STOPLIST = {
    "Vers", "Mars", "Ville", "Grand", "Bourg", "Port", "Lieu", "Mont", "Sept",
    "Moulins", "Fontaine", "Fontaines", "Plus", "Entre", "Sainte", "Saint",
    "Château", "Églises", "Bois", "Val", "Pont", "Rue", "Cours", "Quai", "Orange",
    "Ours", "Poste", "Comte", "Comtes", "Roi", "Lorraine", "Champagne", "Bretagne",
    "Normandie", "Bourgogne", "France", "Seine", "Loire", "Marne", "Moselle",
    "Somme", "Aube", "Oise", "Aisne", "Ain", "Var", "Gard", "Lot", "Cher", "Indre",
    "Vienne", "Charente", "Dordogne", "Garonne", "Isère", "Durance", "Meuse",
    "Rhône", "Saône", "Eure", "Orne", "Sarthe", "Mayenne", "Vendée", "Gers",
    "Tarn", "Aude", "Hérault", "Ariège", "Allier", "Creuse", "Corrèze", "Cantal",
    "Jura", "Doubs", "Vosges", "Ardennes", "Manche", "Calvados", "Nord", "Est",
    "Ouest", "Sud", "Centre", "Bar", "Mer", "Sauveterre", "Villeneuve", "Neuville",
    "Bellevue", "Beaumont", "Montigny", "Monts", "Lac", "Lacs", "Chapelle",
    "Gare", "Rivière", "Plaine", "Sables", "Pins", "Chênes", "Thou", "Y",
}

SECTION_RE = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$", re.MULTILINE)
COMMENT_RE = re.compile(r"<!--.*?(?:-->|$)", re.DOTALL)
NOWIKI_RE = re.compile(r"<nowiki\b[^>]*>.*?</nowiki\s*>", re.DOTALL | re.IGNORECASE)
REF_SELF_RE = re.compile(r"<ref\b[^>]*?/\s*>", re.IGNORECASE)
PAIRED_TAGS = ("ref", "gallery", "timeline", "imagemap", "math", "graph",
               "mapframe", "score", "syntaxhighlight", "source", "references", "templatedata")
PAIRED_RE = re.compile(
    r"<(%s)\b[^>]*?>.*?</\1\s*>" % "|".join(PAIRED_TAGS), re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"</?[a-zA-Z][^<>]*?>")
TPL_TOKEN_RE = re.compile(r"\{\{|\}\}")
TABLE_TOKEN_RE = re.compile(r"^[ \t:]*\{\||^[ \t]*\|\}", re.MULTILINE)
LINK_TOKEN_RE = re.compile(r"\[\[|\]\]")
LINK_RE = re.compile(r"\[\[([^\[\]|]*)(?:\|([^\[\]]*))?\]\]")
MAGIC_RE = re.compile(r"__[A-ZÉ_]+__")
INTERWIKI_RE = re.compile(r"^[a-z]{2,3}(?:-[a-z]+)?$")


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def norm_heading(text: str) -> str:
    text = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"'{2,}", "", text)
    return re.sub(r"\s+", " ", strip_accents(html.unescape(text))).strip().lower()


def norm_title(raw: str) -> str | None:
    """MediaWiki title normalisation (first letter case-insensitive, _ = space)."""
    t = html.unescape(raw).replace("_", " ").replace(" ", " ")
    t = t.split("#", 1)[0]
    t = re.sub(r"\s+", " ", t).strip().lstrip(":").strip()
    if not t:
        return None
    if ":" in t:
        prefix = strip_accents(t.split(":", 1)[0]).strip().lower()
        if prefix in SKIP_NS or INTERWIKI_RE.match(prefix):
            return None
    return t[0].upper() + t[1:]


def remove_balanced(text: str, token_re: re.Pattern, opener_prefix: str) -> str:
    """Remove balanced regions delimited by tokens (openers start with opener_prefix)."""
    out: list[str] = []
    depth = 0
    last = 0
    start = 0
    for m in token_re.finditer(text):
        is_open = m.group(0).lstrip(" \t:").startswith(opener_prefix)
        if is_open:
            if depth == 0:
                out.append(text[last:m.start()])
                start = m.start()
            depth += 1
        elif depth > 0:
            depth -= 1
            if depth == 0:
                last = m.end()
    if depth == 0:
        out.append(text[last:])
    else:  # unbalanced opener: drop to end of paragraph only
        tail = text[start:]
        cut = tail.find("\n\n")
        out.append(tail[cut:] if cut >= 0 else "")
    return "".join(out)


def remove_file_links(text: str) -> str:
    out: list[str] = []
    depth = 0
    last = 0
    open_pos = 0
    dropping = False
    for m in LINK_TOKEN_RE.finditer(text):
        if m.group(0) == "[[":
            if depth == 0:
                open_pos = m.start()
                head = text[m.end():m.end() + 20].split(":", 1)
                dropping = len(head) == 2 and strip_accents(head[0]).strip().lower() in FILE_NS
            depth += 1
        elif depth > 0:
            depth -= 1
            if depth == 0 and dropping:
                out.append(text[last:open_pos])
                last = m.end()
                dropping = False
    out.append(text[last:])
    return "".join(out)


def truncate_end_matter(text: str) -> str:
    for m in SECTION_RE.finditer(text):
        if norm_heading(m.group(2)) in END_SECTIONS:
            return text[:m.start()]
    return text


def clean_body(text: str) -> str:
    text = COMMENT_RE.sub(" ", text)
    text = NOWIKI_RE.sub(" ", text)
    text = REF_SELF_RE.sub(" ", text)
    text = PAIRED_RE.sub(" ", text)
    text = remove_balanced(text, TPL_TOKEN_RE, "{{")
    text = remove_balanced(text, TABLE_TOKEN_RE, "{|")
    text = remove_file_links(text)
    text = truncate_end_matter(text)
    text = TAG_RE.sub(" ", text)
    text = MAGIC_RE.sub(" ", text)
    return text


def split_sections(body: str) -> list[tuple[str, str]]:
    """[(level-2 heading or 'lead', text)]"""
    parts: list[tuple[str, str]] = []
    current = "lead"
    last = 0
    for m in SECTION_RE.finditer(body):
        if len(m.group(1)) == 2:
            parts.append((current, body[last:m.start()]))
            current = norm_heading(m.group(2)) or "untitled"
            last = m.end()
        else:  # sub-heading text itself is not prose: blank it
            parts.append((current, body[last:m.start()]))
            last = m.end()
    parts.append((current, body[last:]))
    return parts


def read_gz(path: Path) -> str:
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except (EOFError, OSError):
        raw = path.read_bytes()
        data = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(raw)
        return data.decode("utf-8", errors="replace")


def latest_revision(folder: Path) -> tuple[int, Path] | None:
    best = (-1, None)
    with os.scandir(folder) as it:
        for e in it:
            if e.name.endswith(".wikitext.gz"):
                stem = e.name.split(".", 1)[0]
                if stem.isdigit() and int(stem) > best[0]:
                    best = (int(stem), Path(e.path))
    return None if best[1] is None else best


INFOBOX_RE = re.compile(r"\{\{\s*(Infobox [^|\n}]+)", re.IGNORECASE)


def infobox_field(text: str, name: str) -> str | None:
    m = re.search(r"^\s*\|\s*%s\s*=\s*(.*?)\s*$" % re.escape(name), text, re.MULTILINE | re.IGNORECASE)
    return m.group(1) if m else None


def parse_coord(raw: str | None) -> float | None:
    if not raw:
        return None
    raw = COMMENT_RE.sub("", raw).replace("−", "-").strip()
    # DMS forms such as 48/51/24/N
    dms = re.match(r"^(-?\d+(?:[.,]\d+)?)\s*/\s*(\d+(?:[.,]\d+)?)?\s*/?\s*(\d+(?:[.,]\d+)?)?\s*/?\s*([NSEOW])?", raw)
    if dms and "/" in raw:
        d, mi, s, h = dms.groups()
        degrees = float(d.replace(",", "."))
        val = abs(degrees) + float((mi or "0").replace(",", ".")) / 60 + float((s or "0").replace(",", ".")) / 3600
        return -val if degrees < 0 or h in ("S", "O", "W") else val
    m = re.search(r"-?\d+(?:[.,]\d+)?", raw)
    return float(m.group(0).replace(",", ".")) if m else None


# --------------------------------------------------------------------------- worker
RESOLVE: dict[str, str] = {}
BASENAME: dict[str, str] = {}


def init_worker(resolve_path: str) -> None:
    global RESOLVE, BASENAME
    RESOLVE = json.loads(Path(resolve_path).read_text(encoding="utf-8"))
    BASENAME = {t: re.sub(r"\s*\([^)]*\)\s*$", "", t) for t in set(RESOLVE.values())}


def process(args: tuple[str, str]) -> tuple[dict, list[tuple]]:
    slug, title = args
    latest = latest_revision(CONTENT_DIR / slug)
    if latest is None:
        return {"slug": slug, "title": title, "status": "no_revision"}, []
    oldid, path = latest
    raw = read_gz(path)
    head = raw[:30000]
    ib = INFOBOX_RE.search(head)
    info = {
        "slug": slug, "title": title, "oldid": oldid, "status": "ok",
        "infobox": ib.group(1).strip() if ib else "",
        "insee_infobox": (re.sub(r"[^0-9AB]", "", (infobox_field(head, "insee") or "").upper())[:5]) or None,
        "lat_infobox": parse_coord(infobox_field(head, "latitude")),
        "lon_infobox": parse_coord(infobox_field(head, "longitude")),
        "redirect_page": bool(re.match(r"\s*#(REDIRECT|REDIRECTION)", raw, re.IGNORECASE)),
        "raw_chars": len(raw),
    }
    body = clean_body(raw)
    info["body_chars"] = len(body)
    self_title = RESOLVE.get(norm_title(title) or "", title)
    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    plain_by_section: list[tuple[str, str]] = []
    linked_targets: set[str] = set()
    for section, text in split_sections(body):
        def repl(m: re.Match) -> str:
            tgt = norm_title(m.group(1))
            if tgt is not None:
                canon = RESOLVE.get(tgt)
                if canon is not None and canon != self_title:
                    counts[(canon, section)][0] += 1
                    linked_targets.add(canon)
            return " \x00 "
        plain = LINK_RE.sub(repl, text)
        plain_by_section.append((section, plain))
    # plain-text repeats of linked communes
    by_base: dict[str, list[str]] = defaultdict(list)
    for t in linked_targets:
        by_base[BASENAME[t]].append(t)
    self_base = BASENAME.get(self_title, title)
    for base, targets in by_base.items():
        if len(targets) != 1 or len(base) < 3 or base in TEXT_STOPLIST or base == self_base:
            continue
        # the article's own short name ("Cormeilles" in Cormeilles-en-Parisis, "Montereau" in
        # Montereau-Fault-Yonne) must not be credited to a homonymous linked commune
        if re.search(r"(?:^|[-\s'’])%s(?:$|[-\s'’])" % re.escape(base), self_base):
            continue
        pat = re.compile(r"(?<![\w-])%s(?![\w-])" % re.escape(base))
        for section, plain in plain_by_section:
            n = len(pat.findall(plain))
            if n:
                counts[(targets[0], section)][1] += n
    info["n_link_mentions"] = sum(v[0] for v in counts.values())
    rows = [(slug, tgt, sec, v[0], v[1]) for (tgt, sec), v in counts.items()]
    return info, rows


# --------------------------------------------------------------------------- driver
def build_resolver() -> Path:
    """normalised title or redirect -> canonical pages_meta title."""
    meta = pd.read_csv(META_CSV, encoding="utf-8")
    titles = [re.sub(r"\s+", " ", t.replace("_", " ")).strip() for t in meta["fr_title_canon"].astype(str)]
    resolve: dict[str, str] = {}
    for t in titles:
        resolve[norm_title(t)] = t
    red = pd.read_csv(CACHE / "wiki_redirects.csv", encoding="utf-8")
    meta_set = set(titles)
    # meta titles that are now redirects to a renamed article: renamed -> meta title
    renamed = {}
    for r, t in zip(red["redirect"], red["target"]):
        if r in meta_set and t not in meta_set:
            renamed[t] = r
            resolve.setdefault(norm_title(t), r)
    for r, t in zip(red["redirect"], red["target"]):
        canon = t if t in meta_set else renamed.get(t)
        if canon is None:
            continue
        key = norm_title(r)
        if key and key not in resolve:  # an article title always wins over a redirect
            resolve[key] = canon
    path = CACHE / "title_resolver.json"
    path.write_text(json.dumps(resolve, ensure_ascii=False), encoding="utf-8")
    print(f"resolver: {len(titles)} article titles + {len(resolve) - len(titles)} redirects")
    return path


def safe_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_\-]+", "_", text.strip())
    return slug[:200] if slug else "page"


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    resolver_path = build_resolver()
    meta = pd.read_csv(META_CSV, encoding="utf-8")
    titles = [re.sub(r"\s+", " ", t.replace("_", " ")).strip() for t in meta["fr_title_canon"].astype(str)]
    by_slug: dict[str, list[str]] = defaultdict(list)
    for t in titles:
        by_slug[safe_slug(t)].append(t)
    jobs = []
    for slug in sorted(os.listdir(CONTENT_DIR)):
        cands = by_slug.get(slug, [])
        # one known collision (mojibake title in pages_meta): keep the first candidate
        jobs.append((slug, cands[0] if cands else slug.replace("_", " ")))
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    jobs = jobs[:limit] if limit else jobs
    t0 = time.time()
    infos, rows = [], []
    with Pool(max(1, cpu_count() - 1), initializer=init_worker, initargs=(str(resolver_path),)) as pool:
        for i, (info, r) in enumerate(pool.imap_unordered(process, jobs, chunksize=40), 1):
            infos.append(info)
            rows.extend(r)
            if i % 2500 == 0 or i == len(jobs):
                print(f"  parsed {i}/{len(jobs)} pages, {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    suffix = f"_sample{limit}" if limit else ""
    pd.DataFrame(infos).to_parquet(CACHE / f"pages{suffix}.parquet", index=False)
    pd.DataFrame(rows, columns=["source_slug", "target_title", "section", "n_links", "n_text"]).to_parquet(
        CACHE / f"mentions{suffix}.parquet", index=False)
    print("done", len(infos), "pages", len(rows), "mention rows")


if __name__ == "__main__":
    main()
