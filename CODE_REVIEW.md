# Code review — 3 October 2026

All Python sources were reviewed and checked with Ruff's fatal-error and
undefined-/unused-name rules. Both R scripts parsed successfully. The citation
pipeline received the strongest validation; external collection and historical
statistics have the limitations listed below.

## Corrections

- **Extraction:** excluded reference/template/image blocks are removed before
  searching for end-matter headings. Nowiki attributes are handled, negative
  DMS coordinates preserve their sign, and apostrophes are recognized in source
  commune names (`Val-d'Izé` no longer supplies repeats to `Izé`).
- **Caches:** SHA-256 fingerprints bind ordered city IDs, topology, all weights,
  seeds and resolutions. Analysis, preview and extended reporting reject stale
  or unverified Leiden caches.
- **Maps:** small/empty graphs and zero-strength sizing are handled. Captions
  disclose largest-component coverage, display filtering and grey communities.
  The finite palette does not guarantee unique colours to neighbours. Bounds
  are set before label adjustment to avoid clipping edge labels. Hub labels are
  constrained to within 35 km of their actual marker and use leader lines,
  correcting the misleading drift identified on the administration map.
- **Topics:** the extended report and new maps share an ordered classifier;
  every heading and its topic assignment is exported.
- **Reference data:** the three COG seat tables used by the extended report are
  now fetched. Required download failures propagate. Former communes are folded
  only through delegated/associated parent codes, as the implementation supports.
- **Paragraph audit:** sources/targets use the same metropolitan eligibility and
  post-merger self-link exclusions as analysis. Headings do not count as prose.
  Formulaic paragraph patterns are not proof of bot authorship.
- **Collector:** paths resolve to this project regardless of calling directory;
  incomplete/future months are excluded and disabled device splits are missing
  values. Existing raw CSVs were not rewritten.
- **Repository:** dependencies, regression tests, CI and input provenance are
  supplied; large local datasets remain outside version control.

## Validation

```powershell
ruff check .
python -m unittest discover -s tests -v
python reseau_citations/extract_links.py
python reseau_citations/section_maps.py
python scripts/validate_report.py
```

Eighteen regression tests cover excluded markup, subsection inheritance, title
normalization, latest revisions, DMS coordinates, apostrophe names, link
anchoring, conservation, source weights, small/empty graphs, topics, distance
medians, invalidated caches, collector paths, missing API observations and bounded label positions.

The complete 35,118-article corpus was freshly parsed and all five maps built.
Every exported layer was checked for unique IDs/positive directed edges, no
self-links, citation conservation, incoming/outgoing totals, unit normalized
source sums, component coverage, finite distances, image dimensions, valid
report links, matching JSON/CSV counts and input hashes. The PDF has eight
pages: findings, five maps and two method pages. Text extents and map layouts
were checked for clipping.

## Scope of older tools

The earlier map code was reviewed before its folder was removed during the
review. It used greedy modularity on a filtered display graph. The citation
project is now self-contained: its department outline snapshot lives in its
own cache, with source attribution and a download helper.

The live collector was not launched: `listexport.csv` is absent and a full run
would append substantial data through external requests. Device columns already
filled with zeros cannot distinguish disabled collection from genuine zero;
older missing API observations also need care. The 15 GB monthly panel was not
reprocessed. R syntax validation is separate from full pageview analysis.

The original manually assembled report remains in `report/deep-dive.html` with
its historical numbers; it is not rebuilt automatically from new tables. Use
the new generated report for this comparison. Extended spatial spin tests are
approximate diagnostics subject to coast, border and nearest-neighbour
distortions; their p-values are not calibrated evidence of administrative
effects. The new report makes no significance claims.

## Extended pipeline run

The full four-weighting, eight-resolution, five-seed analysis completed in
1,030 seconds. Its selection rule chose gamma 1.25 and 36 communities. The
five-map comparison deliberately keeps gamma 0.5 across topics (17 communities
in the all-text view), matching the earlier map scale. These are distinct
analyses, with their settings documented separately. The reference loader,
paragraph audit, verified-cache preview and extended descriptive generator
were also run. Extended result CSVs and the full analysis log are retained.

## Territories alignment

The report overall map and the territory view now share the same exact
partition and colours (gamma 0.5, 17 groups). Previously the territory PNG
used the independent extended selection (gamma 1.25, 36 groups). That result
is retained under an explicit `_extended` filename and labelled with its
settings. The report validator checks the territory membership fingerprint
against the overall node table. Hull construction changes the visible envelope
only, not a city?s community assignment.
