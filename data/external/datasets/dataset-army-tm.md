# US Army technical manuals (operator and maintenance)

The procedural / product-manual register is the most extreme measured training gap -
181 rows from 30 distinct documents, 0.026% of the mix, against 43.4% of delucionqa arena rows, a
1,645x gap that no re-weighting can close because the rows are absent. This corpus is the only
candidate that fixes document diversity rather than row count, and it is public domain.

- **Source** - `https://www.liberatedmanuals.com/`, index `all.mpl`, one PDF per
  manual identifier
- **Licence** - **public domain**; works of US federal employees carry no copyright
  (17 U.S.C. 105), and the mirror states the manuals "are NOT subject to copyright, can be freely
  copied and redistributed"
- **Size** - 4,792 distinct PDFs in the index, filtered to the **1,766 operator / unit /
  direct-support maintenance manuals**: identifier tails `-10`, `-12`, `-13`, `-14`, `-20`, `-23`,
  `-24`, `-34` with no `P` anywhere in the identifier. The 2,203 parts-list PDFs are excluded as
  catalogue tables, the wrong register
- **Languages** - English
- **Register fit** - measured `procden` 23.45 on `TM-10-3930-630-12` against the arena-median bar
  11.33: warnings, cautions, numbered steps, torque values, "refer to paragraph" cross-references
- **How negatives were made** - none ship with the corpus; unlabeled clean prose, negatives
  manufactured by the admitted DR corruption engine at lane build
- **How labels were made** - unlabeled
- **Mapping onto our task** - manual text chunked to the project's 1,500-char window → evidence;
  claims manufactured at lane build

## Caveats

**Acquisition is rate-limited, not instant.** The mirror allows roughly 100
manuals per IP per day, so the full 1,766-file pull takes about 18 days of polite crawling. The
downloader is detached, budgeted per day and resumable: it sleeps to the next daily window when
the budget or the server's limit is reached, and re-running continues from `army-tm/_state.json`.
The mirror also refuses individual requests transiently - observed once at file 10, with the same
URL serving normally seconds later - so a refusal backs off for two minutes and only a run of five
consecutive refusals is read as the daily allowance being spent.

**The archive.org mirror route does not cover this set.** Probed before acquisition: the
`military-manuals` collection holds 678 items against the 1,766 targets, direct identifier
resolution hit 0 of 12 sampled targets, and full-text search hit 2 of 10 with one of those two a
wrong-document match. Archive.org cannot supply the measured set, so the rate-limited primary
mirror is the route taken.

Older scanned TMs extract with OCR noise ("Highgway", "Diagramn" observed in a 1970s document), so
a text-quality filter is required at lane build. The provenance gate against the arena documents
runs at lane build, not at fetch.

## Provenance

Admitted by the author's ruling of 2026-08-09, clause 3 ("Army TMs"),
acquisition immediate, under hypothesis R14-H136. Scouted and measured in
`experiments/grounding-semantic/R14_corpus_scout.md` section 8, wall verdict CLEAN - no RAGBench
subset draws on US military documentation (delucionqa is Jeep, emanual is Samsung, techqa is
IBM).

Fetched by `scripts/fetch_register_corpora.py army-tm`. The downloaded data under
`data/external/datasets/army-tm/` is gitignored; this sidecar is tracked.
