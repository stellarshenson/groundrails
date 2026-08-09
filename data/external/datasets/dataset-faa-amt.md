# FAA Aviation Maintenance Technician handbooks

Born-digital procedural text that widens the maintenance vocabulary beyond the Army
TMs' ground vehicles - inspection procedures, warnings and cautions, fastener and connector
vocabulary, torque and tolerance numerics - with the clean text extraction the older Army scans
cannot offer.

- **Source** - `https://www.faa.gov/regulations_policies/handbooks_manuals/`
  `aviation/`, direct PDF URLs
- **Licence** - **public domain**; works of the US Government carry no copyright (17 U.S.C. 105)
- **Size** - 3 handbooks, roughly 1,500 pages combined: AMT General (FAA-H-8083-30B), AMT Airframe
  (FAA-H-8083-31B), AMT Powerplant (FAA-H-8083-32B)
- **Languages** - English
- **How negatives were made** - none ship with the corpus; unlabeled clean prose, negatives
  manufactured by the admitted DR corruption engine at lane build
- **How labels were made** - unlabeled
- **Mapping onto our task** - handbook text chunked to the project's 1,500-char window → evidence;
  claims manufactured at lane build

## Caveats

Low document diversity by construction - three documents, which is exactly the
failure mode the register-gap audit diagnosed. This is a SUPPLEMENT to the Army TMs and must never
form a lane alone.

`faa.gov` answers HTTP 403 to an unrecognised user agent, on the PDF URLs as well as the HTML
pages, so the downloader sends a browser-shaped agent. The provenance gate against the arena
documents runs at lane build, not at fetch.

## Provenance

Admitted by the author's ruling of 2026-08-09, clause 3, as the born-digital
supplement to the Army TMs, under hypothesis R14-H136. Scouted in
`experiments/grounding-semantic/R14_corpus_scout.md` section 9, wall verdict CLEAN, register fit
B-strong for the AMT handbooks specifically.

Fetched by `scripts/fetch_register_corpora.py faa-amt`. The downloaded data under
`data/external/datasets/faa-amt/` is gitignored; this sidecar is tracked.
