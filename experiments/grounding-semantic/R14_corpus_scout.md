# R14 corpus scout - public corpora for the two measured training-register gaps

Paper-trail verification only. No corpus was downloaded into `data/`, nothing was added to any
training mix, and no provenance gate was run. Every admission below still requires
`experiments/grounding-semantic/provenance_gate.py` (13-gram bidirectional containment, WARN
0.5%, KILL 2%) before it may enter a lane.

**Answer in one line**: GAP B (procedural-manual) has a **clean, public-domain, high-volume
winner** - US Army technical manuals, 4,792 distinct PDFs of which 1,766 are operator/maintenance
procedure documents, measured at `procden` up to 23.5 against the arena's 11.33 bar; GAP A
(financial discourse) has **no clean high-volume winner** - the only corpus with the right
discourse at scale is EDGAR-CORPUS, and it sits on the *same document population* that FinQA was
built from, so it is SUSPECT-HIGH and admissible only under a company/year exclusion filter.

## What "the gaps" are

From `R14_evidence_E6_train_composition.md`:

- **GAP A, finqa-like** - long (>= 800 char) financial *prose* with embedded numerics and finance
  vocabulary. Training mix: 904 rows carry `finlex >= 2` (0.13%) from 358 documents, against 50.3%
  of finqa arena rows - a 382x gap. The full signature matches 31 rows
- **GAP B, delucionqa-like** - product-manual / troubleshooting register, `procden >= 11.33` hits
  per 1,000 chars at >= 800 chars. Training mix: 181 rows (0.026%) from **30 documents**, 27 of
  them negative, against 43.4% of delucionqa rows - a 1,645x gap

## The contamination wall, with the arena's actual source documents named

The wall forbids RAGBench's ten source corpora and their derivatives. Scouting needs the *document
population* behind each, not just the corpus name, because a new corpus can hit the 13-gram gate
without ever citing the benchmark:

| arena subset | actual source documents | what this bans by substrate |
|---|---|---|
| finqa | S&P 500 earnings reports 1999-2019, via FinTabNet (12,719 pages filtered → 2,789 annotated) | S&P 500 10-K/annual-report text of those years, however re-crawled |
| tat-qa | corporate annual reports | same class |
| delucionqa | **Jeep 2023 Gladiator owner's manual** (not Wrangler - the paper says Gladiator) | Stellantis/FCA owner-manual text; Wrangler shares platform boilerplate |
| emanual | **Samsung Smart TV / remote manuals** (659 questions); the paper's pretraining corpus additionally crawled ~300k web product manuals | Samsung consumer-electronics manuals; broad web-crawled manual dumps are SUSPECT |
| techqa | IBM technotes | IBM support documentation |
| expertqa | expert questions with **web-retrieved evidence documents** | any broadly web-scraped corpus can collide; this is why StackExchange is SUSPECT below |

Sources: FinQA paper (arXiv 2109.00122), DelucionQA (Findings EMNLP 2023), EManual (Findings EMNLP
2021), RAGBench (arXiv 2407.11005).

---

## GAP A candidates - financial discourse

### 1. EDGAR-CORPUS (`eloukas/edgar-corpus`)

Annual-report 10-K filings from SEC EDGAR, split into the 15 standard items.

- **License** - `apache-2.0` (the HF tag; underlying filings are US government-published public
  records)
- **Size** - 220,375 filings, 40.7 GB, years 1993-2020; per-year configs `year_1993` … `year_2020`
  plus a `full` config (176,289 / 22,050 / 22,036 train/val/test)
- **Format** - one row per filing: `filename`, `cik`, `year`, and `section_1` … `section_15`
  (including 1A, 1B, 7A, 9A, 9B). **Section 7 is MD&A** - exactly the finqa register: management
  narrative with embedded numerics, currency and finance vocabulary, not tables
- **Lineage** - independently crawled from EDGAR with the authors' `EDGAR-CRAWLER` tool. It is
  **not** a FinQA derivative and does not descend from FinTabNet
- **Wall verdict** - **SUSPECT (high)**. The doubt is not lineage, it is *population identity*:
  FinQA's pages come from S&P 500 annual reports 1999-2019 via FinTabNet, and EDGAR-CORPUS
  1999-2020 contains those same companies' 10-K filings. The arena's finqa windows are prose
  (`pre_text`/`post_text`) lifted off those pages, so verbatim 13-gram collision is likely at
  material rate. This matches the standing local ruling in `R10-H108_gate_report.md`
  ("no SEC/EDGAR material") and `R14_hypotheses_L3_data_mixture.md` ("raw-EDGAR packagings all sit
  at material 13-gram containment risk")
- **Mitigation that could turn it CLEAN** - filter before gating: drop every filing whose CIK maps
  to an S&P 500 constituent in 1999-2019, or invert it and keep only small/mid-cap filers and only
  years 2020 (and, if the author permits a later pull via EDGAR-CRAWLER, 2021+). FinTabNet never
  saw those filers. Then run the gate on the survivors
- **Register fit** - **A, strong**. The single best discourse match found anywhere in this scout
- **Usable document count** - 220k filings raw; a section-7-only, non-S&P-500 slice plausibly
  leaves 50k-100k documents, each chunking to many 1,500-char windows
- **Arrives as** - clean prose, unlabeled. Fine: the corruption engine manufactures negatives
- **Retrieval effort** - low (HF `load_dataset`, per-year configs keep memory sane), plus moderate
  effort to build the CIK exclusion list

### 2. GovReport (`launch/gov_report`)

Congressional Research Service and GAO reports with expert-written summaries.

- **License** - `CC BY 4.0` (underlying reports are US federal works, public domain)
- **Size** - 19,466 reports; median ~9.4k words per document; splits 17,519 / 974 / 973. A
  `structured` config exposes hierarchical sections with titles and paragraph depth
- **Lineage** - CRS and GAO publications. No contact with FinQA, TAT-QA or any RAGBench source
- **Wall verdict** - **CLEAN**
- **Register fit** - **A, moderate**. GAO financial-audit and budget reports carry dense
  numeric-plus-currency narrative ("the agency reported obligations of ... a decrease of ...
  percent"), which is the finqa *discourse shape*; what it does not carry is corporate accounting
  vocabulary (EBITDA, goodwill, shares outstanding). Expect it to fire `digden` and `curr` reliably
  and `finlex` only partially
- **Usable document count** - ~19k documents, long → high chunk yield
- **Arrives as** - clean prose with summaries (the summaries are unused; take the report bodies)
- **Retrieval effort** - low

### 3. Common Pile USGPO (`common-pile/usgpo`, or `common-pile/usgpo_filtered`)

All plain-text documents from the GovInfo.gov developer API.

- **License** - **Public Domain** (US federal works)
- **Size** - 2,732,677 documents, 74.5 GB UTF-8, dates 1993-2024, 21 `collection` values
- **Format** - `collection`, `id`, `title`, `date`, `author`, `text`, `metadata` (license + source
  URL). The `collection` field is the lever: **Budget of the US Government**, **Economic
  Indicators**, **Economic Report of the President**, congressional hearings on financial topics,
  Federal Register
- **Lineage** - government publishing only. No RAGBench contact
- **Wall verdict** - **CLEAN**
- **Register fit** - **A, moderate**. Budget and economic-indicator documents are the densest
  numeric-narrative public-domain text available; the vocabulary is fiscal rather than corporate
- **Usable document count** - millions raw; a budget/economic-collections filter plausibly yields
  tens of thousands of finance-dense documents
- **Arrives as** - clean prose, unlabeled
- **Retrieval effort** - low to moderate (74.5 GB; stream and filter by `collection` before
  materialising)

### 4. ECTSum (`github.com/rajdeep345/ECTSum`)

Earnings-call transcripts with telegram-style bullet summaries.

- **License** - repository carries **GPL-3.0**, which covers the code. The *documents* are earnings
  call transcripts scraped from **The Motley Fool** and the summaries derive from **Reuters**
  articles - neither publisher granted a redistribution licence, so the data's legal footing is
  the authors' fair-use research posture, not an open licence
- **Size** - 2,425 document-summary pairs (EMNLP 2022)
- **Lineage** - Motley Fool transcripts. Not a FinQA/TAT-QA derivative
- **Wall verdict** - **CLEAN** on the contamination wall; **blocked on licensing** for a
  commercially shipped model. Same conflict class as the SciFact NC flag already awaiting the
  author, and arguably worse (no licence at all, versus a non-commercial one)
- **Register fit** - **A, moderate**. Spoken financial discourse, numeric and finance-worded, but
  conversational rather than report prose
- **Usable document count** - 2,425 - too small to move a 685k-row mix even if licensing cleared
- **Arrives as** - document + summary pairs
- **Retrieval effort** - low (git clone)

### 5. FiQA (`BeIR/fiqa`)

- **License** - `cc-by-sa-4.0`
- **Size** - 57,638 corpus documents, 6,648 queries
- **Lineage** - financial forum and StackExchange-style investor discussion posts. Not a RAGBench
  source
- **Wall verdict** - **CLEAN** (small residual ExpertQA-style web-overlap doubt, resolved by the
  gate)
- **Register fit** - **A, weak**. The corpus is short retail-investor opinion prose: low numeric
  density, no report structure, few of them clear the 800-char floor. It matches finqa's *domain*
  and misses its *discourse* - which is precisely the axis E6 says is the deficit
- **Usable document count** - 57k, but most fail the length and digit-density masks
- **Retrieval effort** - low
- **Recommendation** - not worth a lane on its own

### 6. MultiHiertt

- **License** - MIT on the repo
- **Lineage** - **built on FinTabNet**, the same S&P 500 annual-report table corpus that FinQA was
  built from (confirmed in the MultiHiertt paper, arXiv 2206.01347)
- **Wall verdict** - **FORBIDDEN**. Lineage touches a FinQA source corpus, per the mission rule and
  per the standing local ruling that already rejected it at R10 review
- No further fields recorded - it cannot be admitted

### 7. ConvFinQA

- **Lineage** - conversational extension built directly on FinQA
- **Wall verdict** - **FORBIDDEN** (known FinQA derivative; recorded here for the register)

Also FORBIDDEN by the same substrate argument and already on the local refused list: **FinanceBench**,
**FinTabNet**, **TAT-HQA**.

---

## GAP B candidates - procedural / product-manual prose

### 8. US Army technical manuals (liberatedmanuals.com) - RECOMMENDED

Direct-download mirror of Department of the Army TMs, LOs, TBs and MWOs.

- **License** - **public domain**. Site states "All the manuals shown here are a product of the US
  Federal Government. They are NOT subject to copyright, can be freely copied and redistributed."
  Confirmed independently: works of US federal employees carry no copyright
- **Size, measured not estimated** - the index at `https://www.liberatedmanuals.com/all.mpl` lists
  **4,792 distinct PDFs**: 9,178 TM links / 190 TB / 184 LO / 26 MWO / 6 SB before dedup. Page
  counts run from 3 to 725 in the sample drawn
- **Composition, and the filter that matters** - 2,203 of the 4,792 are **parts lists** (names
  ending in `P`, e.g. `TM-9-2320-289-20P`). Those are catalogue tables, not procedures - measured
  `digden` 27.9-36.8 with `procden` 3.6-4.3, wrong register. **1,766 files** are
  operator/unit/direct-support maintenance manuals (`-10`, `-12`, `-13`, `-14`, `-20`, `-23`,
  `-34` tails without `P`); 1,600 of those are TMs. Those are the target
- **Register fit - measured on live samples** (mid-document, 25 pages, same keyword set and formula
  as E6):

  | manual | pages | procden | digden | read |
  |---|---:|---:|---:|---|
  | `TM-10-3930-630-12` (forklift, operator+maintenance) | 135 | **23.45** | 4.39 | clears the 11.33 arena bar by 2x |
  | `TM-10-9925-100-12-and-P` | 257 | 8.01 | 3.61 | near the bar |
  | `TM-11-5965-269-50` | 3 | 4.60 | 4.40 | short admin document |
  | `TM-11-5815-206-34P-1` (parts list) | 554 | 4.29 | 36.84 | parts catalogue - exclude |
  | `TM-9-2320-289-20P` (parts list) | 725 | 3.59 | 27.89 | parts catalogue - exclude |

  **B, strong** for the filtered slice. Warnings, cautions, numbered steps, torque values, connector
  and cable vocabulary, "refer to paragraph" cross-references - the delucionqa surface, in
  public-domain text
- **Lineage / wall verdict** - **CLEAN**. No RAGBench subset draws on US military documentation;
  delucionqa is Jeep, emanual is Samsung, techqa is IBM
- **Usable document count** - ~1,766 procedure manuals averaging low-hundreds of pages → order
  100k+ pages → at the project's 1,500-char chunk cap, on the order of 10^5-10^6 chunks. This alone
  is enough to take the register from 30 documents to four figures
- **Arrives as** - clean prose, unlabeled. Fine
- **Retrieval effort** - **moderate**. Reachability verified live: `HTTP/1.1 200`,
  `Content-Type: application/pdf`. Two frictions: the site imposes a **100 manuals per IP per day**
  limit, so a filtered 1,766-file pull needs ~18 days of polite crawling or a mirror; and older
  scanned TMs extract with OCR noise ("Highgway", "Diagramn" seen in a 1970s document), so a text-
  quality filter is required. Mirrors that remove the rate limit: `everyspec.com/ARMY/TM-Tech-Manual/`,
  Internet Archive `archive.org/details/military-manuals`

### 9. FAA handbooks and manuals (faa.gov)

- **License** - **public domain** (US federal works)
- **Size** - roughly 25 aviation handbooks plus aircraft handbooks and thousands of Advisory
  Circulars. Named high-value items: Aviation Maintenance Technician Handbook **General**
  (FAA-H-8083-30B), **Airframe** vols 1-2, **Powerplant** (FAA-H-8083-32B); Airplane Flying
  Handbook (FAA-H-8083-3C); Pilot's Handbook of Aeronautical Knowledge (FAA-H-8083-25)
- **Format** - chaptered PDFs, typically 300-600 pages, born-digital (clean text extraction, unlike
  the older Army scans)
- **Lineage / wall verdict** - **CLEAN**
- **Register fit** - **B, strong** for the AMT handbooks specifically: inspection procedures,
  warnings and cautions, component/connector/fastener vocabulary, torque and tolerance numerics.
  The flight handbooks are more expository and fit only moderately
- **Usable document count** - **low** as documents (tens), high as pages (~10k). Good register,
  small document diversity - which is exactly the failure mode E6 diagnosed. Use it as a
  *supplement* to the Army TMs, never alone
- **Retrieval effort** - **low, with one caveat**: `faa.gov` returns **HTTP 403 to automated HTML
  fetches**, but direct PDF URLs serve fine (verified: `01_afh_front.pdf`, HTTP/2 200,
  8,570,670 bytes). Collect the PDF URLs manually or via a real browser, then fetch directly

### 10. Common Pile StackExchange (`common-pile/stackexchange`)

- **License** - `CC BY-SA 3.0` and `4.0`, per-document in metadata. **Commercially usable**
- **Size** - ~30.4M documents, 39.2 GB on disk / ~103.7 GB UTF-8. Built from the December 2024
  community dumps plus the July 2024 official dumps (Stack Exchange ended public XML dumps in
  July 2024)
- **Format** - one document per question with its answers and comments, answers ordered by votes,
  accepted answer first. `metadata` carries the **site URL**, which is the filter lever:
  `mechanics.stackexchange.com` (automotive troubleshooting - the delucionqa domain),
  `diy.stackexchange.com`, `electronics.stackexchange.com`, `superuser.com`
- **Lineage / wall verdict** - **SUSPECT (named doubt)**. Not a RAGBench source, but **ExpertQA's
  evidence documents were retrieved from the open web**, so StackExchange pages can plausibly
  appear among arena documents. Run the gate against expertqa and techqa specifically before
  admitting. Note the sibling risk is real but bounded: the gate exists for exactly this
- **Register fit** - **B, moderate**. Genuine troubleshooting discourse with device vocabulary, but
  conversational Q&A, not manual prose; `procden` will be lower than a TM's
- **Usable document count** - `mechanics` + `diy` + `electronics` together are on the order of
  10^5 questions
- **Arrives as** - clean prose, unlabeled
- **Retrieval effort** - moderate (39 GB; stream and filter on the site URL)

### 11. wikiHow

- **HF versions** - `gursi26/wikihow-cleaned` (tagged `cc-by-nc-sa-3.0`),
  `0x22almostEvil/multilingual-wikihow-qa-16k` (16.8k rows, `cc-by-nc-3.0`). The original
  `wikihow` dataset requires a manual download
- **License** - **CC BY-NC-SA 3.0. NC CONFLICT - FLAGGED.** Same class as the SciFact conflict
  already awaiting the author's ruling. Additionally, wikiHow's own terms have been read as
  **forbidding machine-learning use of the content** outright, which is a stricter bar than NC
- **Lineage / wall verdict** - **CLEAN** on contamination
- **Register fit** - **B, moderate**. Step-structured instructional prose, but overwhelmingly
  human-activity how-to (cooking, relationships, study habits) rather than device/hardware. The
  `procden` keyword set is hardware-weighted, so much of wikiHow will not fire it
- **Recommendation** - do not pursue. Two licence problems for a moderate register match, when a
  public-domain strong match exists

### 12. iFixit / MyFixit (`github.com/rub-ksv/MyFixit-Dataset`)

- **License** - all iFixit content, guides included, is **CC BY-NC-SA 3.0**. **NC CONFLICT -
  FLAGGED**, identical in kind to the pending SciFact question. No HF mirror with a cleaner licence
  was found; a licence cannot be laundered by re-hosting
- **Size** - 31,601 repair manuals across 15 device categories; the Mac Laptop category (1,497
  manuals, 36,659 steps) carries human annotations for required tool, disassembled part and removal
  verb
- **Format** - JSON, one record per guide with ordered steps
- **Lineage / wall verdict** - **CLEAN** on contamination (iFixit is not an EManual or DelucionQA
  source)
- **Register fit** - **B, strong**. Disassembly steps, tools, fasteners, connectors, cautions - the
  closest register match after the Army TMs, and it arrives pre-segmented into steps
- **Usable document count** - 31,601
- **Retrieval effort** - low (git clone)
- **Recommendation** - **hold pending the author's NC ruling**, alongside SciFact. If the author
  rules NC acceptable for a research-only checkpoint, this is the second-best Gap B corpus. If NC
  is rejected, the Army TMs cover the same register at public-domain terms

### 13. Rejected or not worth a section

- **`Heralax/us-army-fm-instruct`** - synthetic instruct pairs generated from Army field manuals by
  Augmentoolkit (2.33M tokens). LLM-rewritten, not source prose; take the manuals directly instead
- **`forcemultiplier/517_Military_Field_Manuals_US_corpus`** - 1.49 GB of page-level PDF extractions
  from US military manuals, schema `pid`/`docid`/`content`/`page_number`/`metadata`. **No licence
  tag on the card** and the viewer fails to load. The underlying documents are public domain, so
  this is a convenience mirror, not a rights problem - but it is undocumented and unverifiable.
  Use as a fallback only if the liberatedmanuals rate limit proves intolerable
- **MPMQA / CheckManual** - product-manual benchmarks, but multimodal (page images); the project is
  torch-free and text-only
- **ManualsLib / manua.ls and similar aggregators** - no licence grant of any kind, and they
  aggregate OEM manuals including Samsung and Jeep. Both a rights problem and a direct
  contamination hazard against emanual and delucionqa. Do not crawl

---

## Ranked shortlist

### GAP B - procedural / product-manual (the more extreme gap, 1,645x)

1. **US Army technical manuals**, filtered to the 1,766 non-parts-list operator/maintenance
   documents - public domain, measured `procden` 23.45 on a representative operator manual against
   an 11.33 bar, and the only candidate that fixes *document diversity* rather than row count.
   Recommendation: take this one
2. **FAA maintenance handbooks** (AMT General / Airframe / Powerplant) - public domain, born-digital
   clean text, strong register; too few documents to stand alone, ideal as a second source that
   widens vocabulary beyond ground vehicles
3. **iFixit / MyFixit**, 31,601 pre-segmented repair guides - strongest step structure of the three,
   **blocked on CC BY-NC-SA until the author rules on NC**; queue it behind the SciFact decision

### GAP A - financial discourse (382x on vocabulary, 2.3x on numerics)

1. **EDGAR-CORPUS section 7 (MD&A), restricted to non-S&P-500 filers and/or year 2020** - the only
   true 10-K discourse at volume, Apache-2.0, and the exclusion filter is the difference between
   SUSPECT-HIGH and a gate the corpus can actually pass. Recommendation: propose it *with the
   filter written into the hypothesis*, and expect the author to weigh it against the standing
   no-EDGAR ruling
2. **Common Pile USGPO**, filtered to Budget / Economic Indicators / Economic Report of the
   President - public domain, unambiguously CLEAN, millions of documents, fiscal rather than
   corporate vocabulary
3. **GovReport** (CRS + GAO) - CC BY 4.0, CLEAN, 19,466 long reports with audit-grade numeric
   narrative; the safest option and the smallest lift

**Honest caveat on Gap A**: none of the three delivers corporate accounting vocabulary at volume
without touching the EDGAR population. Options 2 and 3 close the *discourse* half of the gap
(long-form numeric narrative) while leaving the *lexicon* half (EBITDA, goodwill, shares
outstanding) open. Only option 1 closes both, and only if its filter survives the gate.

## Flags requiring the author's word

1. **EDGAR-CORPUS versus the standing no-EDGAR ruling** - `R10-H108_gate_report.md` and
   `R14_hypotheses_L3_data_mixture.md` both bar SEC/EDGAR material. This scout finds the bar
   well-founded for raw EDGAR and possibly over-broad for a filtered non-S&P-500 slice. The author
   decides whether the filter reopens the door
2. **Two NC conflicts** - iFixit/MyFixit (CC BY-NC-SA 3.0) and wikiHow (CC BY-NC-SA 3.0, plus a
   terms-of-use reading that forbids ML use). Both are the same question already open for SciFact:
   does a non-commercial corpus bar the shipped model
3. **ECTSum has no data licence at all** - GPL-3.0 covers the code; the transcripts are scraped
   Motley Fool content. Stricter than NC, and it is too small to be worth the argument

## Reproduction of the measured numbers

```bash
# liberatedmanuals inventory (4,792 distinct PDFs; 2,203 parts lists; 1,766 procedure manuals)
curl -s https://www.liberatedmanuals.com/all.mpl -o lm.html
grep -oiE 'HREF=/[A-Za-z0-9._-]+\.pdf' lm.html | sed 's#HREF=/##;s#\.pdf$##' | sort -u > u.txt
wc -l < u.txt
grep -cE -- 'P(-[0-9]+)?$' u.txt
grep -E -- '-(10|12|13|14|20|23|24|34)(-[0-9]+)?$' u.txt | grep -vc 'P'

# register measurement per manual: pypdf text extraction over 25 mid-document pages,
# procden = E6 keyword hits per 1,000 chars, digden = digit chars per 100 chars
```
