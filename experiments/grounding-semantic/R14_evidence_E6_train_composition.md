# R14 evidence E6 - are the finqa-like and delucionqa-like registers under-represented in training?

Forensic composition read of the clean public training mix against the blind arena's two
hardest register outliers. Analysis only - no model was trained, no arena number was tuned
against, no tracked artefact was modified.

**Answer in one line**: the delucionqa (automotive product-manual / procedural) register is
effectively **absent** from training - 181 rows out of 685,670 (0.026%), carried by **30
distinct evidence documents**; the finqa (financial-report) register is **under-represented as
a domain** - 904 rows (0.13%) carry finance vocabulary - while the *numeric* surface it shares
with finqa is abundant (43.2% of rows) but delivered in the wrong form (pipe-delimited
Wikipedia tables, not dense financial prose).

## Method

The training mix has no materialized parquet; it is built in memory by `public_train()` in
`experiments/grounding-semantic/R10-H108_lane.py` and consumed positionally by
`R13-H129_trainer.py` and `DR_lane_trainer.py`. It was reconstructed byte-for-byte in build
order by `tmp/R14_E6_build_mix.py` (same zips, same filters, same 1,500-char chunk cap from
`groundrails.config.load_document_processing_config().chunk_max_chars`), with torch and
transformers omitted and one extra `subtag` column added so the RAGTruth task types are
visible. The reconstruction reproduces the registered total exactly: **685,670 rows, 12 DANN
groups**.

Arena profiles come from `experiments/grounding-semantic/R12-H121_gateA_scores.parquet`
(77,171 sentence x window rows, 10 RAGBench subsets). Evidence-side proxies are computed on
`win_text` for the arena and on `chunk` for training; claim-side proxies on `sent_text` and
`claim`. All work in Polars.

## Mix composition (exact)

| group | rows | share | pos rate |
|---|---:|---:|---:|
| vitaminc | 370,653 | 54.06% | 0.5010 |
| tabfact | 92,585 | 13.50% | 0.5508 |
| psiloqa | 61,712 | 9.00% | 0.1092 |
| halueval | 40,000 | 5.83% | 0.5000 |
| ragtruth_en | 15,090 | 2.20% | 0.5546 |
| ragtruth_de | 15,090 | 2.20% | 0.5546 |
| ragtruth_fr | 15,090 | 2.20% | 0.5547 |
| ragtruth_es | 15,090 | 2.20% | 0.5547 |
| ragtruth_it | 15,090 | 2.20% | 0.5546 |
| ragtruth_pl | 15,090 | 2.20% | 0.5548 |
| ragtruth_hu | 15,090 | 2.20% | 0.5547 |
| ragtruth_cn | 15,090 | 2.20% | 0.5547 |
| **total** | **685,670** | **100%** | 0.4819 |

RAGTruth task types ARE distinguishable in the loaded data (`task_type` column, present in the
English parquet and in all seven translations). Each language carries the same split:

- **Data2txt** 5,298 rows per language, 42,384 rows total = 6.18% of the mix - Yelp business
  records serialized as Python-dict / JSON-like key:value blobs
- **QA** 5,034 per language, 40,272 total = 5.87% - MS MARCO web snippets
- **Summary** 4,758 per language, 38,064 total = 5.55% - CNN/DailyMail-style news

PsiloQA carries a `language` column (16 languages, `psiloqa/en` largest at 16,074 rows);
HaluEval splits into `qa` (20,000) and `summarization` (20,000).

## Register proxies

All computed over the evidence text (`chunk` / `win_text`), lower-cased matching where noted.

**Numeric density**

- `digden` - digit characters per 100 chars
- `dig8` - row has >= 8 digit characters
- `curr` - contains `$`, `€`, `£`, `%`, `USD`, `EUR`, `GBP`, `million`, `billion`, `percent`

**Table structure** - `table` fires if any of: >= 3 pipe characters, >= 3 tab characters,
>= 3 JSON-like `"key":` runs, or a `Row N` / `Column N` header.

**Procedural / manual register** - `procden` = keyword hits per 1,000 chars over a fixed
imperative-and-hardware keyword set:

```
press push click tap select ensure "make sure" "turn on/off" "switch on/off"
warning caution "refer to" "see section" "step N"
dashboard indicator vehicle engine brake ignition steering transmission
lever knob button pedal windshield wiper airbag "seat belt" tire tyre fuse
coolant odometer infotainment touchscreen "owner's manual"
install uninstall reinstall remove replace tighten loosen plug unplug
connector cable socket battery reset reboot restart configure setting
```

**Domain lexicons** (>= 2 distinct hits required, to suppress single-word noise)

- `finlex` - revenue, net sales, net income, operating income, gross profit, earnings, EBITDA,
  EPS, fiscal year, quarter, shareholders, shares outstanding, assets, liabilities, equity,
  cash flow, amortization, depreciation, balance sheet, income statement, expenses, dividend,
  goodwill, impairment, tax rate
- `autolex` - vehicle, engine, brake, ignition, steering, transmission, windshield, wiper,
  airbag, seat belt, tire/tyre, coolant, odometer, infotainment, dashboard, Uconnect, Jeep,
  Wrangler, 4WD, AWD, towing, trailer, clutch, axle, headlamp, parking brake, cruise control,
  tailgate, glove compartment

## Arena register profile (evidence windows)

| subset | rows | uniq windows | len med | digden med | dig8 | curr | table | procden med | finlex | autolex | pos rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| techqa | 39,058 | 3,299 | 1500 | 3.00 | 0.912 | 0.123 | 0.007 | 0.67 | 0.001 | 0.020 | 0.096 |
| expertqa | 13,949 | 2,327 | 1500 | 0.60 | 0.505 | 0.147 | 0.011 | 0.00 | 0.002 | 0.010 | 0.631 |
| pubmedqa | 6,570 | 1,154 | 340 | 0.00 | 0.245 | 0.153 | 0.000 | 0.00 | 0.001 | 0.000 | 0.355 |
| **delucionqa** | **5,093** | **398** | **1500** | **0.07** | **0.144** | **0.018** | **0.000** | **11.33** | **0.002** | **0.911** | **0.173** |
| emanual | 2,983 | 131 | 1123 | 0.00 | 0.065 | 0.000 | 0.000 | 6.14 | 0.000 | 0.000 | 0.172 |
| **finqa** | **2,918** | **964** | **1500** | **3.93** | **0.928** | **0.749** | **0.009** | **0.00** | **0.503** | **0.000** | **0.748** |
| tatqa | 1,942 | 706 | 340 | 4.35 | 0.627 | 0.533 | 0.000 | 0.00 | 0.401 | 0.000 | 0.656 |
| hagrid | 1,941 | 694 | 687 | 1.04 | 0.481 | 0.188 | 0.002 | 0.00 | 0.002 | 0.004 | 0.869 |
| covidqa | 1,540 | 900 | 674 | 0.74 | 0.370 | 0.140 | 0.001 | 0.00 | 0.000 | 0.042 | 0.868 |
| hotpotqa | 1,177 | 1,001 | 435 | 2.18 | 0.592 | 0.044 | 0.000 | 0.00 | 0.002 | 0.000 | 0.873 |

Two facts fix the thresholds. finqa evidence is dense-numeric, currency-bearing, financially
worded, and **not tabular** (0.9% table share) - it is SEC-filing prose:

> *"...taking deposits, securities underwriting and trading financial instruments, we make and
> manage direct investments... gs bank usa computes its capital ratios in accordance with the
> regulatory capital requirements..."*

delucionqa evidence is the mirror image - almost no digits, no currency, no tables, and 91% of
windows carry >= 2 automotive terms:

> *"...The trailcam view can also be activated by pressing the icon on the back up camera view.
> When the vehicle is shifted out of REVERSE with Camera Delay turned off..."*

## Training register profile (evidence chunks)

| group | rows | len med | digden med | dig8 | curr | table | procden mean | finlex | autolex |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vitaminc | 370,653 | 139 | 3.01 | 0.271 | 0.160 | 0.000 | 0.06 | 0.0006 | 0.0002 |
| tabfact | 92,585 | 881 | 13.60 | 0.974 | 0.058 | **1.000** | 0.09 | 0.0020 | 0.0127 |
| psiloqa | 61,712 | 281 | 2.43 | 0.520 | 0.019 | 0.001 | 0.04 | 0.0003 | 0.0008 |
| halueval | 40,000 | 816 | 1.20 | 0.595 | 0.118 | 0.000 | 0.11 | 0.0066 | 0.0088 |
| ragtruth_en | 15,090 | 1500 | 1.40 | 0.728 | 0.148 | 0.001 | **0.63** | 0.0095 | 0.0187 |
| ragtruth_cn | 15,090 | 907 | 3.83 | 0.849 | 0.095 | 0.001 | 0.01 | 0.0004 | 0.0004 |
| ragtruth_de/es/fr/hu/it/pl | 15,090 ea | 1500 | 1.27-1.47 | 0.75-0.78 | 0.04-0.09 | 0.001 | 0.01-0.02 | <0.001 | <0.005 |

TabFact is the only genuinely tabular group and it is 100% pipe-delimited by construction (the
loader rewrites `#` separators to ` | `). Its median digit density (13.6 per 100 chars) is 3.5x
the finqa arena median - it is *more* numeric than finqa, in a different surface form and a
different domain (Wikipedia sports/census tables).

## Intensity thresholds and matched mass

Thresholds are set from the arena distributions, not chosen freely:

| threshold | value | source |
|---|---:|---|
| `T_DIG` | 3.933 digits/100 chars | finqa median `digden` |
| `T_DIG_HI` | 6.823 | finqa 75th percentile `digden` |
| `T_PROC` | 11.333 hits/1,000 chars | delucionqa median `procden` |
| `T_LEN` | 800 chars | both subsets sit at the 1,500-char window cap; 800 is the loose floor |

Composite masks, and how they behave on the arena (validation that each mask selects its own
register) versus the training mix:

| mask | definition | finqa | delucionqa | emanual | tatqa | techqa | **train rows** | **train share** | pos rate in slice |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NUM_ANY | `digden >= 3.93` | 0.506 | 0.001 | 0.010 | 0.527 | 0.380 | 296,359 | 43.22% | 0.468 |
| TABLE_ANY | table markers | 0.009 | 0.000 | 0.000 | 0.000 | 0.007 | 92,830 | 13.54% | 0.550 |
| FINQA_LIKE | `len>=800 & digden>=3.93 & curr` | 0.267 | 0.000 | 0.000 | 0.068 | 0.053 | 5,840 | 0.85% | 0.502 |
| FINQA_STRICT | `+ digden>=6.82 & finlex` | 0.029 | 0.000 | 0.000 | 0.035 | 0.000 | 31 | 0.0045% | 0.516 |
| FIN_LEX_ANY | `finlex >= 2 terms` | 0.503 | 0.002 | 0.000 | 0.401 | 0.001 | 904 | 0.13% | 0.519 |
| DELUC_LIKE | `len>=800 & procden>=11.33` | 0.000 | 0.434 | 0.190 | 0.000 | 0.002 | 181 | 0.026% | 0.691 |
| DELUC_STRICT | `+ autolex` | 0.000 | 0.434 | 0.000 | 0.000 | 0.001 | 36 | 0.0053% | 0.639 |
| AUTO_LEX_ANY | `autolex >= 2 terms` | 0.000 | 0.911 | 0.000 | 0.000 | 0.020 | 2,140 | 0.31% | 0.550 |

Every mask separates cleanly on the arena: `FIN_LEX_ANY` fires on finqa (50%) and tatqa (40%)
and nowhere else; `AUTO_LEX_ANY` fires on delucionqa (91%) and nowhere else; the procedural mask
fires on delucionqa (43%) and emanual (19%) and nowhere else.

### Representation ratios

Arena prevalence divided by training prevalence - how many times more common the register is in
the arena subset than in the training mix:

| register mask | train share | finqa share | delucionqa share | finqa / train | delucionqa / train |
|---|---:|---:|---:|---:|---:|
| `finlex >= 2` | 0.132% | 50.34% | 0.22% | **382x** | 1.6x |
| `autolex >= 2` | 0.312% | 0.00% | 91.14% | 0.0x | **292x** |
| `procden >= 11.33 & len >= 800` | 0.026% | 0.00% | 43.43% | 0.0x | **1,645x** |
| `digden >= 3.93 & len >= 800` | 12.02% | 27.72% | 0.06% | **2.3x** | 0.005x |

## Which groups carry the matched mass

**FINQA_LIKE (5,840 rows)** - TabFact 3,157 (54% of the mask, hit rate 3.4%), RAGTruth Data2txt
across all 8 languages 1,998 (34%), remainder scattered. Notably `vitaminc` collapses from
43,767 digit-dense-plus-currency rows to **17** once the 800-char length floor is applied: its
evidence is 139 chars median, so its numeric mass is one-sentence Wikipedia numerics, not
report-length numeric documents.

**FINQA_STRICT (31 rows)** - TabFact only. Nothing else in 685,670 rows clears
"long + very numeric + currency + finance vocabulary" at once.

**FIN_LEX_ANY (904 rows)** - HaluEval 262, VitaminC 234, TabFact 187, ragtruth_en 144, rest
scattered; **358 distinct evidence documents**, 311 of them appearing with a negative label.

**DELUC_LIKE (181 rows)** - ragtruth_en/QA 168 (93%), TabFact 13 (false positives: table cells
containing "setting", "button"). **30 distinct evidence documents.** The RAGTruth QA row count
is inflated 6x because RAGTruth serves the same context to six generator models. The documents
themselves are consumer how-to web snippets from MS MARCO, not OEM manuals:

> *"1 Chock your tires on the side that you won't be working on, and set the tow vehicle's
> parking brake. 2 Use your lug wrench to remove the lug nuts on the flat tire..."*
> *"Step 1 Brush off - Loosen as much rust as possible off the nut with a wire brush..."*

**AUTO_LEX_ANY (2,140 rows)** - TabFact 1,179 (Wikipedia vehicle/transport tables, procden 2.9 -
tabular, not procedural), HaluEval 350, ragtruth_en 282, VitaminC 84. 649 distinct documents.
The non-English RAGTruth automotive slices are 75-100% positive-labelled, so they teach almost
nothing about hallucination in that register.

**TABLE_ANY (92,830 rows)** - 99.7% TabFact. The tabular register is well supplied, but it is
Wikipedia-table-shaped; the arena's table-heavy subsets (finqa 0.9%, tatqa 0.0%) do not present
evidence in that form at all after RAGBench's window serialization.

## Negative-label balance inside the matching slices

| slice | rows | pos rate | negative rows | distinct negative documents |
|---|---:|---:|---:|---:|
| FINQA_LIKE | 5,840 | 0.502 | 2,911 | - |
| FIN_LEX_ANY | 904 | 0.519 | 435 | **311** |
| AUTO_LEX_ANY | 2,140 | 0.550 | 963 | **459** |
| DELUC_LIKE | 181 | 0.691 | 56 | **27** |
| DELUC_STRICT | 36 | 0.639 | 13 | - |

Label balance is not the failure mode: every matched slice carries negatives at roughly 30-50%.
The failure mode is **document diversity**. The entire procedural-manual register is taught by
27 distinct negative evidence documents, and the entire financial register by 311. One epoch
over 685,670 rows shows the model these registers a few dozen to a few hundred times, against
370,653 short Wikipedia sentence pairs and 92,585 pipe tables.

Two secondary observations:

- `psiloqa`'s finance slice (21 rows) is 100% negative - its overall positive rate is 0.109, so
  the group is negative-skewed everywhere, not specifically here
- the non-English RAGTruth automotive rows (ragtruth_es/pl/hu/cn, 54 rows combined) are 94-100%
  positive - the automotive register outside English is positives-only

## Verdict

**Procedural / product-manual register (delucionqa-like): UNDER-REPRESENTED, severely.**
At the arena-median intensity (`procden >= 11.33` per 1,000 chars, evidence >= 800 chars) the
training mix contains **181 rows = 0.026%** of 685,670, drawn from **30 distinct documents**, of
which **27 carry a negative label**. delucionqa hits the same mask on 43.4% of its rows - a
**1,645x** prevalence gap. On the softer automotive-vocabulary test the mix has 2,140 rows
(0.31%) versus 91.1% of delucionqa rows - a **292x** gap - and most of that training mass is
Wikipedia vehicle *tables* (TabFact, procden 2.9), not instructions. No OEM-manual-shaped
evidence exists in the mix at all.

**Financial-report register (finqa-like): UNDER-REPRESENTED as a domain, adequately covered as a
numeric surface.** Finance vocabulary appears in **904 rows = 0.13%** from **358 documents**,
against 50.3% of finqa arena rows - a **382x** gap; the full finqa signature (long + very
numeric + currency + finance vocabulary) matches **31 rows = 0.0045%**, all TabFact. But raw
numeric density is not scarce: 43.2% of training rows clear the finqa median digit density and
12.0% clear it at report length, versus 27.7% of finqa rows - a mild **2.3x** gap that a
2.2%-per-group DANN mix can plausibly bridge. The deficit is domain and discourse, not digits.

**The tabular mass is mis-aimed.** TabFact supplies 13.5% of the mix and 99.7% of all
table-marked rows, at 13.6 digits per 100 chars - richer than finqa. It does not transfer,
because RAGBench's finqa windows are serialized 10-K *prose* (0.9% table markers), and TabFact's
domain is Wikipedia sports and census tables. The mix trains "read a pipe table", the arena asks
"read a financial narrative".

**Consequence for the campaign.** Both blind-arena weak spots sit in registers the training
distribution barely contains, and the delucionqa gap is the more extreme of the two by an order
of magnitude. Neither is fixable by re-weighting existing DANN groups - the rows are not there
to re-weight. Closing them requires new evidence sources: procedural/manual documents with
adversarial negatives (30 documents is not a register), and financial-narrative documents that
stay behind the contamination wall (no ConvFinQA / TAT-HQA / FinanceBench / FinTabNet / EDGAR
material, per the R10-H108 provenance gate).

## Reproduction

```bash
cd /home/lab/workspace/private/ai-assistants/groundrails
HF_HUB_OFFLINE=1 uv run python tmp/R14_E6_build_mix.py      # rebuilds the 685,670-row mix
HF_HUB_OFFLINE=1 uv run python tmp/R14_E6_registers.py      # per-group proxies, arena thresholds
HF_HUB_OFFLINE=1 uv run python tmp/R14_E6_registers2.py     # composite masks + samples
HF_HUB_OFFLINE=1 uv run python tmp/R14_E6_registers3.py     # lexicon slices, representation ratios
```

Scripts live in `tmp/` (gitignored, analysis-only). No tracked file was modified; no arena label
was used for anything except characterizing the arena.
