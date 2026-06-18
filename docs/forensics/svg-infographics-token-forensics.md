# SVG Infographics - Token and Time Forensics

Why producing 7 SVG infographics in this project (5 executive-summary images + 2 slides) consumed an outsized share of session time and tokens. Sources: the 42 MB session transcript, the project artifacts and journal (entries 41-46, 54), and a design-cost audit of the svg-infographics plugin. Companion document: `reports/svg-infographics-plumbing-forensics.md` (the 14+1 correction table).

## Session accounting

One session transcript covering 2026-06-02 → 2026-06-12 (SVG work is a subset; the session also carried grounding-library and RAG-assistant work).

- **Output tokens** - 14,665,007 total; SVG-attributable share estimated 1-2M (design cycles, fix rounds, subagent spawns)
- **Input tokens** - 2,618,266 fresh; 103,343,573 cache-creation; 5,004,882,452 cache-read (cheap, 0.1x)
- **Messages** - 5,835 assistant / 3,153 user; 2,550 tool-use blocks
- **svg-designer subagent spawns** - 14, for 7 shipped SVGs: every spawn reloads the full plugin context
- **Fix churn** - 1,720 transcript lines matching fix/correct, 262 matching regenerate/retry, against 14 documented human-requested corrections
- **Render verification** - 3-4 Playwright render→inspect→fix rounds per image (journal 44: "each round render-verified"); 19 screenshots, 66 browser operations
- **Downstream churn** - DOCX re-exported after every visible fix (journal 44-46), v11/v12 versioned exports

## Root causes, ranked

### 1. Validator blindness → human render loop (dominant)

The plugin's `check`/`finalize` gates passed all five files, yet 14 corrections were needed - 10 of them invisible to any checker, 1 flagged but acked away, 3 enabled by false-positive noise that trained the ack reflex.

- **Missed entirely** - icon/text collisions (heuristic char-width bbox, not real glyph extents), uneven corner padding, dead-canvas imbalance (115px), label centering, edge margins, cross-file consistency, card-anatomy drift, stat-slot parity
- **Consequence** - every defect was caught by eye on a render, fixed, re-rendered: the most expensive loop in the whole pipeline (model regenerates SVG sections + re-runs validators + re-renders, times 14)
- **Cost multiplier** - each correction round re-enters a subagent or fix flow that re-reads the SVG and re-justifies validator findings

### 2. Per-spawn context reload

Each of the 14 svg-designer forks loads ~150 KB before any work: SKILL.md 14 KB, four reference docs 72 KB (standards 33.6 KB alone), preflight rule cards 16-43 KB, plus a mandated read of 3-5 example SVGs (~60 KB). The plugin's create command forbids batching ("each image... no batching"), so sibling images in the same deck cannot share a loaded context - 14 spawns ≈ 1.4 MB of redundant context, ~20k tokens overhead per spawn before design starts.

### 3. Noisy validators trained the ack reflex

- 73 grid-snap notices on one file (checker tolerance defaults to 0; half-pixel centring residue each emitted as a finding)
- 10 "missing dark override" false positives per file (a non-greedy regex parses only the first class inside the `@media` dark block)
- Icon paths inside `scale()` groups misclassified as routed connectors (parent transform never applied)
- 51 warnings acknowledged one-by-one on a single file - the gate requires one `--ack-warning TOKEN=reason` per finding, no bulk ack
- Net effect: justifying noise costs tokens directly, and the ack habit neutralised the one real finding the gate did raise (correction 4)

### 4. Validation-loop design

`finalize` runs only 3 of the 12 checkers; the validate command mandates 7 separate CLI invocations per file; the workflow's Phase 6 says "loop until clean" - re-run everything after each fix - and the Default-Bad rule demands an individual written defence of every finding. A 20-violation image costs 120+ validator invocations, each with output read back into context.

## Per-image cost model

~50-65k tokens per image before any visual defect: ~20k spawn overhead (context reload) + ~12k example reads + ~10k authoring phases + ~3-10k validation loop. The render-loop multiplier then dominates: each human-caught defect adds a fix round (resend SVG section, re-validate, re-render). Seven images with 14 corrections ≈ 350k-700k tokens beyond the ~400k structural floor.

## Remediation (svg-infographics plugin, agreed plan)

Implemented in the plugin repository, ordered by value/effort:

1. **Bug fixes** - speech-bubble spike corner-radius clamp (row 15); dark-mode `@media` parser (brace counting); connector classification (apply transforms, require stroke/no-fill, exempt icon groups)
2. **Noise reduction** - grid-snap tolerance 0.5 default with aggregated summary line; `--ack-class` bulk ack for SOFT/NOTICE findings
3. **Consolidated gate** - `finalize` runs all checkers in one call, one severity-tiered report (HARD/SOFT/NOTICE); batch-fix protocol: validate once → fix everything → re-validate once, hard cap 3 iterations; individual defence required for HARD findings only
4. **Render-based visual gate** - no new browser machinery: the existing `render-png` Chromium path additionally extracts real `getBBox()` extents per element in the same render; pure-geometry checks over that data catch the documented failure modes (text/icon collisions, corner-padding symmetry, label centering, canvas balance, stat-slot parity); a `consistency` check diffs card anatomy across sibling SVGs; fontTools text extents as the browser-free fallback
5. **One readback instead of N render rounds** - after the gate passes, render PNG once (light+dark) and run a single fixed-checklist model readback - replacing the 3-4 interactive rounds per image
6. **Context-cost cuts** - mandatory per-spawn reads shrink from ~150 KB to ≤30 KB (index + 1 example + preflight rule bundle; references on demand); deck batching: one agent spawn per up to ~5 sibling images with a closing cross-file consistency check
7. **Placement helpers** - `place --ref-id` so icon/badge padding is measured from a reference element (accent bar) instead of converging over three correction rounds; manifold/spline suggestions emitted at preflight for bipartite layouts

## Expected effect

- Structural floor per image drops ~40% (context reload + example reads)
- The dominant render-loop multiplier collapses: machine-caught geometry defects are fixed inside the batch-fix protocol; visual judgment spends one readback pass per image instead of 3-4 human rounds
- Ack noise no longer trains agents to dismiss real findings; remaining acks are bulk-class, not per-token
