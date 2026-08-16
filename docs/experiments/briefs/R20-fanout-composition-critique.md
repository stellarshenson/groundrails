# R20 fanout addendum - composition critique (unsolicited, from the H174 lane-build agent)

Fable design brief (2026-08-16). Produced by the R20-H174 lane-build agent after its stage-0 deliverables, delivered on stop. ADJUDICATED 2026-08-16 (canonical log, "R20 COMPOSITION-CRITIQUE ADJUDICATION" block): the token-head card is REFUTED by the record (R8-H102 took that exact read at full scale and was KILLED 0.7051 vs 0.7172; R9-H106 killed the P-A fusion salvage); the mmBERT lambda-0 hole is CONFIRMED real (candidate card, queue decision deferred); the lexical late-fusion card is CONFIRMED unspent and registered as R20-H178; the H166-A1 budget-cut proposal is DECLINED without author word. Read the agent text below against that adjudication.

---

## The reframing

The gap is **0.0245** from the flagship pair mean 0.71549. Everything registered on the board sums to **+0.018 central**, landing 0.7335. Brief B's independent arithmetic is more pessimistic still at ~0.726. So the queued plan does not reach the target, and that is not a supply problem - it is a **composition** problem.

The decisive evidence is already banked and it is being under-weighted: **R19-H159 added five lanes at once and landed 0.68941, which is 0.026 BELOW the two-lane flagship.** Every in-domain and anti-gaming guard was green. The per-lane yield in this campaign is +0.005..+0.010; reaching 0.0245 by lanes needs four or five of them to add, and the one time five were tried the sum was negative. The diagnosed mechanism is that lanes teaching "high lexical overlap is not support" destroy the table subsets where high overlap *is* support.

So the research question is not "which lane next". It is **"does anything in this design compose, and if not, which single lever is large enough on its own?"**

## What the record says is unspent

Two things sit in the log with large predictions and no blind read, and both have been quietly skipped for a dozen rounds.

**1. Token-classification head at full scale.** The transfer ranking at line 1946 records it as candidate (1): *"token-head-only blind read never taken; +0.01-0.03 predicted, needs multi-seed"*. That is the **largest unspent single prediction in the campaign**, it changes the readout geometry rather than the data, and it costs one draw to first read. Nothing in 12 rounds explains why it was never taken.

**2. The mmBERT lambda-0 control.** The log states three separate times, in escalating language, that *"after ~6.5 GPU-h across two arms, the author's ordered EuroBERT-versus-mmBERT comparison does not exist"* - because there is no mmBERT counterpart at DANN lambda 0. That single missing draw does double duty: it settles the ordered trunk question, and it is the direct test of the composition hypothesis. DANN pushes toward domain-*invariant* features; the behaviour lanes need is domain-*conditional* (overlap means support in tabfact, not in fava). If the adversary is what makes lanes fight, one ablation draw shows it. H169 ran lambda 0 on EuroBERT and it still failed, but EuroBERT was broken for other reasons, so the mmBERT arm remains the clean test.

**3. Lexical fusion, never re-tested at arena stage.** R19-H162 measured, on the arena: emanual model **0.6973** against plain token containment **0.7763** - the trained cross-encoder is *below* a lexical baseline on a whole subset - while delucionqa runs the other way at model 0.8009 vs lexical 0.5889. That is strong error decorrelation. The lexical layer was dropped back in round 2 on gold-set evidence, before the arena existed and before the current single-cross-encoder recipe, and the drop has never been revisited. It costs **zero training**: a blend weight selected on gold_full only, then one blind arena read. The lexical tier already ships in `groundrails`, so this is architecturally native, not a bolt-on.

## The graph, as dependencies

**Level 0 - fix the ruler.** H172 k=6 must land before any bar is priced. The multiplicity question (48 blind reads, flagship is the max) is a threat edge, not a contribution: applying a correction moves the baseline down and widens the gap.

**Level 1 - three cards, three questions, in parallel.** The current queue serializes H174 → H166-A1 → H177 and spends 52 GPU-h before learning anything about composition. That is the wrong order. Instead:

- **Card 1**: H174 draw 1 (the promotion route, lanes already built, gate at 0.695 + table guard)
- **Card 2**: mmBERT at lambda 0, one draw - answers composition *and* the ordered trunk question
- **Card 3**: token-head at full scale, one draw - the +0.01..0.03 that was never read

Lexical fusion runs alongside on CPU plus one read; it needs no card.

**Level 2 - branch on Level 1.** If lambda-0 improves lane composition, the portfolio route is alive and H177 stacks on top. If it does not, the data route is capped and the budget moves to capacity: the Qwen3-0.6B decoder scorer, recorded as *"the only candidate changing capability class, attacks numeracy"* - which is where finqa and tatqa live. That one needs explicit author reopening of the sub-400M budget.

**Level 3 - convention parity (H175), author-gated.** The measured gap is +0.155 on hagrid and +0.169 on emanual. Full transfer on those two subsets alone is +0.032 on the mean; quarter transfer is +0.008. It is the only lever sized to a residual above 0.015. But the concat half is measured *not* to move hagrid (-0.003), so the size rests on the question channel, which is blocked on whether `ground()` gets a `question=` parameter.

## Two routes that actually reach 0.74

The reason to prefer this ordering is orthogonality. Lanes fight each other because they all edit the same overlap prior. A readout change, a score blend, and a data lane edit three different parts of the system, so they have a real claim to adding.

- **Author-free route**: H174 +0.014 central, token head +0.015 central, lexical fusion +0.008 → **0.7525**, clears with margin
- **Author-gated route**: H174 +0.014, convention parity +0.008..0.032 → 0.7375 to 0.7615

Both need one of the unspent levers. Neither works on lanes alone.

## What I would cut

**H166-A1's 13 GPU-h.** Its own registered prediction is [-0.005, +0.008] on the mean with a negative lower bound, and its PRIMARY is a mechanism gate, not a mean. That budget buys the token-head draw, which has a prediction three times larger and an actual promotion route. Keep H166-A1 for its contradiction channel, run it last.

## The falsifier

If H172's k=6 mean lands materially below 0.7155, or a multiplicity correction is applied, the gap widens past 0.03 and even the author-free route above stops clearing. At that point the honest move is to record that 0.74 is not reachable with this trunk and this convention, rather than keep spending draws against it. The campaign's own oracle bound for a *perfect* content-conditional gate is 0.7369 - under target - which is the strongest single hint that the finish is not in read-side selection at all.

Everything above is a proposal for the coordinator to adjudicate and register; I have registered nothing. The stage-0 lane artifacts from my actual task are unchanged and complete.
