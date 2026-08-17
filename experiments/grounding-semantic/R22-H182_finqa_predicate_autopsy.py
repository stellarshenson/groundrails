"""R22-H182 finqa predicate autopsy - assemble the classification artifact.

Manual read of all 20 finqa negatives and 40 seed-182 positives. Arithmetic
verified numerically (see `computed_value` fields). Analysis only; CPU only.
"""
import json, random, datetime, pathlib
import polars as pl

SRC = "experiments/grounding-semantic/R21-H179_arena_items.parquet"
OUT = pathlib.Path("experiments/grounding-semantic/R22-H182_finqa_predicate_autopsy.json")
SEED = 182

df = pl.read_parquet(SRC).filter(pl.col("subset") == "finqa")
neg = df.filter(pl.col("label") == 0).sort("item")
pos_pool = df.filter(pl.col("label") == 1).sort("item")
POS_ITEMS = sorted(random.Random(SEED).sample(pos_pool["item"].to_list(), 40))
rows = {r["item"]: r for r in df.iter_rows(named=True)}

D = "operands_present_derivation_wrong"
R = "operands_present_direction_wrong"
W = "wrong_operand_selected"
A = "operand_absent_from_evidence"
F = "attribution_failure"
O = "other"

# item: (class, deciding_span, evidence_span, asserted, computed, has_deriv, deriv_ok, reason)
NEG = {
31: (R, "Total shareholders' equity decreased by $2,749 million from 2012 to 2013.",
     '["total shareholders 2019 equity", "$ 78467", "$ 75716"]', -2749.0, 2751.0, True, False,
     "Both operands verbatim in the table; 78,467 - 75,716 = +2,751, so the asserted change is wrong in sign (an increase, not a decrease) and off by 2 in magnitude."),
36: (W, "So as of December 31, 2012, the scheduled maturities of long-term debt made up 109.7% of the long-term debt obligations.",
     "the following are the scheduled maturities of long term debt as of december 31 , 2011", 109.7, 109.7146, True, True,
     "Arithmetic exactly right (77,724/70,842 = 109.71%); the defect is the period label - the table is as of December 31 2011 and the answer attaches the result to 2012. Twin item 5 (label 1) makes the same 2012 attribution on identical question and documents."),
41: (D, "The company paid a total of approximately $82,660,099 for the repurchase of shares on non-announced plans or programs ($109.32 per share x 75,671 shares).",
     '["september 29 2019 through november 2 2019", "1023202", "$ 109.32", "949531", "$ 2070927831"]', 82660099.0, 8053713.72, True, False,
     "109.32 is verbatim; the share count is a mis-derived subtraction (1,023,202 - 949,531 = 73,671, not 75,671) and the product is out by a factor of ten even against its own operands (109.32 x 75,671 = 8,272,354). Whole-quarter answer is 110.17 x 81,832 = 9,015,431."),
43: (O, "Total purchase commitments over the next two years are $3,772,000.",
     '["total", "$ 3772"]', 3772000.0, 3772000.0, True, True,
     "Scale/unit mismatch, not an attribution failure: 3,772 (in thousands) = $3,772,000 is the correct dollar amount, but the question asks for the answer in thousands (3,772). The scale derivation itself is right."),
48: (D, "The total value of issued securities that are approved by security holders is $59,808,536.48 million.",
     '["equity compensation plans approved by security holders", "2590898", "$ 22.08", "942512"]', 59808536.48, 57207027.84, True, False,
     "Both operands verbatim; 2,590,898 x 22.08 = 57,207,027.84, so the asserted product is wrong (and the trailing 'million' is a spurious unit)."),
67: (W, "The portion of total smokeless products shipments related to the Copenhagen segment during 2014 is 448.6 million cans and packs.",
     '["copenhagen", "448.6", "426.1", "392.5"]', 448.6, 56.549, False, None,
     "448.6 is the correct Copenhagen-2014 cell and is fully attributable verbatim; it is the wrong quantity for a question asking a portion (448.6/793.3 = 56.5%). No derivation is attempted at all."),
71: (W, "The value in thousands of unvested restricted stock and performance awards at the weighted-average grant-date fair value as of December 31, 2017, was $78.29 million.",
     '["unvested at december 31 2017", "1226", "78.29"]', 78.29, 95983.54, False, None,
     "78.29 is the per-share weighted-average grant-date fair value, verbatim in the table, reported as the total. Right cell, wrong quantity; the required product 1,226 x 78.29 = 95,983.54 (thousands) is never formed."),
81: (O, "The yearly depreciation rate for land improvements is 5%.",
     '["landimprovements", "20"]', 5.0, 5.0, True, True,
     "The derivation is correct - straight-line over a 20-year life is 5%/yr. The label penalises an implicit but correct derivation whose result is not verbatim in the evidence; the annotation itself concedes the rate is derivable and faults only the missing explanation."),
85: (O, "But it does not provide the 2016 pre-tax earnings figure.",
     "pre-tax earnings were $ 1.42 billion in 2017 , 25% ( 25 % ) higher than", None, 1.136, False, None,
     "A refusal: the response claims the 2016 figure is absent. It is not stated explicitly, but it is derivable from the '25% higher than' relation (1.42/1.25 = 1.136 billion). The label turns on a missed derivation, not on a false fact attributed to the evidence."),
114: (R, "Therefore, the percent of the outstanding notes under the ati 7.25% (7.25%) notes decreased by approximately 18.73% from 2005 to 2006.",
     "the company had $ 325.1 million and $ 400.0 million outstanding under the ati 7.25% ( 7.25 % ) notes , respectively", 18.73, 81.275, True, True,
     "Every operand is verbatim and the internal arithmetic is exactly right (400.0 - 325.1 = 74.9; 74.9/400 = 18.725%). The relation is wrong for the question: 'percent of the outstanding notes for 2006 to 2005' is the ratio 325.1/400.0 = 81.3%, not a percent decrease."),
116: (D, "Therefore, if a similar increase occurs in 2009, the satisfaction index would become 87 in 2010.",
     '["customer satisfaction index", "88", "83", "79", "5 pt", "4 pt"]', 87.0, 92.0, True, False,
     "The 2008 increase (79 -> 83 = +4) is read correctly, but the projection is anchored on the 2008 value and labelled 2010; the 2009 value (88) sits in the same table and is ignored, so the 2010 projection should be 88 + 4 = 92."),
157: (W, "In 2007, the total debt was $3,165 million and the sum of lease obligations and purchase obligations was $2,473 million ($144 million lease obligations + $2,329 million purchase obligations).",
     '["total debt ( a )", "$ 692", "$ 129", "$ 1143", "$ 1198", "$ 381", "$ 3680"]', 127.87, 27.9822, True, False,
     "3,165 is the 2007 column TOTAL row, not the total-debt row (692); 144 + 2,329 = 2,473 is right. Correct answer 692/2,473 = 27.98%. Its own ratio is also mis-rounded (3,165/2,473 = 127.98%, not 127.87%)."),
168: (D, "Therefore, without the loss in volume/weather, net revenue would have experienced a 2.8% increase between 2016 and 2017.",
     '["2016 net revenue", "$ 705.4"], ["volume/weather", "-18.2 ( 18.2 )"]', 2.8, 2.254, True, False,
     "Adds the volume/weather loss to the 2016 base instead of to 2017: correct answer is (703.1 + 18.2 - 705.4)/705.4 = +2.25%. Its own working line computes -2.8% and the conclusion then flips the sign to 'a 2.8% increase'."),
189: (R, "The total value of currency maturities for 2010, excluding the euro maturities, would be $305 million.",
     '["euro", "-372 ( 372 )"]', 305.0, -305.0, True, False,
     "Every component is quoted correctly and the sum is right (-305, matching -677 - (-372)); only the sign is dropped in the answer sentence, whose own next line reads 'Total (excluding Euro): -305 million'. Twin item 247 (label 1) keeps the sign on identical question and documents."),
198: (R, "The difference between these two values is $10,728 million - $10,560 million = $168 million decrease.",
     '["total", "$ 10728", "$ 10560"]', -168.0, 168.0, True, False,
     "Right magnitude, wrong direction: 10,728 (2014) exceeds 10,560 (2013), so the change is +168. The response also opens with 'decreased by $85 million', which is the change in unfunded commitments (717 - 802 = -85) - the quantity the question names - so it contains both a defensible answer and a contradictory one."),
200: (W, "To calculate the ratio, we need to divide the net increase in securities sold under agreements to repurchase ($6.2 billion) by the net transfers in for long-term debt ($38.8 billion).",
     "the increase in securities sold under agreements to repurchase of $ 5 2022 billion is driven by a $ 6.2 billion increase from net transfers in", 0.16, 0.8065, True, False,
     "38.8 is the long-term-debt net transfers in - a real number from the passage, the wrong one for this ratio; 6.2/38.8 = 0.16 is arithmetically right on the wrong pair. Correct ratio 5/6.2 = 0.81. Secondary defect: the response asserts '$5.2 billion' where the passage reads '$5 billion' (a stray bullet glyph '2022' sits between '5' and 'billion')."),
214: (D, "28% of the total future minimum lease commitments is due in 2003.",
     '["2003", "$ 30", "$ 4"]', 28.0, 17.7515, True, False,
     "Both operands verbatim; 30/169 = 17.75%. The response's own working line reads '($30 million / $169 million) * 100 = 17.75% ~ 28%' - it computes the right value and then asserts 28."),
215: (O, "If we assume a similar improvement in 2011 compared to 2010, we can expect the operating ratio in 2011 to be around 65.1%.",
     "our operating ratio improved 5.5 points to 70.6% ( 70.6 % ) in 2010", 65.1, 65.1, True, True,
     "Correct hypothetical derivation (70.6 - 5.5 = 65.1), which is exactly what the question asks. The label penalises the counterfactual because the same passage says the 2011 ratio actually rose 0.1 points to 70.7%."),
229: (W, "Therefore, the ratio of the statutory capital and surplus to the statutory net income of the Bermuda subsidiaries in 2009 is approximately 10.24.",
     '["statutory capital and surplus", "$ 7001", "$ 8579", "$ 7605", "$ 5337", "$ 5321", "$ 4431"]', 10.24, 10.2354, True, True,
     "7,001 and 684 are the 2008 Bermuda columns; the table covers 2008/2007/2006 only and no 2009 figures exist. Right arithmetic (7,001/684 = 10.24), wrong year."),
242: (W, "Of this total, approximately 57,800 square feet is occupied by American Tower Corporation's lease administration office and broadcast division headquarters.",
     "approximately 57100 square feet of space is occupied by our lease administration office and our broadcast division", 57800.0, 57100.0, False, None,
     "57,800 is the Woburn size cell in the properties table; 57,100 is the occupied-space figure in the footnote. A real number from the evidence, the wrong cell. The portion asked is 57,100/163,000 = 35.0%, never computed."),
}

# positives: (has_deriv, deriv_ok, asserted, computed, note)
POS = {
5:   (True, True, 109.7, 109.7146, "Correct derivation, operands verbatim. Carries the same 2011->2012 period mislabel for which twin item 36 (label 0, identical question and documents) is faulted."),
7:   (True, False, 14.2, 12.0275, "Derivation present and WRONG in a positive: the table's total row sums to 28,809, giving 3,465/28,809 = 12.03%; the response's own listed addends sum to 28,422, not the 24,419 it states, and it asserts 14.2%."),
8:   (True, True, 11.25, 11.2369, "Correct derivation, operands verbatim (rounding 11.24 -> 11.25)."),
19:  (True, False, 71.98, 72.1005, "Derivation present, mildly wrong: (10,558 + 6,426)/23,556 = 72.10%, asserted 71.98%."),
23:  (True, True, 1.71, 1.7145, "Correct derivation, operands verbatim."),
25:  (True, True, 1.78, 1.7804, "Correct derivation, operands verbatim."),
42:  (True, True, -26.07, -26.0664, "Correct derivation, operands verbatim."),
44:  (True, True, 58.04, 58.04, "Correct derivation, operands verbatim."),
59:  (True, True, 305261.0, 305261.0, "Correct derivation, operands verbatim."),
62:  (True, False, 17.14, 17.2665, "Derivation present, mildly wrong: (177.26 - 151.16)/151.16 = 17.27%, asserted 17.14%."),
64:  (True, True, 2.63, 2.6333, "Correct derivation, operands verbatim."),
65:  (True, False, 28587554.58, 28594940.30, "Derivation present, mildly wrong: 2,111,138 x 9.25 + 1,116,615 x 8.12 = 28,594,940.30 (and its own 3,227,753 x 8.86 = 28,597,891.58); asserted 28,587,554.58."),
75:  (False, None, None, None, "NO arithmetic derivation - two figures quoted verbatim from the passage."),
77:  (True, True, 0.33, 0.3261, "Correct derivation, operands verbatim."),
79:  (True, True, 11.8, 11.8252, "Correct derivation, operands verbatim."),
90:  (True, True, 14.6, 14.6358, "Correct derivation, operands verbatim."),
98:  (True, True, 41.0, 41.0845, "Correct derivation, operands verbatim."),
99:  (True, True, 309.67, 309.6667, "Correct derivation, operands verbatim."),
103: (True, True, 10.0, 10.0, "Correct derivation (200 - 190), operands verbatim."),
126: (True, True, 282.0, 282.0, "Correct derivation (516 - 234), operands verbatim."),
129: (True, True, -6.43, -6.4304, "Correct two-stage derivation (762 x 42, 713 x 42, then percent change)."),
140: (True, True, 4.8, 4.8146, "Correct derivation, operands verbatim."),
143: (True, True, 1.69, 1.6923, "Correct derivation, operands verbatim."),
147: (True, True, -71.0, -71.0, "Correct derivation (38 - 110 + 1 = -71, matching 63 - 134)."),
151: (True, True, 615.38, 615.3846, "Correct derivation, operands verbatim."),
158: (True, True, 283.0, 283.0, "Correct derivation, operands verbatim."),
173: (True, True, 17.28, 17.2775, "Correct derivation, operands verbatim."),
174: (False, None, None, None, "NO arithmetic derivation - goodwill figure quoted verbatim."),
181: (True, True, 93.6, 93.5713, "Correct derivation, operands verbatim."),
182: (True, True, -7.8, -7.7705, "Correct derivation, operands verbatim."),
191: (True, True, 498.0, 498.0, "Correct derivation, operands verbatim."),
196: (True, True, 2.1, 2.1, "Correct derivation, operands verbatim."),
201: (True, True, 34.8, 34.8066, "Correct derivation, operands verbatim."),
202: (True, True, -4079.0, -4079.0, "Correct derivation, operands verbatim."),
210: (True, True, 2.0, 2.0, "Correct derivation, operands verbatim."),
218: (True, True, 53.55, 53.55, "Correct derivation, operands verbatim."),
224: (True, True, -3.44, -3.4375, "Correct derivation, operands verbatim."),
231: (True, True, 6.39, 6.39, "Correct derivation, operands verbatim."),
233: (True, True, -7.92, -7.9167, "Correct derivation, operands verbatim."),
247: (True, True, -305.0, -305.0, "Correct derivation with the sign kept. Twin item 189 (label 0, identical question and documents) drops the sign in its answer sentence - the only difference between the two."),
}

problems, records = [], []
for it, (cls, span, ev, asserted, computed, hd, dok, reason) in sorted(NEG.items()):
    row = rows[it]
    if span not in row["response"]:
        problems.append(f"NEG {it}: deciding span not verbatim in response")
    if ev is not None and not any(ev in d for d in row["documents"]):
        problems.append(f"NEG {it}: evidence span not verbatim in documents")
    records.append(dict(item=it, label=0, leg="negative", **{"class": cls},
                        deciding_span=span, evidence_span=ev, asserted_value=asserted,
                        computed_value=computed, contains_arithmetic_derivation=hd,
                        derivation_correct=dok, undecidable=False, reason=reason))

for it in POS_ITEMS:
    hd, dok, asserted, computed, note = POS[it]
    records.append(dict(item=it, label=1, leg="positive", **{"class": O},
                        deciding_span=None, evidence_span=None, asserted_value=asserted,
                        computed_value=computed, contains_arithmetic_derivation=hd,
                        derivation_correct=dok, undecidable=False, reason=note))

if problems:
    raise SystemExit("SPAN VALIDATION FAILED:\n" + "\n".join(problems))

def counts(recs):
    c = {}
    for r in recs:
        c[r["class"]] = c.get(r["class"], 0) + 1
    return c

negs = [r for r in records if r["leg"] == "negative"]
poss = [r for r in records if r["leg"] == "positive"]
nc = counts(negs)
three = nc.get(D, 0) + nc.get(R, 0) + nc.get(W, 0)
two = nc.get(D, 0) + nc.get(R, 0)
pos_deriv = sum(1 for r in poss if r["contains_arithmetic_derivation"])
neg_deriv = sum(1 for r in negs if r["contains_arithmetic_derivation"])
pos_deriv_wrong = [r["item"] for r in poss if r["derivation_correct"] is False]

art = dict(
    experiment="R22-H182", subset="finqa", source_parquet=SRC,
    method=("Manual per-item read of all 20 finqa negatives and 40 seed-182 positives. Every asserted "
            "derived quantity recomputed from the evidence figures. Deciding spans and evidence spans "
            "are validated as verbatim substrings of the response / documents at build time."),
    seed=SEED, positive_sample_items=POS_ITEMS,
    negative_items=sorted(NEG), n_classified=len(records),
    class_counts_negatives=nc, class_counts_positives=counts(poss),
    negatives_derivation_class_share=dict(
        headline_three_classes=dict(
            classes=[D, R, W], count=three, share=round(three / 20, 4),
            verdict="CONFIRMED" if three / 20 >= 0.60 else ("REFUTED" if three / 20 < 0.30 else "PARTIAL"),
            note=("The three classes enumerated first in the spec - all share the property that every operand "
                  "the response uses is present in the evidence and the failure lies in the numeric/relational layer.")),
        strict_arithmetic_only=dict(
            classes=[D, R], count=two, share=round(two / 20, 4),
            verdict="CONFIRMED" if two / 20 >= 0.60 else ("REFUTED" if two / 20 < 0.30 else "PARTIAL"),
            note=("Narrow reading in which 'derivation' means only a wrong computed result or a wrong "
                  "direction/relation, excluding wrong-cell / wrong-year operand selection.")),
        ambiguity=("The pre-registered bar says 'the three derivation classes' without naming them. Both readings "
                   "are reported; the caller adjudicates which one the bar meant.")),
    positives_containing_arithmetic_derivation=dict(
        count=pos_deriv, n=40, fraction=round(pos_deriv / 40, 4),
        without_derivation=[r["item"] for r in poss if not r["contains_arithmetic_derivation"]],
        with_wrong_derivation=pos_deriv_wrong,
        note=("38 of 40 supported responses perform an arithmetic derivation, so derivation-PRESENCE does not "
              "separate the legs. 4 of those 38 get the arithmetic wrong and are still labelled supported "
              "(item 7 grossly: 14.2% asserted against 12.03% correct; items 19, 62, 65 by rounding-scale margins).")),
    negatives_containing_arithmetic_derivation=dict(count=neg_deriv, n=20, fraction=round(neg_deriv / 20, 4)),
    attribution_findings=dict(
        attribution_failure_count=nc.get(F, 0), operand_absent_count=nc.get(A, 0),
        note=("Zero of the 20 negatives is a non-numeric attribution failure and zero asserts a figure absent "
              "from the evidence as a given. Two near-misses are mis-derivations of present figures: item 41's "
              "'75,671 shares' (1,023,202 - 949,531 = 73,671) and item 200's '$5.2 billion' where the passage "
              "reads '$5 billion'. In 4 of 20 negatives (43, 81, 85, 215) the response's numeric content is "
              "correct or defensible and the item is still labelled unsupported."),
        negatives_with_numerically_wrong_answer=16,
        negatives_with_correct_or_defensible_answer=[43, 81, 85, 215]),
    label_conflicts=[
        dict(items=[36, 5], labels=[0, 1],
             note=("Identical question and identical documents. Both responses compute 77,724/70,842 = 109.7% and "
                   "both attach the result to December 31 2012 although the table is as of December 31 2011. "
                   "Item 36 is labelled unsupported for exactly that; item 5 is labelled supported.")),
        dict(items=[189, 247], labels=[0, 1],
             note=("Identical question and identical documents. Both enumerate the same nine non-euro maturities and "
                   "both sum them to -305. Item 189 writes '$305 million' in its answer sentence (sign dropped) and "
                   "is labelled unsupported; item 247 writes '-305 (in US$ millions)' and is labelled supported. "
                   "The sign of the answer is the only difference.")),
        dict(scan="All 250 finqa items grouped by (question, documents): 13 duplicate-context groups, "
                  "exactly 2 of them carry conflicting labels - the two above.")],
    annotation_disagreements=[
        dict(item=114, disagreement=("Annotation: 'The amount used here ($74.9 million) refers to the amount repurchased, "
             "not the decrease in outstanding notes; hence, this calculation is incorrect.' 400.0 - 325.1 = 74.9 exactly, "
             "so the decrease and the repurchase are the same number and the annotation's stated reason is arithmetically false.")),
        dict(item=198, disagreement=("Annotation: 'Documents show a decrease of $168 million instead.' The table shows an "
             "INCREASE of 168 (10,728 at Dec-31-2014 against 10,560 at Dec-31-2013); the annotation repeats the response's sign error.")),
        dict(item=85, disagreement=("Annotation claims the 2016 pre-tax earnings appear in Document 2 ('sentence 2x'). The passage "
             "is cut off after 'pre-tax earnings were $ 1.42 billion in 2017 , 25% ( 25 % ) higher than' and never states the 2016 "
             "figure; it is only derivable as 1.42/1.25 = 1.136 billion.")),
        dict(item=43, disagreement=("Annotation is self-contradictory: it says the response 'overstates the amount because it "
             "misinterprets the in-thousands notation', then says 'the statement should be interpreted as $3,772,000' - which is "
             "exactly what the response wrote.")),
        dict(item=41, disagreement=("Annotation calls the figures 'not traceable' but does not notice that the response's own share "
             "count is a wrong subtraction (73,671, not 75,671) nor that its product is out by a factor of ten.")),
    ],
    predicate_observations=[
        "Item 67: the annotation states the response 'correctly cites the shipment volume for Copenhagen in 2014 as 448.6 from Document 1' and marks it unsupported solely because the requested ratio was not computed. Attribution is conceded; the derivation is what fails.",
        "Item 81: the annotation concedes 'one could derive this rate by dividing 100 by the useful life in years (20 years)' and marks the item unsupported because the response 'fails to explain that this calculation was performed'. A derivation-transparency requirement, not an attribution one.",
        "Item 215: the response answers the question's own counterfactual correctly (70.6 - 5.5 = 65.1) and is marked unsupported because the actual 2011 ratio rose instead.",
    ],
    per_item=records,
    written=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
)
OUT.write_text(json.dumps(art, indent=1))
print("wrote", OUT, OUT.stat().st_size, "bytes")
print("neg classes:", nc)
print("three-class share", round(three/20,4), "| two-class share", round(two/20,4))
print("pos with derivation", pos_deriv, "/40 | wrong-derivation positives", pos_deriv_wrong)
print("neg with derivation", neg_deriv, "/20")
