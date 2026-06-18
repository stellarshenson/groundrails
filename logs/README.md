# Logs

Background job logs for groundrails.

- `full-gold-eval.log` - groundrails grounder run over the full gold-v2 dataset (equivalence/metrics check; aggregate metrics only, no client text)
- `joint-wirings-benchmark.log` - lexical+semantic wiring benchmark (experiments/grounding-semantic/joint_wirings.py); live lexical pass + cached cascade scores; aggregate metrics only, no client text
- `semantic-live-smoke.log` - live `--semantic` cascade smoke (pulls the int8 IRs, scores a supported + a fabricated public claim); verifies the serving path end-to-end
- `cascade-score-gold-v3.log` - OV int8 cascade scored over all 7,976 gold v3 pairs (golden + synthetic aug) -> `data/processed/golden_v3_cascade_scores.parquet`; aggregate progress only, no client text
- `lex-pass-gold-v3.log` - lexical manifold (effort=high) pass over gold v3 for `lex_p` -> `data/processed/golden_v3_lex.parquet`; aggregate progress only, no client text
- `vitaminc-combined.log` - combined grounder (lexical + cascade) scored over the 800-row VitaminC slice of gold v4 -> `reports/grounding_vitaminc_combined.md`; aggregate metrics only, public benchmark
- `joint-premise-score.log` - Round 4 joint-premise (SummaC) NLI recompute over the 3,215 cascade-fired gold v3 rows -> `data/processed/golden_v3_joint_premise.parquet`; aggregate progress only, no client text
