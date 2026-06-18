# Logs

Background job logs for groundrails.

- `full-gold-eval.log` - groundrails grounder run over the full gold-v2 dataset (equivalence/metrics check; aggregate metrics only, no client text)
- `joint-wirings-benchmark.log` - lexical+semantic wiring benchmark (experiments/grounding-semantic/joint_wirings.py); live lexical pass + cached cascade scores; aggregate metrics only, no client text
- `semantic-live-smoke.log` - live `--semantic` cascade smoke (pulls the int8 IRs, scores a supported + a fabricated public claim); verifies the serving path end-to-end
- `cascade-score-gold-v3.log` - OV int8 cascade scored over all 7,976 gold v3 pairs (golden + synthetic aug) -> `data/processed/golden_v3_cascade_scores.parquet`; aggregate progress only, no client text
- `lex-pass-gold-v3.log` - lexical manifold (effort=high) pass over gold v3 for `lex_p` -> `data/processed/golden_v3_lex.parquet`; aggregate progress only, no client text
