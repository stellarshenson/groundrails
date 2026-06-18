# Logs

Background job logs for groundrails.

- `full-gold-eval.log` - groundrails grounder run over the full gold-v2 dataset (equivalence/metrics check; aggregate metrics only, no client text)
- `joint-wirings-benchmark.log` - lexical+semantic wiring benchmark (experiments/grounding-semantic/joint_wirings.py); live lexical pass + cached cascade scores; aggregate metrics only, no client text
- `semantic-live-smoke.log` - live `--semantic` cascade smoke (pulls the int8 IRs, scores a supported + a fabricated public claim); verifies the serving path end-to-end
