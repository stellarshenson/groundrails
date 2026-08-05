# Recovery State

## BRACE 2026-08-05 ~22:05 local

**HORIZON: SERVER RESTART assumed (worst case; unstated at invocation). Every running job WILL be killed by a host reboot - relaunch all from the commands below. If it turns out SESSION-ONLY, all three jobs are detached (PPID 1) and survive - just reattach via their logs.**

### Running at brace time (do NOT assume alive after restart)

1. **Lane training campaign, GPU1** - pid 87024, `bash experiments/grounding-semantic/R10_lane_campaign.sh`, attempt 2, H107 draw 1 at step 1400/16028 (~9%), log `logs/R10_lane_campaign.log` (attempt 1 died on /dev/shm exhaustion → hung; its log is `logs/R10_lane_campaign.attempt1.log`)
   - No mid-draw checkpoint exists - a restart loses the current draw and restarts the campaign from H107 draw 1
   - **Relaunch**: FIRST `sudo -n mount -o remount,size=32G /dev/shm` (container default 64MB kills DataLoader workers - MANDATORY), THEN `cd /home/lab/workspace/private/ai-assistants/groundrails && nohup setsid bash experiments/grounding-semantic/R10_lane_campaign.sh > logs/R10_lane_campaign.log 2>&1 &` (rotate the old log first)
   - Chain: H107 draw1→reads→draw2→reads→marker `=== H107 LANE DONE ===` → H108 same → `=== H108 LANE DONE ===`; per-draw results `experiments/grounding-semantic/R10-H10{7,8}_lane_draw{1,2}_result.json`, models under `models/`
2. **Long-form top-up RERUN, GPU0** - pids ~135116/136086, `DR_pilot_longform_topup.py` with the dedup-BEFORE-edit fix, log `logs/DR_pilot_longform.log`, marker `=== DR LONGFORM TOPUP DONE ===`, ~17 min runtime
   - **Relaunch**: `cd /home/lab/workspace/private/ai-assistants/groundrails && nohup setsid bash -c 'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 uv run python experiments/grounding-semantic/DR_pilot_longform_topup.py' > logs/DR_pilot_longform.log 2>&1 &` (verify the exact env/invocation against the top of the previous log or Executor A's `DR_pilot_longform_RELAUNCH.md` if present)
   - **Acceptance**: `DR_pilot_longform_summary.json` recheck MUST read `docs_char_exact_rate == 1.0`; if not, quarantine and do not merge
3. **Executor A subagent** - notified to brace; may have written `experiments/grounding-semantic/DR_pilot_longform_RELAUNCH.md`

Session-owned watchers/monitors are dead after any restart - re-arm marker watchers on both logs (patterns: markers above + `^Traceback \(most recent` + `FAILED ===`; do NOT grep case-insensitive "failed" - the sub-gate JSON echo contains "failed-span" and false-alarms).

### Valid on disk (headline numbers)

- `experiments/grounding-semantic/DR_pilot_raw.parquet` - **61,100 rows FINAL** sentence-level pilot: H112 31,000 (debris 7.6% PASS vs 12.4%), H114 22,998 (11.9% PASS vs 28.6%), H113 7,102 (**2.04% - KILL BAR FIRED, engine DROPPED from lane**, rows stay in parquet but are excluded at lane assembly); nli_fwd/nli_bwd populated; judge pass NOT run
- `experiments/grounding-semantic/DR_H116_subgate_result.json` - verdict SURVIVES (main-session adjudication over executor KILL; both readings preserved; splice 129/129, all-spans-degen 5.43%)
- `experiments/grounding-semantic/DR_pilot_gen_summary.json` - per-engine bars as above
- Lane parquets staged: `data/external/datasets/R10-H107_pairs.parquet` (83,672), `experiments/grounding-semantic/R10-H108_pairs.parquet` (61,184), `R10-H111_pairs_final.parquet` (26,142, fallback lane)
- Canonical docs CURRENT through: DR-2 registration, H116 sub-gate result + adjudication, DR-2 generation result + H113 drop (`docs/experiments/semantic-dataset-enhancements.md`); training-campaign authorization + H103 PARKED (`docs/experiments/semantic-grounding-experiments.md`)

### Invalid / quarantined

- `experiments/grounding-semantic/DR_pilot_longform.attempt1.parquet` (+ .ckpt, + `DR_pilot_longform_summary.attempt1.json` if present) - **DEFECTIVE, do not merge**: cross-lane dedup dropped rows AFTER edits were applied, 546/3,934 docs carry unledgered corruptions (docs_char_exact 86.1%); kept as evidence only

### Pending decisions / recordings (next session)

- DR-2 top-up result block → canonical DR doc once rerun lands clean (defect + fix + rerun numbers, one block)
- Journal entry for DR-2 pilot (via /journal:update ONLY)
- H107/H108 lane draw adjudications vs the 0.7031 clean mean (PRIMARY windowed read) as markers land
- **DR lane close-out (task #48)**: after `=== H108 LANE DONE ===` frees GPU1 → judge pass (Qwen3-32B-FP8 vLLM on GPU1; env block: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 VLLM_WSL2_ENABLE_PIN_MEMORY=1, /home/lab/venvs/vllm/bin/python; pattern in `R10-H111_judge.py`) over DR_pilot_raw (H112+H114 only, H113 excluded) + clean long-form parquet → still-entailed veto nli_fwd ≥ 0.8 on negatives → 50-pair stratified eyeball (bar ≥85%, else gpt-oss-120b escalation per author trigger) → lane parquet ~26k → DR training draws x2 → adjudication
- Commit of post-brace results needs fresh author approval (this brace commit is the authorized exception)

### FIRST ACTION for a fresh session

Read this file, then: (1) `df -h /dev/shm` - if 64M, remount 32G (command above); (2) check `nvidia-smi` + the two logs for survivors vs casualties; (3) relaunch dead jobs with the commands above (lane campaign restarts from draw 1 - accept the loss); (4) re-arm marker watchers; (5) resume the pending-decisions list. Campaign context: three lanes race the 0.7031 clean baseline, order H107 → H108 → DR (replaces H111); admission = lane mean over 2 draws blind > 0.7031, PRIMARY windowed decomposed-min read, frozen R8-H77 gate. TRAINING OF NEW ROUNDS beyond this authorized campaign, git ops, and journal writes all need the author's word.
