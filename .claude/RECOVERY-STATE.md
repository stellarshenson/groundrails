# Recovery State

## CURRENT 2026-08-07 ~23:35 local - DR judge pass launching (supersedes everything below)

Lane campaign CLOSED: **H107 REFUTED** (0.66472), **H108 ADMITTED** (0.70496 > 0.7031, first admitted lane) - both recorded in `docs/experiments/semantic-grounding-experiments.md`. Task #47 completed, #48 (DR close-out) in progress.

**DR judge pass** - `experiments/grounding-semantic/DR_judge.py` (NEW, this session): contrastive judge Qwen3-32B-FP8 over 50,387 usable rows (H112 26,165 + H114 19,616 sentence + 4,606 long-form; H113 excluded). Registered cascade: factual delta → label 0 → regrounding drop → still-entailed veto nli_fwd ≥ 0.8; judge no-delta + bidir NLI ≥ 0.8 → label-1 reclaim. Outputs: `DR_judged.parquet`, `DR_judge_summary.json`, `DR_judge_eyeball.md` (50-pair, main-session grades vs ≥85% bar). Checkpoint `DR_judged.parquet.ckpt` every 4k rows - RESUMES automatically.

**Toolchain repairs needed after container refresh (all applied, may need re-applying after another refresh):**
1. `/dev/shm` 64M → `sudo -n mount -o remount,size=32G /dev/shm`
2. FlashInfer JIT: `/usr/local/cuda/bin/nvcc` gone → `VLLM_USE_FLASHINFER_SAMPLER=0` (native sampler, no JIT)
3. Triton launcher: `Python.h` gone → `sudo -n apt-get update && sudo -n apt-get install -y python3.12-dev`

**Relaunch command (idempotent, resumes from ckpt):**
```
sudo -n mount -o remount,size=32G /dev/shm
cd /home/lab/workspace/private/ai-assistants/groundrails && nohup setsid bash -c 'export VLLM_USE_FLASHINFER_SAMPLER=0; export CUDA_HOME=/home/lab/venvs/vllm/lib/python3.12/site-packages/nvidia/cu13; export PATH=$CUDA_HOME/bin:$PATH; conda run -n cudabuild --no-capture-output /home/lab/venvs/vllm/bin/python experiments/grounding-semantic/DR_judge.py' > logs/DR_judge.log 2>&1 &
```
Success marker: `=== DR JUDGED ===` in `logs/DR_judge.log`.

**After the judge**: main-session eyeball of `DR_judge_eyeball.md` (bar ≥85%, else gpt-oss-120b escalation per standing author trigger) → lane assembly ~26k (H112 55% / H114 25% rebalanced across survivors, up to 20% of H112 share long-form, label-1 reclaim ≤ 4k) → 2 training draws (adapt R10-H108_lane.py pattern, has checkpointing) → adjudication vs 0.7031.

**Awaiting author**: commit approval (many files since `90bd3fb`), journal entry (via /journal:update ONLY), host-side restart supervisor (proposed).

## BRACE 2026-08-07 ~15:50 local - SESSION LIMIT (superseded)

**HORIZON: SESSION DEATH. All compute is detached and survives; a fresh session picks up from here.**

### Live right now (15:50)

- **H108 draw 2, GPU1** - step ~14600/15560 (~94%), detached (`R10_lane_campaign_h108d2.sh`, log `logs/R10_lane_campaign.log`). Training ends ~16:10, then truncated read + windowed read (PRIMARY) + collect. Marker on success: `=== H108 LANE DONE ===`. Result files it will write: `experiments/grounding-semantic/R10-H108_lane_draw2_result.json` + `R10-H108_lane_draw2_windowed_result.json`
- Container restarted 3x today (08:40, 09:20, ~10:05); each kill reverts `/dev/shm` to 64M and kills all jobs. Resume checkpoint saved every 1000 steps at `models/R10-H108-lane-draw2/resume.pt`

### FIRST ACTION for a fresh session

1. Check `=== H108 LANE DONE ===` in `logs/R10_lane_campaign.log` and the two draw2 result JSONs
2. If chain dead mid-draw: `sudo -n mount -o remount,size=32G /dev/shm`, then `cd /home/lab/workspace/private/ai-assistants/groundrails && nohup setsid bash experiments/grounding-semantic/R10_lane_campaign_h108d2.sh >> logs/R10_lane_campaign.log 2>&1 &` (auto-resumes from latest checkpoint; script is idempotent)
3. If done: adjudicate H108 = mean(0.70618, draw2 primary mean) vs bar 0.7031 → ADMITTED if > bar, else REFUTED. Ignore the JSON `verdict` field (legacy rule); the registered bar is the lane mean over 2 draws, PRIMARY windowed read
4. Then record H107 REFUTED (mean 0.66472: draws 0.6704 / 0.65904; finqa destroyed -0.19/-0.25, delucionqa +0.12/+0.11) + the H108 verdict in `docs/experiments/semantic-grounding-experiments.md`
5. Then DR close-out (task #48): judge pass on GPU1 (Qwen3-32B-FP8 vLLM; env `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 VLLM_WSL2_ENABLE_PIN_MEMORY=1`, `/home/lab/venvs/vllm/bin/python`, pattern `R10-H111_judge.py`) over `DR_pilot_raw.parquet` (H112+H114 only, H113 excluded) + `DR_pilot_longform.parquet` → still-entailed veto nli_fwd ≥ 0.8 → 50-pair eyeball ≥85% → lane ~26k → 2 training draws → adjudication

### Awaiting the author (unchanged)

- Commit approval: DR-2 records, checkpointing patch (both `R10-H10{7,8}_lane.py`), lane results, `R10_lane_campaign_h108d2.sh`, this file
- Journal entry for DR-2 pilot + Round 10 campaign (via /journal:update ONLY)
- Host-side supervisor for container restarts (proposed, not authorized)

## RESUMED 2026-08-07 ~08:58 local

Container was RESTARTED (not recreated - `/.dockerenv` still Aug 6 09:09) around 08:40: `/dev/shm` reverted to 64 MB and every process was killed. H108 draw 2 died at step ~6200; its resume point at step 6000 survived.

- **Recovery executed** - shm remounted 32 GB, draw relaunched from `models/R10-H108-lane-draw2/resume.pt` at step 6001/15560 via the new tail script `experiments/grounding-semantic/R10_lane_campaign_h108d2.sh` (H108 draw 2 train + both reads + collect, appending to `logs/R10_lane_campaign.log`)
- **Relaunch after any further kill** - FIRST `sudo -n mount -o remount,size=32G /dev/shm`, THEN `cd /home/lab/workspace/private/ai-assistants/groundrails && nohup setsid bash experiments/grounding-semantic/R10_lane_campaign_h108d2.sh >> logs/R10_lane_campaign.log 2>&1 &` (the draw resumes from its latest checkpoint automatically)
- **H107 REFUTED** - lane mean 0.66472 (draw 1 0.6704, draw 2 0.65904) vs the 0.7031 clean bar; both draws destroyed finqa (-0.1921, -0.2469) while lifting delucionqa (+0.1206, +0.1080). Not yet written into the canonical log
- **H108 draw 1 = 0.70618, ABOVE the bar** by +0.0031; gold 0.8653 / gold_full 0.8589 (clean baseline 0.8514), finqa +0.0561, no subset collapse. Draw 2 must read >= 0.7000 for the lane mean to clear
- **Ignore the `verdict` field in the lane result JSONs** - it is a legacy per-subset rule; the registered admission bar is the lane mean over 2 draws on the PRIMARY windowed read
- **Next after `=== H108 LANE DONE ===`** - record both lane verdicts in `docs/experiments/semantic-grounding-experiments.md`, then the DR close-out (task #48 below)

## RESUMED 2026-08-06 ~00:05 local

Container was recycled at 23:37 (host never rebooted, but `/dev/shm` reverted to Docker's 64 MB default and both GPU jobs died). Resume executed: shm remounted 32 GB, both jobs relaunched, long-form lane repaired and shipped.

- **Long-form top-up CLOSED** - `DR_pilot_longform.parquet` 5,432 spans / 3,379 docs, 100% char-exact, debris 7.5%, NLI populated; recorded in the canonical DR doc. Quarantined evidence: `DR_pilot_longform.attempt1.parquet`, `DR_pilot_longform.FAILED.parquet`
- **Lane campaign attempt 3 RUNNING** - GPU1, H107 draw 1, restarted 23:41 from step 0; relaunch command unchanged below (ALWAYS remount shm first)
- **GPU0 free** - DR generation phase complete: `DR_pilot_raw.parquet` (61,100 sentence-level) + `DR_pilot_longform.parquet` (5,432 long-form) both final and awaiting the judge pass
- **Next**: judge pass on GPU1 after `=== H108 LANE DONE ===` (task #48 detail below), lane adjudications as markers land, journal entry for DR-2 pending author's word

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
