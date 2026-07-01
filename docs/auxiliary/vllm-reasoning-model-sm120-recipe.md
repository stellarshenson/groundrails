# Recipe - running a reasoning model on vLLM (Blackwell sm_120, WSL2)

Reproducible recipe to serve a quantized reasoning model (here DeepSeek-R1-Distill-Llama-70B at INT4) with vLLM continuous batching on a single RTX PRO 6000 Blackwell (compute capability 12.0, sm_120) running under WSL2, with an FP8 KV cache for maximum concurrency on long prompts. The model emits a `<think>...</think>` chain before its answer, which the caller strips before parsing.

Five platform/packaging issues sit between a fresh install and a working batched run on this stack. None is a model fault; each is a one-line fix. They are listed in the order vLLM hits them so the recipe can be followed top to bottom.

## Why this setup

- **Batched, not single-stream** - vLLM decodes a batched GEMM on the FP8/INT4 tensor cores that reads the weights once per batch, so concurrency scales; llama.cpp's K-quant decode is mat-vec and stays flat under batching
- **INT4 weights (w4a16), not FP8** - on one 96GB card the binding constraint for long (15K-token) prompts is KV room, not GEMM speed; 4-bit weights (~37GB) free ~3x the KV of FP8 weights (~70GB), which is what raises concurrency
- **FP8 KV cache** - halves KV per token, roughly doubling how many long prompts fit at once; break-even context is ~7k tokens, so it is a net speedup for 15K prompts, at ~0.7-2 point reasoning-accuracy cost
- **Measured result** - w4a16 + FP8 KV on this card reports a GPU KV cache of 319,328 tokens and a maximum concurrency of 15.59x at 20,480 tokens per request

> The w4a16 config below is a valid working recipe, but it is **not** the fastest scheme for prefill-bound work. The measured prefill win is **NVFP4** - see [The prefill win - native FP4](#the-prefill-win---native-fp4-nvfp4-beats-weight-only-quant). The five platform fixes apply to any scheme.

## The prefill win - native FP4 (NVFP4) beats weight-only quant

The judge workload is prefill-bound (≈14K-token evidence : ≈350 output, 40:1), so prefill GEMM throughput is the metric. Measured on this card, same LONG ≈14K-prompt workload, 16 concurrent sequences, FP8 KV.

- **NVFP4 wins decisively** - 2094 tok/s prefill (peak instantaneous 2624), 3.2x over INT4-Marlin, 2.2x over FP8, at ~40GB footprint (half of FP8's ~71GB)
- **The lever is activation precision, not weight precision** - a 4-bit *weight-only* scheme (W4A16) speeds up bandwidth-bound **decode** by storing fewer bytes, but in compute-bound **prefill** it must dequantize INT4→FP16 and run the GEMM in **FP16** on the tensor cores - zero compute benefit, plus a dequant tax; the GEMM runs at the **activation** dtype, so only quantizing activations too (FP4/FP8/INT8) moves prefill
- **That is why the ranking inverts** - INT4-Marlin (654) is *slower* than FP8 (941): FP8 runs the GEMM in FP8 (2x FP16 tensor-core throughput), INT4-W4A16 runs it in FP16; NVFP4 runs it in FP4 (native FP4 tensor cores, ~2x FP8 again)
- **The kernel names are the proof** - the INT4 run logs `Using MarlinLinearKernel for CompressedTensorsWNA16` (`WNA16` = Weights N-bit, **Activations 16-bit**) with `dtype=torch.bfloat16`; a tensor core multiplies two same-precision inputs, so Marlin **up-converts the 4-bit weights to bf16** and runs a bf16 matmul plus a dequant tax - the 4-bit is storage only. The NVFP4 run logs `Using CutlassNvFp4LinearKernel for NVFP4 GEMM` with `fuse_act_quant: True`, feeding **FP4 x FP4** into the native FP4 tensor cores - no dequant. Same 4 bits of weight, different compute path
- **What "went wrong" was a category error** - W4A16 is a *decode* optimization (decode is memory-bandwidth-bound: 4-bit storage = 4x fewer weight bytes to read per token, the tiny matmul's 16-bit compute is free); the judge is *prefill-bound* (compute-bound), where the matmul is the wall and W4A16 does it in 16-bit. "INT4" names the storage, not the math

| Scheme | GEMM math | Prefill tok/s | VRAM (70B) | Notes |
|--------|-----------|---------------|------------|-------|
| INT4 W4A16, default Marlin | FP16 (dequant) | 654 | ~37GB | weight-only; the floor |
| FP8 `-dynamic` per-tensor | FP8 | 941 | ~71GB | Cutlass path |
| INT4 W4A16 + FP8-Marlin | FP8 | blocked | ~37GB | needs non-act-order checkpoint, see below |
| NVFP4 (`modelopt_fp4`), pinned 0.24.0 venv | FP4 | 2094 | ~40GB | native FP4 tensor cores, no dequant; peak 2624 |
| **NVFP4, cu130-nightly Docker** | **FP4** | **2952** | **~40GB** | **+41% from fresher kernels (`fuse_act_quant`); peak 4750** |

**NVFP4 config** - runs in the pinned vLLM 0.24.0 venv, no source build or Docker needed (`modelopt_fp4` is a registered quant method):

```python
# checkpoint: nvidia/Llama-3.3-70B-Instruct-NVFP4 (ModelOpt NVFP4, group_size 16, kv FP8)
os.environ["VLLM_NVFP4_GEMM_BACKEND"] = "cutlass"   # FP4 GEMM backend
llm = LLM(
    model="nvidia/Llama-3.3-70B-Instruct-NVFP4",
    quantization="modelopt_fp4",                    # force the native FP4 path
    max_model_len=18432, kv_cache_dtype="fp8",
    gpu_memory_utilization=0.90, max_num_seqs=16,
    enable_prefix_caching=True, enable_chunked_prefill=True,
    max_num_batched_tokens=16384, dtype="auto",
)
```

**The INT4+FP8-Marlin path (W4A8) is real but needs the right checkpoint** - the rtx6kpro Kimi recipe runs INT4 weights through an FP8 GEMM (`VLLM_TEST_FORCE_FP8_MARLIN=1`, `VLLM_MARLIN_INPUT_DTYPE=fp8`, `VLLM_MARLIN_USE_ATOMIC_ADD=1`), which *would* be fast; it failed here because the cached RedHat `DeepSeek-R1-70B-w4a16` was quantized with act-order, and the FP8-Marlin repack rejects it: `gptq_marlin_repack: Unsupported repack config: num_bits=4, has_perm=1, is_a_8bit=1`. A non-act-order INT4 checkpoint (`actorder="weight"`) would unblock it, but NVFP4 already beats what an FP8 GEMM can reach.

### The nightly Docker image lifts NVFP4 a further +41%

The pinned 0.24.0 venv is not the ceiling - the `vllm/vllm-openai:cu130-nightly` image runs the same NVFP4 checkpoint at **2952 tok/s** sustained (peak instantaneous **4750 tok/s**), +41% over the venv's 2094.

- **The lever is `fuse_act_quant` + sm120f-compiled FP4 kernels** - the nightly engine sets `compilation_config.pass_config.fuse_act_quant: True` (the 0.24.0 release has it `False`), fusing the activation-quantization into the FP4 GEMM; and its FlashInfer/CUTLASS kernels are compiled for the **`sm120f`** family-conditional target, which unlocks the native `cvt.rn.satfinite.e2m1x2.f32` FP32→FP4 conversion PTX instruction (rtx6kpro: this was the exact reason NVFP4 was "consistently slower than INT4 AWQ" until FlashInfer PRs #2650/#2716 added the sm120f path). The pinned 0.24.0 wheel runs NVFP4 but without the full sm120f FP4 path
- **Most of the gain is a venv config flag, no Docker** - `fuse_act_quant` is a `compilation_config` pass that *exists in the pinned 0.24.0* but defaults off; turning it on recovers +27% (2094 → 2669 tok/s, peak 3909) with zero wheel swap:

```python
llm = LLM(
    model="nvidia/Llama-3.3-70B-Instruct-NVFP4", quantization="modelopt_fp4",
    max_model_len=18432, kv_cache_dtype="fp8", gpu_memory_utilization=0.90,
    max_num_seqs=16, enable_prefix_caching=True, enable_chunked_prefill=True,
    max_num_batched_tokens=16384, dtype="auto",
    compilation_config={"pass_config": {"fuse_act_quant": True}},   # the +27% lever
)
```

- **The residual 2669 → 2952 gap needs a source build, not a pip upgrade** - the gap is the nightly's sm120f-compiled cutlass FP4 kernels, a compile-time difference the flag cannot add. It is *not* pip-installable: the public vLLM nightly index (`wheels.vllm.ai/nightly`) only carries `vllm==0.23.1rc1.dev672`, **older** than the pinned 0.24.0 (and it pulls a conflicting `pynvvideocodec==2.1.0`); the Docker's own `0.19.2...+gfe9c3d6c5` build is lower-versioned still and its wheel is stripped from the final image (not extractable via `docker cp`). The only venv path to the last +11% is a **source build** (rtx6kpro `build_vllm_venv.sh`, `ENABLE_SM120=1` + `flash_attn 2.8.3` for SM120, ~30-60 min). For most work it is not worth it - the venv already ships 2669 tok/s with no Docker, and a w4a16 accuracy-first judge does not use the FP4 path at all
- **Caveat: nightly NVFP4 cache corruption is a known issue** - rtx6kpro reports "some vLLM nightly builds exhibit NVFP4 cache corruption"; before trusting a nightly for real labeling, validate *output correctness* (e.g. JSON validity + spot-check verdicts), not just throughput
- **Version is dev-scheme, not lower** - the nightly reports `v0.19.2rc1.dev134+gfe9c3d6c5`; the string is below `0.24.0` but the build is newer (main-branch dev versioning), with the more recent FP4 kernels
- **Peak ≈ 4750 tok/s** approaches the card's practical FP4 prefill ceiling; the sustained 2952 is lower because the batch has ramp/drain and decode interleaves (GEN=192) - a pure-prefill batch sits closer to peak
- **Load is fast (~69s)** via the image's torch.compile AOT cache; `Maximum concurrency 15.19x at 18,432 tokens`
- **Concurrency and chunk size are inert (refuted)** - a tuned run at `max_model_len=15360` (concurrency ceiling 17.65x), `max_num_seqs=18`, `max_num_batched_tokens=24576` held sustained prefill at 2911 tok/s (vs 2952), zero preemptions; NVFP4 prefill is GEMM-kernel-bound, not concurrency- or chunk-bound, same as the INT4 sweep
- **2952 sustained vs 4750 peak is the inference engine and kernel, not 300W power throttling** - the proof is that pure *software* changes raise the *sustained* floor: `fuse_act_quant` lifted it 2094 → 2669 (+27%) and the sm120f-compiled cutlass kernels a further +11% to 2952, neither of which a power wall would permit (power clamps the ceiling regardless of which kernel runs). The sustained-below-peak gap is the scheduler interleaving decode steps plus kernel-launch gaps starving the FP4 GEMM between chunks, not the card clamping clocks - a pure-prefill run (gen=1, no decode) still oscillates 1700↔3947 tok/s (avg ~2542, only 2 of 12 log intervals ≥3000) because the engine cannot keep the tensor cores continuously fed, and the *same* card hits a higher peak with better kernels (4750 Docker vs 3947 venv) - headroom the venv kernel left unused. The 300W cap is real (hard max 325W = +8%; `nvidia-smi -pl` returns `Insufficient Permissions` without host root) and may bind only at the very top (4750+ peak), but it is not the constraint across the 2094-2952 sustained band - the kernel is. So **~2670 (venv+flag) / 2952 (nightly) is the engine-bound sustained prefill floor, liftable by a better kernel, not a fixed power ceiling** - which is exactly why the source-build sm120f kernels recover the last +11%. The advertised ~5000 tok/s is a peak/burst figure (matched: peak 4750)

**Reproducing it in a sibling-daemon container (the gotcha)** - if Claude/the shell runs inside a container talking to a *sibling* docker daemon (not true DinD), bind mounts (`-v /host/path:/c`) resolve on the **daemon's** filesystem, not the caller's, so `-v ~/.cache/huggingface:...` mounts an empty dir and the model is not found (vLLM exits before printing anything). Two fixes are required together:

- **Stage the model into a named volume** via `docker cp` (the docker CLI reads the caller-side path and streams it across the socket), then mount the volume and point `HF_HOME` at it:

```bash
docker volume create hf-nvfp4
CID=$(docker run -d -v hf-nvfp4:/vol --entrypoint sleep vllm/vllm-openai:cu130-nightly 7200)
docker exec "$CID" mkdir -p /vol/hub
docker cp ~/.cache/huggingface/hub/models--nvidia--Llama-3.3-70B-Instruct-NVFP4 "$CID:/vol/hub/"
docker cp bench.py "$CID:/vol/bench.py"; docker stop "$CID"; docker rm "$CID"
```

- **Capture output with `docker logs`, not the attached stream** - without a TTY the daemon does not forward the container's stdout/stderr to the client (runs look silent, exit code preserved); run detached and read the daemon's log driver afterward:

```bash
RID=$(docker run -d --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --mount type=tmpfs,destination=/usr/local/cuda-13.0/compat \
  -e CUDA_VISIBLE_DEVICES=0 -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e VLLM_NVFP4_GEMM_BACKEND=cutlass -e HF_HUB_OFFLINE=1 -e HF_HOME=/vol \
  -v hf-nvfp4:/vol --entrypoint python3 vllm/vllm-openai:cu130-nightly \
  /vol/bench.py 16384 fp8 16 16 192)
docker wait "$RID"; docker logs "$RID" > out.log 2>&1; docker rm "$RID"
```

The `--mount type=tmpfs,destination=/usr/local/cuda-13.0/compat` empties the container's CUDA-compat dir so it uses the host driver (host CUDA 13.2 / container 13.0), per the rtx6kpro note.

## Using a reasoning model for structured (JSON) output - validation gotchas

Serving R1-Distill as a strict-JSON judge surfaced four failure modes that silently corrupt output; each was caught only by validating the *parsed* result, not just that the engine ran. Measured on the DeepSeek-R1-Distill-Llama-70B judge against a private gold set.

- **NVFP4 PTQ degrades reasoning quality, not just speed** - the same R1 as a judge scored single-judge accuracy 0.748 at INT4 `w4a16` vs **0.571** at NVFP4 (PiehSoft), dual-judge Cohen κ 0.873 vs 0.689; one NVFP4 build (Kaleto) was unusable (0% parseable). Use weight-only INT4/INT8 for accuracy-critical structured tasks; reserve NVFP4 for raw-throughput work where a small quality loss is acceptable - it is a *throughput* format, not a drop-in for a precise judge
- **Uncalibrated checkpoints must use bf16 KV** - a checkpoint whose `hf_quant_config.json` has `kv_cache_quant_algo: null` (no calibrated k/v scales) produces **incoherent, hallucinated output under `kv_cache_dtype="fp8"`** (uncalibrated fp8 attention, scale=1.0); `kv_cache_dtype="auto"` (bf16) restores coherence at ~2x the KV memory (so ~half the concurrency)
- **The R1-Distill tokenizer leaks byte-BPE markers into vLLM detok** - generated `.text` contains `Ġ` (space) and `Ċ` (newline) literally; compact JSON survives but spaced JSON (`{"idx": 1}`) becomes invalid - normalize `Ġ → " "`, `Ċ → "\n"` before `json.loads`. Gemma's tokenizer detokenizes clean, so it is tokenizer-specific
- **vLLM v1 in-process teardown does not free VRAM** - loading a second model in the same process after `destroy_model_parallel()` OOMs (the first model's ~86GB persists); run each model in its **own process** (the OS frees the card on exit) rather than relying on in-process teardown for sequential multi-model runs
- **The cache must key on the model** - a per-(trace, chunk) result cache silently reuses a *prior model's* verdicts on re-run; delete or namespace the cache when the judge model changes

## Environment (verified)

- GPU - RTX PRO 6000 Blackwell Max-Q, sm_120 (compute 12.0), 96 GB; driver 596.72 (CUDA 13.2 capable)
- Host - WSL2, kernel `6.18.33.2-microsoft-standard-WSL2`, Python 3.12
- vLLM 0.24.0, torch 2.11.0+cu130, FlashInfer 0.6.12 (all in an isolated venv)
- CUDA toolkit - the pip `nvidia-*-cu13` packages under the venv provide nvcc 13.2.78, headers, and libraries; the system has no `/usr/local/cuda`

Toolkit location (adjust the venv path to yours):

```
VENV=/home/lab/.venvs/groundrails-judge
CUDA=$VENV/lib/python3.12/site-packages/nvidia/cu13   # bin/nvcc, include/, lib/
```

## The five fixes

### 1. compressed-tensors rejects a dense sparsity_config

vLLM 0.24 removed sparsity support and raises `DeprecationWarning: Sparsity support has been removed from compressed-tensors` on any checkpoint whose `quantization_config.sparsity_config` has a non-empty `targets` list - even when `format: "dense"` (no actual sparsity). Many RedHat w4a16 / FP8 checkpoints ship exactly such a dense block.

Strip the `sparsity_config` from the cached `config.json`. It is metadata only; the weights are dense pack-quantized INT4, so removing it changes nothing:

```python
import json, os, glob
snap = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--<ORG>--<MODEL>/snapshots/*/config.json"))[0]
cfg = json.load(open(snap))
cfg.get("quantization_config", {}).pop("sparsity_config", None)
if os.path.islink(snap):            # replace the HF symlink with a real file; blob left intact
    os.unlink(snap)
json.dump(cfg, open(snap, "w"), indent=2)
```

### 2. multiprocessing spawn and clean teardown

The vLLM v1 engine core defaults to a spawned subprocess. A plain script that calls `LLM()` at module top level re-spawns infinitely (`An attempt has been made to start a new process before the current process has finished its bootstrapping phase`). Running the engine core in-process avoids the spawn entirely and, more importantly, gives deterministic VRAM release when loading two models in sequence:

```
VLLM_ENABLE_V1_MULTIPROCESSING=0
```

### 3. WSL2 disables pinned memory, which the V2 model runner requires

vLLM turns pinned memory off on WSL by default. The new GPU model runner allocates a UVA (unified-addressing) buffer for `all_token_ids` and hard-raises `RuntimeError: UVA is not available` when pin memory is off. Current WSL2 + recent drivers handle pinned memory fine; re-enable it:

```
VLLM_WSL2_ENABLE_PIN_MEMORY=1
```

### 4. FlashInfer JIT needs nvcc on PATH

The FP8 KV cache runs through FlashInfer, which JIT-compiles its `e4m3` attention kernel at first use via ninja + nvcc. With no `/usr/local/cuda`, ninja calls a missing `nvcc` and fails with exit 127. Point the JIT at the pip cu13 toolkit before importing vLLM:

```python
import os, sysconfig
cu13 = os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "cu13")
os.environ["CUDA_HOME"] = os.environ["CUDA_PATH"] = cu13
os.environ["PATH"] = f"{cu13}/bin:" + os.environ["PATH"]
os.environ["LD_LIBRARY_PATH"] = f"{cu13}/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ.setdefault("NVCC_APPEND_FLAGS", "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK")
```

`NVCC_APPEND_FLAGS` pre-empts the CCCL version-skew check that the pip cu13 toolkit can trip; harmless when it does not fire.

### 5. FlashInfer links against lib64, the pip toolkit ships lib

The JIT link step passes `-L$CUDA/lib64 -lcudart`, but the pip cu13 package puts its libraries in `lib/`, so `ld: cannot find -lcudart`. Create the unversioned `.so` symlinks and a `lib64 → lib` alias once:

```bash
for base in cudart cublas cublasLt nvrtc; do
  [ -e "$CUDA/lib/lib$base.so" ] || \
    ln -sf "$(ls "$CUDA"/lib/lib$base.so.* | head -1 | xargs basename)" "$CUDA/lib/lib$base.so"
done
[ -e "$CUDA/lib64" ] || ln -s lib "$CUDA/lib64"
```

The CUDA driver stub `-lcuda` resolves from the system path (`/lib/x86_64-linux-gnu/libcuda.so`), so no stub is needed.

## The environment block (copy-paste)

Set all of this before importing vLLM:

```python
import os, sysconfig
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"                 # the 96GB Blackwell
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"      # FP8 KV path on Blackwell
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"       # in-process: no spawn, clean teardown
os.environ["VLLM_WSL2_ENABLE_PIN_MEMORY"] = "1"          # WSL2: V2 runner needs UVA/pin memory
cu13 = os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "cu13")
os.environ["CUDA_HOME"] = os.environ["CUDA_PATH"] = cu13
os.environ["PATH"] = f"{cu13}/bin:" + os.environ["PATH"]
os.environ["LD_LIBRARY_PATH"] = f"{cu13}/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ.setdefault("NVCC_APPEND_FLAGS", "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK")
os.environ.setdefault("CUDA_CACHE_MAXSIZE", "4294967296")
```

## Load and serve

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="RedHatAI/DeepSeek-R1-Distill-Llama-70B-quantized.w4a16",
    max_model_len=18432,            # right-size to 15K evidence + claims + reasoning; smaller ctx => higher KV concurrency
    kv_cache_dtype="fp8",           # halve KV -> ~2x concurrency on long prompts
    gpu_memory_utilization=0.92,
    max_num_seqs=16,                # <= the reported max concurrency (17.37x at 18432) to avoid preemption; not the long-prompt throughput lever, see Tuning
    enable_prefix_caching=True,     # share the fixed instruction prefix across calls
    enable_chunked_prefill=True,    # interleave the long prefills with decode
    max_num_batched_tokens=8192,    # chunked-prefill chunk size; prefill-critical, larger speeds long-prompt prefill (under tuning, see Tuning)
    dtype="auto",
)
```

Reasoning-model specifics:

- **Strip the think block** - the model emits `<think>...</think>` then the answer; drop everything up to and including the last `</think>` before parsing
- **Budget the generation** - `max_tokens` must cover the reasoning chain plus the answer (here ~350 reasoning + ~283 JSON, set to 1280 for headroom)
- **Greedy** - `SamplingParams(temperature=0.0, seed=0)` for deterministic labels

## Tuning - the long-prompt bottleneck is prefill

On 15K-token grounding prompts the dominant cost is prefill of the input, not decode and not KV preemption. Measured on this card, isolating one variable.

- **Two batches, identical except input length** - 24 short prompts (gen 256) ran at 257 agg tok/s; 24 × 15K-token prompts (same gen 256) ran at 12 agg tok/s; the 22x gap is entirely the cost of prefilling the long inputs (~360K prefill tokens at ~710 tok/s)
- **Preemption was ruled out** - `max_num_seqs=12` is below the 17.37x ceiling, so zero sequences were preempted; capping concurrency did not move long-prompt throughput, so over-subscription is not the cause here
- **Cap `max_num_seqs` anyway** - exceeding the load-time `Maximum concurrency for N tokens per request: X.XXx` (15.59x at 20,480 ctx, 17.37x at 18,432 ctx) does cause preemption + recompute; the rtx6kpro guide runs `max_num_seqs=16` on this card; necessary, but not the throughput lever for long prompts
- **Config levers do not move it** - a sweep held prefill at ~570-650 tok/s across `max_num_batched_tokens` 4096 vs 16384 (chunked-prefill granularity) and `kv_cache_dtype` fp8 vs auto (FlashInfer e4m3 vs prebuilt prefill kernel); none of these is the bottleneck
- **Activation precision is the real lever, not weight precision** - Marlin dequantizes INT4 to FP16 before the GEMM (654 tok/s); FP8 runs the GEMM in FP8 (941); NVFP4 runs it in FP4 (2094) - the prefill GEMM executes at the *activation* dtype, so the speedup comes from quantizing activations, not storage; see [The prefill win - native FP4](#the-prefill-win---native-fp4-nvfp4-beats-weight-only-quant) for the full table
- **Verify the kernels before blaming the build** - a common claim is that prebuilt wheels lack sm_120 kernels and need a source build; check first with `cuobjdump --list-elf $(python -c 'import vllm,glob,os;print(glob.glob(os.path.dirname(vllm.__file__)+"/_C*.so")[0])') | grep -oE 'sm_[0-9]+' | sort | uniq -c`; the vLLM 0.24 wheel here ships 76 native sm_120 SASS kernels, so a rebuild changes nothing
- **The card is a Max-Q (300W cap), but low MFU is a kernel problem, not a power ceiling** - at ~941 tok/s FP8 prefill the GPU runs only ~27% MFU, far from saturating the tensor cores; a power wall shows up as high MFU pinned at the cap, not 27%, so the bottleneck at this rate is kernel/engine efficiency, not the power budget. The 300W cap (hard max 325W, +8%) only bounds the topmost peak; raising 300→325W buys ~8% there but does nothing for the sustained kernel-bound floor
- **Model size is a secondary lever (after quant scheme)** - within one scheme a smaller model prefills faster and frees KV (a 70B FP8 at ~72GB caps concurrency ~4.4x; a 27-32B FP8 frees ~55GB → higher concurrency); but the scheme dominates - NVFP4 at the full 70B (2094) beats FP8 at 70B (941) 2.2x, so fix the scheme first, then size down only if quality allows
- **Prefix caching only helps multi-chunk traces** - if the long evidence is a shared prefix reused across many short queries it pays off (once-per-document prefill); measured here at ~1.04 chunks/trace (97% single-chunk), so it saved ~4% - check your claims/document ratio before wiring it
- **Watch the preemption counter** - load with `disable_log_stats=False`; a non-zero `Preempted` count means `max_num_seqs` is too high, a separate problem from the prefill cost above

### Cross-checked against the rtx6kpro field notes

The community knowledge base for this card (see Sources) frames the likely cause - 4-bit on Blackwell needs the right kernels, and a hand-assembled venv is the slow path.

- **4-bit underperforms unless kernels are compiled for sm120f** - the rtx6kpro notes record 4-bit (NVFP4/AWQ) running "consistently slower" until the GEMM/attention kernels were built for the `sm120f` family-conditional path (FlashInfer PRs #2650/#2716); their working stacks are the prebuilt `vllm/vllm-openai:cu130-nightly` Docker image or a source build with `ENABLE_SM120=1` + `flash_attn 2.8.3` for SM120 - but here the pinned vLLM 0.24.0 wheel already served NVFP4 at 2094 tok/s via the cutlass FP4 GEMM backend, so the 0.24.0 wheel already carries usable sm_120 FP4 kernels; the nightly Docker then lifts it a further +41% to 2952 tok/s (peak 4750) via `fuse_act_quant`, see [The nightly Docker image lifts NVFP4](#the-nightly-docker-image-lifts-nvfp4-a-further-41)
- **`max_num_batched_tokens=4096` is their standard** - used even at 900K context; do not raise it to chase prefill (our sweep confirms it is inert)
- **`max_num_seqs=16` for single-user, 128 for multi-user** - matches the concurrency-cap rule above
- **FP8 is their throughput choice when VRAM allows, but NVFP4 beat it here** - their dense-model guidance ("FP8 fastest") favours FP8 over 4-bit weight-only; our measurement found native FP4 (NVFP4, activation-quantized) at 2094 tok/s vs FP8 941 - the distinction is FP4-with-FP4-activations vs INT4-weight-only, not 4-bit vs 8-bit per se
- **Speculative decoding (MTP) accelerates decode, not prefill** - their +50-70% gains are decode-side; a 14K-in / ~350-out judge call is prefill-bound, so MTP barely moves it
- **Their benchmarks do not cover this exact workload** - the rtx6kpro throughput tables are decode-side, on MoE models, multi-GPU (TP4/TP8), long-context; there is no dense-70B single-GPU prefill number to copy, so the levers above are transferable guidance, not a turnkey config

## Batched generation (resumable)

Pass many prompts at once so vLLM's scheduler keeps the batch full; group them only to checkpoint progress. The drain at each group boundary is negligible when the group is far larger than the concurrency:

```python
sp = SamplingParams(temperature=0.0, max_tokens=1280, seed=0)
GEN_BATCH = 256
for g0 in range(0, len(prompts), GEN_BATCH):
    grp = prompts[g0:g0 + GEN_BATCH]
    outs = llm.chat([[{"role": "user", "content": p}] for p in grp], sp, use_tqdm=False)
    for p, o in zip(grp, outs):
        text = o.outputs[0].text
        answer = text[text.rfind("</think>") + 8:] if "</think>" in text else text
        # parse answer, cache to disk for resumability
```

## Sequential two-model teardown

Running a second model in the same process requires releasing the first model's VRAM. In-process mode (fix 2) makes this deterministic:

```python
import gc, torch
from vllm.distributed.parallel_state import (destroy_model_parallel,
                                              destroy_distributed_environment)
del llm
destroy_model_parallel()
destroy_distributed_environment()
gc.collect()
torch.cuda.empty_cache()
```

## Gotchas

- **Dense sparsity_config** - strip it from the cached `config.json`; the weights are unaffected (fix 1)
- **WSL2 pin memory** - `VLLM_WSL2_ENABLE_PIN_MEMORY=1`, or the V2 model runner aborts with `UVA is not available` (fix 3)
- **nvcc not on PATH** - the FP8-KV FlashInfer kernel JIT-compiles at first use; point `CUDA_HOME` at the pip cu13 toolkit (fix 4)
- **lib vs lib64** - the JIT link wants `$CUDA/lib64`; symlink `lib64 → lib` and create the unversioned `.so` aliases (fix 5)
- **First-batch latency** - the FlashInfer kernel compiles once (~1-2 min) and caches under `~/.cache/flashinfer`; subsequent runs skip it
- **FP8 KV vs fp16 KV** - if the JIT path cannot be made to work, `kv_cache_dtype="auto"` uses prebuilt kernels (no JIT) at roughly half the concurrency (~8x here)
- **Block-scaled FP8 breaks DeepGEMM on sm_120** - a block-scaled FP8 checkpoint (e.g. `Qwen3-32B-FP8`) routes through DeepGEMM and dies with `Unknown SF transformation` at `deepgemm/layout.hpp`; use a per-tensor `-dynamic` FP8 checkpoint (Cutlass path) instead
- **Preemption thrash** - `max_num_seqs` above the load-time concurrency ceiling preempts and re-prefills 15K-token sequences in a loop; cap it at the ceiling and right-size `max_model_len` (see Tuning)

## Sources

- **rtx6kpro repo** - community knowledge base for the RTX PRO 6000 Blackwell (drivers, CUDA, inference-engine recipes, tuning notes): <https://github.com/local-inference-lab/rtx6kpro>
- **rtx6kpro vLLM guide** - card-specific engine args (incl. `max_num_seqs=16`) for the RTX PRO 6000 Blackwell: <https://github.com/local-inference-lab/rtx6kpro/blob/master/inference-engines/vllm.md>
- **NVFP4 checkpoint** - `nvidia/Llama-3.3-70B-Instruct-NVFP4` (ModelOpt NVFP4, group_size 16, FP8 KV), the 2094 tok/s prefill winner: <https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-NVFP4>
- **vLLM INT4 quantization (llm-compressor)** - confirms llm-compressor INT4 is W4A16 weight-only (the FP16-dequant prefill path): <https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/int4/>
- **vLLM production guide 2026** - serving and tuning on owned hardware: <https://vrlatech.com/running-vllm-on-your-own-hardware-the-production-guide-for-2026/>
- **Spheron RTX PRO 6000 overview** - card specs and capability context: <https://www.spheron.network/blog/rent-nvidia-rtx-pro-6000/>
- **vLLM preemption docs** - the `max_num_seqs` / KV-budget relationship and the recompute penalty on long context (vLLM optimization / performance guide)
