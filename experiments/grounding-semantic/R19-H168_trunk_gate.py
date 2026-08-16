"""R19-H168 GATE A - can EuroBERT-210m be loaded and driven on this host at all?

The cheapest possible kill. EuroBERT ships `modeling_eurobert.py` and requires
`trust_remote_code`, and the campaign's hardware record has custom-remote-code
models throwing device-side asserts on these cards. A miss here kills the arm
before any mix is assembled or any GPU-hour is spent on training.

Runs on GPU2 (RTX 5000 Ada, sm_89) rather than a Blackwell card: the recorded
failure mode is custom CUDA kernels compiled against older architectures, and
sm_89 is the compatibility fallback.

Checks, in order, each fatal:
  1. config + tokenizer load
  2. encoder loads as `AutoModel` (NOT the MaskedLM head - the 128,256x768 LM
     output layer is 98.5M params we never use and would put the arm over its
     own reported size)
  3. parameter count matches the campaign's published arithmetic (211.7M total,
     98.5M embedding, 113.2M body) and sits under the 400M budget
  4. hidden size is 768, so `task_head`, `domain_head` and the adapter transfer
     from the mmBERT recipe with ZERO dimensional change
  5. a real fp32 forward on real text produces finite CLS vectors
  6. a bf16-autocast forward - the dtype the trainer actually encodes in
  7. a 512-token padded batch, the trainer's real shape

Run: CUDA_VISIBLE_DEVICES=2 uv run python R19-H168_trunk_gate.py
"""

import json
import os
import pathlib
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import importlib.util  # noqa: E402

import torch  # noqa: E402
from transformers import AutoConfig, AutoModel, AutoTokenizer  # noqa: E402

HERE = pathlib.Path(__file__).parent


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


COMPAT = _mod("h168compat", "R19-H168_eurobert_compat.py")
REPO = "EuroBERT/EuroBERT-210m"
BUDGET = 400_000_000

# The campaign's published arithmetic for this trunk
# (reports/research-grounding-architecture.md, validated against the vendor card).
EXPECTED = {"total_m": 211.7, "embed_m": 98.5, "body_m": 113.2, "hidden": 768,
            "layers": 12, "vocab": 128_256, "tol_m": 1.0}

SAMPLES = [
    ("en", "The company reported revenue of 4.2 billion dollars in fiscal 2023."),
    ("de", "Das Unternehmen meldete im Geschaeftsjahr 2023 einen Umsatz von 4,2 Milliarden."),
    ("fr", "La societe a declare un chiffre d'affaires de 4,2 milliards en 2023."),
    ("es", "La empresa registro ingresos de 4.200 millones en el ejercicio 2023."),
    ("it", "La societa ha registrato ricavi per 4,2 miliardi nell'esercizio 2023."),
    ("pl", "Spolka odnotowala przychody w wysokosci 4,2 miliarda w roku 2023."),
    ("hu", "A vallalat 4,2 milliard bevetelt jelentett a 2023-as penzugyi evben."),
    ("cn", "该公司报告称,2023财年收入为42亿美元。"),
    ("table", "| quarter | revenue | margin |\n| Q1 2023 | 1,024 | 12.4% |\n| Q2 2023 | 1,180 | 13.1% |"),
]


def fail(msg):
    print(f"\n=== H168 GATE A FAILED: {msg} ===", flush=True)
    raise SystemExit(1)


def main():
    t0 = time.time()
    out = HERE / "R19-H168_trunk_gate.json"
    res = {"arm": "R19-H168 EuroBERT-210m trunk swap", "gate": "A - load and drive",
           "repo": REPO, "status": "RUNNING"}

    print(f"=== R19-H168 GATE A  {time.strftime('%F %T')} ===", flush=True)
    print(f"  device: {torch.cuda.get_device_name(0)} "
          f"(sm_{''.join(map(str, torch.cuda.get_device_capability(0)))})", flush=True)
    res["device"] = torch.cuda.get_device_name(0)

    # --- 1. config + tokenizer -------------------------------------------------
    cfg = AutoConfig.from_pretrained(REPO, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
    print(f"  config: {cfg.model_type} L={cfg.num_hidden_layers} "
          f"H={cfg.hidden_size} vocab={cfg.vocab_size} "
          f"ctx={cfg.max_position_embeddings}", flush=True)
    res["config"] = {"model_type": cfg.model_type, "layers": cfg.num_hidden_layers,
                     "hidden": cfg.hidden_size, "vocab": cfg.vocab_size,
                     "max_pos": cfg.max_position_embeddings,
                     "heads": cfg.num_attention_heads,
                     "kv_heads": getattr(cfg, "num_key_value_heads", None),
                     "intermediate": cfg.intermediate_size,
                     "rope_theta": getattr(cfg, "rope_theta", None)}

    # --- 1b. load-compatibility shim + its analytic positive control ----------
    needed = COMPAT.install()
    ctrl = COMPAT.verify(cfg)
    print(f"  rope shim: needed={needed}  head_dim={ctrl['head_dim']} "
          f"theta={ctrl['rope_theta']}  n_freqs={ctrl['n_freqs']}  "
          f"max_err_vs_analytic={ctrl['max_abs_err_vs_analytic']:.2e} "
          f"-> {'PASS' if ctrl['pass'] else 'FAIL'}", flush=True)
    res["rope_shim"] = {"needed": needed, "control": ctrl}
    if not ctrl["pass"]:
        fail("the RoPE compatibility shim does not reproduce the analytic "
             "inverse frequencies - a wrong positional encoding would train "
             "silently, so this is fatal")

    if cfg.hidden_size != EXPECTED["hidden"]:
        fail(f"hidden size {cfg.hidden_size} != {EXPECTED['hidden']} - the heads "
             f"would need re-dimensioning and the recipe would NOT be exact")
    if cfg.num_hidden_layers != EXPECTED["layers"]:
        fail(f"layers {cfg.num_hidden_layers} != {EXPECTED['layers']}")

    # --- 2. encoder only, no LM head ------------------------------------------
    try:
        model = AutoModel.from_pretrained(REPO, trust_remote_code=True,
                                          torch_dtype=torch.float32)
    except Exception as e:  # noqa: BLE001
        fail(f"AutoModel.from_pretrained raised {type(e).__name__}: {e}")

    # --- 3./4. parameter arithmetic -------------------------------------------
    total = sum(p.numel() for p in model.parameters())
    embed = sum(p.numel() for n, p in model.named_parameters() if "embed" in n.lower())
    body = total - embed
    print(f"  params: total {total/1e6:.1f}M  embed {embed/1e6:.1f}M  "
          f"body {body/1e6:.1f}M", flush=True)
    res["params"] = {"total": total, "embed": embed, "body": body,
                     "total_m": round(total / 1e6, 1), "embed_m": round(embed / 1e6, 1),
                     "body_m": round(body / 1e6, 1),
                     "expected": EXPECTED, "budget": BUDGET,
                     "under_budget": bool(total < BUDGET)}
    if total >= BUDGET:
        fail(f"{total/1e6:.1f}M is over the {BUDGET/1e6:.0f}M model budget")
    if abs(total / 1e6 - EXPECTED["total_m"]) > EXPECTED["tol_m"]:
        fail(f"total {total/1e6:.1f}M != published {EXPECTED['total_m']}M "
             f"(tol {EXPECTED['tol_m']}M) - the campaign's size arithmetic is wrong "
             f"somewhere and the size-fairness claim must be re-derived before use")

    # --- 5. real fp32 forward --------------------------------------------------
    try:
        model = model.to("cuda").eval()
    except Exception as e:  # noqa: BLE001
        fail(f".to('cuda') raised {type(e).__name__}: {e}")

    texts = [s for _, s in SAMPLES]
    enc = tok(texts, padding=True, truncation=True, max_length=512,
              return_tensors="pt").to("cuda")
    res["forward_input_keys"] = sorted(enc.keys())
    try:
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        fail(f"fp32 forward raised {type(e).__name__}: {e}")
    cls = h[:, 0]
    if not torch.isfinite(cls).all():
        fail("fp32 forward produced non-finite CLS values")
    print(f"  fp32 forward OK: {tuple(h.shape)}  "
          f"cls_norm mean {cls.norm(dim=-1).mean():.3f}", flush=True)
    res["fp32_forward"] = {"shape": list(h.shape),
                           "cls_norm_mean": round(float(cls.norm(dim=-1).mean()), 4),
                           "finite": True}

    # --- 6. bf16 autocast, the trainer's encode dtype --------------------------
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hb = model(**enc).last_hidden_state
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        fail(f"bf16 autocast forward raised {type(e).__name__}: {e}")
    clsb = hb[:, 0].float()
    if not torch.isfinite(clsb).all():
        fail("bf16 forward produced non-finite CLS values")
    cos = torch.nn.functional.cosine_similarity(cls, clsb, dim=-1)
    print(f"  bf16 forward OK: cosine to fp32 min {cos.min():.5f} "
          f"mean {cos.mean():.5f}", flush=True)
    res["bf16_forward"] = {"cos_min": round(float(cos.min()), 5),
                           "cos_mean": round(float(cos.mean()), 5), "finite": True}

    # --- 7. the trainer's real batch shape -------------------------------------
    big = [SAMPLES[i % len(SAMPLES)][1] * 40 for i in range(96)]
    encb = tok(big, padding="max_length", truncation=True, max_length=512,
               return_tensors="pt").to("cuda")
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(**encb)
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        fail(f"96x512 batch raised {type(e).__name__}: {e}")
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"  96x512 batch OK: peak {peak:.2f} GB", flush=True)
    res["batch_96x512"] = {"peak_alloc_gb": round(peak, 2), "ok": True}

    # --- 8. THE decisive check: does the shimmed model still speak? -----------
    # A wrong RoPE loads cleanly and emits finite activations, so every check
    # above would pass on a broken positional encoding. Masked-token prediction
    # is the one that cannot: garbage positions give garbage fills. This is the
    # positive control for the shim as a whole, not just its arithmetic.
    del model
    torch.cuda.empty_cache()
    from transformers import AutoModelForMaskedLM  # noqa: PLC0415
    mlm = AutoModelForMaskedLM.from_pretrained(
        REPO, trust_remote_code=True, dtype=torch.float32).to("cuda").eval()
    probes = [
        ("The capital of France is <|mask|>.", ("Paris",)),
        ("Water freezes at zero degrees <|mask|>.", ("Celsius", "celsius", "C")),
        ("The company reported revenue of four <|mask|> dollars.", ("billion", "million")),
    ]
    mlm_rows, n_hit = [], 0
    for text, wanted in probes:
        e = tok(text, return_tensors="pt").to("cuda")
        pos = (e["input_ids"][0] == tok.mask_token_id).nonzero()
        if pos.numel() == 0:
            fail(f"mask token not found in tokenized probe: {text!r}")
        with torch.no_grad():
            lg = mlm(**e).logits[0, int(pos[0])]
        top = [tok.decode([i]).strip() for i in lg.topk(5).indices.tolist()]
        hit = any(w in top for w in wanted)
        n_hit += hit
        mlm_rows.append({"probe": text, "top5": top, "wanted": list(wanted), "hit": hit})
        print(f"  MLM {'HIT ' if hit else 'MISS'} {text!r} -> {top}", flush=True)
    res["mlm_control"] = {"probes": mlm_rows, "n_hit": n_hit, "n_probes": len(probes),
                          "pass": bool(n_hit >= 2)}
    if n_hit < 2:
        fail(f"masked-token control: only {n_hit}/{len(probes)} probes recovered a "
             f"sensible fill - the shimmed model does not speak, so the RoPE or the "
             f"weight mapping is wrong and NOTHING may be trained on it")
    print(f"  MLM control {n_hit}/{len(probes)} PASS - the shimmed trunk speaks",
          flush=True)

    res["status"] = "PASS"
    res["seconds"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(res, indent=1))
    print(f"\n  -> {out.name}", flush=True)
    print("=== H168 GATE A PASS ===", flush=True)


if __name__ == "__main__":
    main()
