"""R14-H132 (R14-A3 kill-gate) - frozen-representation ladder on PRETRAINED,
un-fine-tuned mmBERT-base vs mmBERT-small.

`R12-H123_layerprobe.py` is imported and its protocol reused UNMODIFIED: the
same `slice_rows()` fixed-seed 20,000-row slice of the clean public mix, the
same seed, the same 30% test split, the same per-layer standardized logistic
probe, the same 23 hidden states (both models carry 22 transformer layers). No
arena, no gold.

The only difference from H123 is the object encoded: a pretrained checkpoint
loaded by Hub id rather than a fine-tuned `trunk/` directory. The DANN-group
probe is retained and reported, but it carries no clause here - the models are
un-fine-tuned, so there is no adversary to have erased anything.

  LICENSE  best-layer AUC(base) - best-layer AUC(small) >= 0.010
  KILL     < 0.005
  NO-READ  AUC(base) < 0.60 (floor effect)

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      uv run python experiments/grounding-semantic/R14_H132_layerprobe.py
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import importlib.util
import json
import pathlib
import time

import numpy as np

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
RESULT = HERE / "R14_gate_H132_layerprobe.json"
CACHE = ROOT / "tmp"
CACHE.mkdir(exist_ok=True)

MODELS = {"base": "jhu-clsp/mmBERT-base", "small": "jhu-clsp/mmBERT-small"}


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H123 = _mod("h123", "R12-H123_layerprobe.py")


def encode_pretrained(hub_id, claims, chunks, cache):
    """H123.encode with the trunk loaded from the Hub id instead of `<ckpt>/trunk`."""
    if cache.exists():
        arr = np.load(cache)["h"]
        print(f"  cached {cache.name} {arr.shape}", flush=True)
        return arr
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(hub_id)
    trunk = AutoModel.from_pretrained(hub_id).cuda().eval()
    trunk.config.reference_compile = False
    d = trunk.config.hidden_size
    n_layers = trunk.config.num_hidden_layers + 1
    assert n_layers == H123.N_LAYERS, f"{hub_id}: {n_layers} hidden states, protocol wants {H123.N_LAYERS}"
    out = np.zeros((len(claims), H123.N_LAYERS, d), dtype=np.float16)
    t0 = time.time()
    with torch.inference_mode():
        for j in range(0, len(claims), H123.BATCH):
            enc = tok(claims[j : j + H123.BATCH], chunks[j : j + H123.BATCH],
                      return_tensors="pt", padding=True, truncation=True,
                      max_length=H123.MAX_LEN)
            enc = {k: v.cuda() for k, v in enc.items()}
            hs = trunk(**enc, output_hidden_states=True).hidden_states
            assert len(hs) == H123.N_LAYERS
            out[j : j + H123.BATCH] = torch.stack([h[:, 0] for h in hs], dim=1).half().cpu().numpy()
            if j % (H123.BATCH * 100) == 0:
                print(f"    {j}/{len(claims)} ({time.time() - t0:.0f}s)", flush=True)
    np.savez(cache, h=out)
    del trunk
    torch.cuda.empty_cache()
    print(f"  encoded {hub_id} in {time.time() - t0:.0f}s -> {cache.name}", flush=True)
    return out


def main():
    import torch

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    claims, chunks, y, g, tag_names = H123.slice_rows()
    print(f"slice: {len(y)} rows, grounded rate {y.mean():.3f}, {len(tag_names)} groups",
          flush=True)

    rng = np.random.default_rng(H123.SEED + 1)
    perm = rng.permutation(len(y))
    n_te = int(H123.TEST_FRAC * len(y))
    te, tr = perm[:n_te], perm[n_te:]

    per_model = {}
    for name, hub in MODELS.items():
        print(f"\n== {name}: {hub}", flush=True)
        H = encode_pretrained(hub, claims, chunks, CACHE / f"R14_H132_cls_{name}.npz")
        auc, gacc = H123.probe(H, y, g, tr, te)
        del H
        per_model[name] = {
            "hub_id": hub,
            "hidden_size": int(np.load(CACHE / f"R14_H132_cls_{name}.npz")["h"].shape[2]),
            "auc_per_layer": auc,
            "group_acc_per_layer": gacc,
            "best_layer_auc": max(auc),
            "best_layer": int(np.argmax(auc)),
            "auc_final_layer22": auc[22],
        }

    gap = per_model["base"]["best_layer_auc"] - per_model["small"]["best_layer_auc"]
    gap = round(float(gap), 4)
    auc_base = per_model["base"]["best_layer_auc"]

    if auc_base < 0.60:
        verdict, clause = "NO-READ", f"AUC(base) {auc_base:.4f} < 0.60 - floor effect"
    elif gap >= 0.010:
        verdict, clause = "LICENSE", f"base-minus-small best-layer AUC gap {gap:+.4f} >= 0.010"
    elif gap < 0.005:
        verdict, clause = "KILL", f"base-minus-small best-layer AUC gap {gap:+.4f} < 0.005"
    else:
        verdict, clause = "UNRESOLVED", f"gap {gap:+.4f} lies in [0.005, 0.010)"

    res = {
        "gate": "R14-H132 (R14-A3 kill-gate) pretrained layer probe, mmBERT-base vs mmBERT-small",
        "protocol": "R12-H123_layerprobe.py unmodified (slice_rows, probe, seed, split)",
        "layer_convention": H123.__doc__.split("Layer convention recorded in the result JSON:")[-1]
        .split("Data:")[0].strip() if "Layer convention" in (H123.__doc__ or "") else
        "layer l = hidden_states[l]; l=0 embedding output, l=1..22 transformer layers",
        "n_rows": int(len(y)), "test_frac": H123.TEST_FRAC, "seed": H123.SEED,
        "max_len": H123.MAX_LEN,
        "data": "fixed-seed slice of R9-H105_clean_mix.public_train(); no arena, no gold",
        "dann_groups": tag_names,
        "per_model": per_model,
        "best_layer_auc_gap_base_minus_small": gap,
        "bar": "LICENSE >= 0.010; KILL < 0.005; NO-READ if AUC(base) < 0.60",
        "verdict": verdict, "clause_fired": clause,
        "note": "group-accuracy columns are reported for the record only; both models are "
                "un-fine-tuned, so no adversarial-erasure clause applies here",
    }
    RESULT.write_text(json.dumps(res, indent=2))
    print("\n" + "=" * 92)
    print("R14-H132 PRETRAINED LAYER PROBE")
    print("=" * 92)
    for n, v in per_model.items():
        print(f"  {n:6s} best-layer AUC {v['best_layer_auc']:.4f} @ layer {v['best_layer']}   "
              f"final-layer AUC {v['auc_final_layer22']:.4f}")
    print(f"  gap (base - small) = {gap:+.4f}")
    print(f"\n  VERDICT: {verdict}\n  {clause}\n  -> {RESULT}")


if __name__ == "__main__":
    main()
