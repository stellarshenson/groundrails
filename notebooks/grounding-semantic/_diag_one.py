"""Score ONE model on GPU against a tiny record. Run per-model in its own
subprocess so a CUDA crash is isolated and the exit code names the culprit.

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python notebooks/_diag_one.py <model_name>
"""
import sys

import grounding_models as gm

name = sys.argv[1]
rec = [{"claim": "The torque is 25 Nm.",
        "chunks": ["Tighten the bolt to 25 Nm before use.", "Unrelated cleaning text."],
        "label": 1, "lang": "en"}]

emb = {c["name"]: c for c in gm.EMBEDDERS}
cro = {c["name"]: c for c in gm.CROSS}
dev = gm.pick_gpu()
if name in emb:
    s = gm.embed_scores(emb[name], rec, dev, bs=8)
    print(f"OK embed {name} score={s[0]:.3f}", flush=True)
else:
    s = gm.cross_scores(cro[name], rec, dev, bs=8)
    print(f"OK {cro[name]['kind']} {name} score={s[0]:.3f}", flush=True)
