"""Push the OpenVINO int8 grounder IRs to the HuggingFace Hub with model cards.

Reads HF_TOKEN from the environment (retrieved from the pass-cli vault by the caller).
Creates one repo per model under the token's namespace, writes a technical-documentation
style model card, and uploads the IR + config + tokenizer from models/ov/<name>/.
"""
import os
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
NS = api.whoami()["name"]
ROOT = Path(__file__).resolve().parents[1]
OV = ROOT / "models" / "ov"
PRIVATE = os.environ.get("HF_PRIVATE", "1") == "1"

MODELS = [
    dict(
        dir="bge-reranker-v2-m3", repo="bge-reranker-v2-m3-openvino-int8",
        base="BAAI/bge-reranker-v2-m3", arch="XLM-RoBERTa-large cross-encoder",
        license="apache-2.0", role="relevance reranker", method="NNCF int8 (Fast Bias Correction)",
        parity="0.9976", size_mb=571, fp32="> 2 GB", pipeline_tag="text-classification",
    ),
    dict(
        dir="bge-m3", repo="bge-m3-openvino-int8",
        base="BAAI/bge-m3", arch="XLM-RoBERTa-large bi-encoder",
        license="mit", role="top-k chunk pre-filter (CLS-pooled embeddings)",
        method="NNCF int8 (Fast Bias Correction)",
        parity="0.9941", size_mb=570, fp32="~2.2 GB", pipeline_tag="feature-extraction",
    ),
    dict(
        dir="mDeBERTa-v3-nli", repo="mdeberta-v3-base-mnli-xnli-openvino-int8",
        base="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", arch="DeBERTa-v2 NLI cross-encoder",
        license="mit", role="NLI entailment scorer (entailment = class index 0)",
        method="NNCF **SmoothQuant** (alpha 0.7) + Fast Bias Correction",
        parity="0.9863 (full-gold 0.9841)", size_mb=318, fp32="1.12 GB",
        pipeline_tag="zero-shot-classification",
    ),
]

CARD = """---
license: {license}
base_model: {base}
library_name: openvino
pipeline_tag: {pipeline_tag}
tags:
- openvino
- int8
- nncf
- quantized
- grounding
- cross-encoder
---

# {base} - OpenVINO int8

OpenVINO int8 IR of [`{base}`](https://huggingface.co/{base}) ({arch}), quantized for CPU inference as the **{role}** in a single-engine semantic grounding pipeline. The int8 preserves the fp32 max-over-chunks grounding signal at pearson **{parity}** - no measurable quality loss.

## Model

- **Base model** - [`{base}`](https://huggingface.co/{base}) ({arch})
- **Format** - OpenVINO IR (`openvino_model.xml` + `openvino_model.bin`), int8
- **Quantization** - {method}
- **int8 parity vs fp32** - pearson {parity} on the max-over-chunks grounding score
- **Size** - {size_mb} MB int8 (fp32 {fp32})
- **Role** - {role}
- **Runtime** - OpenVINO on CPU (x86-64 Intel/AMD via AVX2 / AVX-512-VNNI; ARM/Graviton via the OpenVINO ARM plugin)

## Usage

```python
from huggingface_hub import snapshot_download
import numpy as np, openvino as ov
from transformers import AutoTokenizer

d = snapshot_download("{ns}/{repo}")
core = ov.Core()
model = core.compile_model(core.read_model(f"{{d}}/openvino_model.xml"), "CPU")
tok = AutoTokenizer.from_pretrained(d)

enc = tok(["a sentence"], ["another sentence"], return_tensors="np",
          padding=True, truncation=True, max_length=512)
feed = {{"input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64)}}
logits = model(feed)[model.output(0)]
```

## Quantization details

- **Method** - {method}, via NNCF / `optimum-intel`
- **Calibration** - 128 (claim, chunk) pairs sampled from 3 selected agentic chat RAG datasets
- **Why this matters** - DeBERTa-v2 disentangled attention is int8-hostile (plain dynamic int8 collapses to pearson 0.35); NNCF SmoothQuant migrates the activation outliers into the per-channel weights, which is the only int8 method that preserves the signal. The standard-attention bge models quantize cleanly with plain NNCF int8

## License

Inherits the `{license}` license of the base model.
"""


def main():
    print(f"namespace: {NS}  private: {PRIVATE}", flush=True)
    for m in MODELS:
        repo_id = f"{NS}/{m['repo']}"
        src = OV / m["dir"]
        assert (src / "openvino_model.xml").exists(), f"missing IR {src}"
        try:
            api.create_repo(repo_id, repo_type="model", private=PRIVATE, exist_ok=True)
        except Exception as e:
            print(f"CREATE FAIL {repo_id}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            raise
        (src / "README.md").write_text(CARD.format(ns=NS, **m))
        api.upload_folder(folder_path=str(src), repo_id=repo_id, repo_type="model",
                          commit_message="OpenVINO int8 IR + model card")
        print(f"OK -> https://huggingface.co/{repo_id}", flush=True)
    print("ALL UPLOADED", flush=True)


if __name__ == "__main__":
    main()
