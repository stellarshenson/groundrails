"""Standalone: pick best SmoothQuant alpha by parity, score full gold with the int8
NLI (OpenVINO async throughput), cache to data/interim/model_scores/. One-time heavy
compute (~90 min on CPU) so the notebook can load the cache and run end-to-end fast."""
import os, time, json
os.environ["CUDA_VISIBLE_DEVICES"]=""; os.environ["TOKENIZERS_PARALLELISM"]="false"; os.environ["HF_HUB_OFFLINE"]="1"
import numpy as np, openvino as ov
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer
from grounding_models import load_gold, SCORES_DIR

MODEL_ID="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"; ENT_IDX=0
WORK="/tmp/mde_quant"; ALPHAS=[0.5,0.7,0.9]; SEED=42
recs=load_gold(); y=np.array([r["label"] for r in recs])
cached_nli=np.load(SCORES_DIR/"MoritzLaurer__mDeBERTa-v3-base-mnli-xnli.npy")
tok=AutoTokenizer.from_pretrained(MODEL_ID); core=ov.Core()
np.random.seed(SEED); half=25
sidx=np.sort(np.concatenate([np.random.choice(np.where(y==0)[0],half,replace=False),
                             np.random.choice(np.where(y==1)[0],half,replace=False)]))

def compiled(alpha):
    return core.compile_model(core.read_model(f"{WORK}/sq_a{int(alpha*100)}/openvino_model.xml"),
                              "CPU", {"PERFORMANCE_HINT":"THROUGHPUT"})

def score_async(cm, rs, bs=16):
    names=[i.get_any_name() for i in cm.inputs]
    n=cm.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
    pairs=[]; owner=[]
    for k,r in enumerate(rs):
        for c in r["chunks"]:
            pairs.append((c,r["claim"])); owner.append(k)
    best=np.full(len(rs),-1e9)
    q=ov.AsyncInferQueue(cm,n)
    def cb(req,ud):
        idxs,=ud; lg=req.get_output_tensor(0).data
        ex=np.exp(lg-lg.max(1,keepdims=True)); s=(ex/ex.sum(1,keepdims=True))[:,ENT_IDX]
        for m,oi in enumerate(idxs):
            if s[m]>best[oi]: best[oi]=s[m]
    q.set_callback(cb)
    for j in range(0,len(pairs),bs):
        sub=pairs[j:j+bs]; ow=owner[j:j+bs]
        e=tok([p[0] for p in sub],[p[1] for p in sub],padding=True,truncation=True,max_length=512,return_tensors="np")
        feed={"input_ids":e["input_ids"].astype(np.int64),"attention_mask":e["attention_mask"].astype(np.int64)}
        q.start_async({nm:feed[nm] for nm in names if nm in feed},(ow,))
    q.wait_all()
    return best

# 1. parity across alphas on the balanced sample
ref=cached_nli[sidx]; sample=[recs[i] for i in sidx]; par={}
for a in ALPHAS:
    t=time.time(); cm=compiled(a); s=score_async(cm,sample)
    p=float(pearsonr(s,ref)[0])
    par[a]={"pearson":p,"spearman":float(spearmanr(s,ref)[0]),
            "max_abs":float(np.abs(s-ref).max()),"std":float(s.std())}
    print(f"alpha={a} pearson={p:.4f} std={par[a]['std']:.3f} ({time.time()-t:.0f}s)",flush=True)
best_a=max(ALPHAS,key=lambda a:par[a]["pearson"])
print(f"BEST alpha={best_a} pearson={par[best_a]['pearson']:.4f}",flush=True)

# 2. full-gold score the winner, cache
t=time.time(); cm=compiled(best_a); full=score_async(cm,recs)
print(f"full-gold scored in {(time.time()-t)/60:.1f} min",flush=True)
np.save(SCORES_DIR/"mDeBERTa-v3-int8-sq.npy", full)
meta={"alpha":best_a,"parity_sample":par,"n":len(recs),
      "full_pearson_vs_fp32":float(pearsonr(full,cached_nli)[0]),
      "size_mb":318}
(SCORES_DIR/"mDeBERTa-v3-int8-sq.json").write_text(json.dumps(meta,indent=2))
print("CACHED ->",SCORES_DIR/"mDeBERTa-v3-int8-sq.npy",flush=True)
print("full vs fp32 pearson:",round(meta["full_pearson_vs_fp32"],4),flush=True)
