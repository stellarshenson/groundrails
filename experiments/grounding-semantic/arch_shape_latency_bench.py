import os, time, json, math
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'; os.environ['CUDA_VISIBLE_DEVICES']='1'
import torch, torch.nn as nn, torch.nn.functional as F

class Blk(nn.Module):
    def __init__(s,d,h,dff,parallel=False,attn_only=False):
        super().__init__(); s.d=d; s.h=h; s.par=parallel; s.ao=attn_only
        s.n1=nn.LayerNorm(d); s.qkv=nn.Linear(d,3*d,bias=False); s.o=nn.Linear(d,d,bias=False)
        if not attn_only:
            s.n2=nn.LayerNorm(d); s.wi=nn.Linear(d,2*dff,bias=False); s.wo=nn.Linear(dff,d,bias=False)
    def fwd_attn(s,x):
        B,T,_=x.shape
        q,k,v=s.qkv(x).chunk(3,-1)
        q,k,v=[t.view(B,T,s.h,-1).transpose(1,2) for t in (q,k,v)]
        return s.o(F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(B,T,s.d))
    def fwd_ffn(s,x):
        a,b=s.wi(x).chunk(2,-1); return s.wo(F.gelu(a)*b)
    def forward(s,x):
        if s.ao: return x+s.fwd_attn(s.n1(x))
        if s.par:
            n=s.n1(x); return x+s.fwd_attn(n)+s.fwd_ffn(n)
        x=x+s.fwd_attn(s.n1(x)); return x+s.fwd_ffn(s.n2(x))

class Enc(nn.Module):
    def __init__(s,L,d,h,dff,**kw):
        super().__init__(); s.b=nn.ModuleList([Blk(d,h,dff,**kw) for _ in range(L)]); s.n=nn.LayerNorm(d)
    def forward(s,x):
        for b in s.b: x=b(x)
        return s.n(x)

def gflops(L,d,dff,T,ao=False):
    per=8*T*d*d+4*T*T*d
    if not ao: per+= 2*T*(2*d*dff)+2*T*(dff*d)
    return L*per/1e9

def bench(name,L,d,h,dff,T=512,B=3,**kw):
    m=Enc(L,d,h,dff,**kw).cuda().to(torch.bfloat16).eval()
    n=sum(p.numel() for p in m.parameters())
    x=torch.randn(B,T,d,device='cuda',dtype=torch.bfloat16)
    with torch.no_grad():
        for _ in range(15): m(x)
        torch.cuda.synchronize(); t0=time.perf_counter()
        for _ in range(60): m(x)
        torch.cuda.synchronize(); eager=(time.perf_counter()-t0)/60*1e3
        # cuda graph
        g=torch.cuda.CUDAGraph(); st=torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3): m(x)
        torch.cuda.current_stream().wait_stream(st)
        with torch.cuda.graph(g): y=m(x)
        for _ in range(10): g.replay()
        torch.cuda.synchronize(); t0=time.perf_counter()
        for _ in range(60): g.replay()
        torch.cuda.synchronize(); graph=(time.perf_counter()-t0)/60*1e3
    gf=gflops(L,d,dff,T,kw.get('attn_only',False))*B
    print(f"{name:<34}{L:>4}{d:>6}{n/1e6:>9.1f}{gf:>9.1f}{eager:>10.3f}{graph:>10.3f}{gf/graph/1e3*1e3:>9.1f}",flush=True)
    del m,x; torch.cuda.empty_cache()
    return dict(name=name,L=L,d=d,body_M=n/1e6,gflop=gf,eager_ms=eager,graph_ms=graph)

print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}  bf16  B=3 T=512",flush=True)
print(f"{'shape':<34}{'L':>4}{'d':>6}{'body M':>9}{'GFLOP':>9}{'eager ms':>10}{'graph ms':>10}{'TFLOP/s':>9}",flush=True)
R=[]
R+=[bench("D  narrow-deep  d512 L53",53,512,8,1152)]
R+=[bench("C  d768 L28 (NeoBERT-ish)",28,768,12,1152)]
R+=[bench("B  d1024 L16",16,1024,16,1536)]
R+=[bench("A  wide-shallow d1536 L8",8,1536,24,2304)]
R+=[bench("B-par d1024 L16 parallel blk",16,1024,16,1536,parallel=True)]
R+=[bench("C-par d768 L28 parallel blk",28,768,12,1152,parallel=True)]
R+=[bench("SAN attn-only d1024 L36",36,1024,16,0,attn_only=True)]
R+=[bench("ref ModernBERT-base body",22,768,12,1152)]
R+=[bench("ref mDeBERTa-base body(mlp-ish)",12,768,12,1536)]
print(flush=True)
print("--- B=8 (top-8 rerank batch) ---",flush=True)
for a in [(53,512,8,1152,{}),(28,768,12,1152,{}),(16,1024,16,1536,{}),(8,1536,24,2304,{}),(22,768,12,1152,{})]:
    bench(f"B=8 L{a[0]} d{a[1]}",*a[:4],B=8,**a[4])
json.dump(R,open('/tmp/shape_bench.json','w'),indent=1)
