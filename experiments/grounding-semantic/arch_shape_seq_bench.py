import os,time
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'; os.environ['CUDA_VISIBLE_DEVICES']='1'
import torch,torch.nn as nn,torch.nn.functional as F
class Blk(nn.Module):
    def __init__(s,d,h,dff,parallel=False):
        super().__init__(); s.d=d;s.h=h;s.par=parallel
        s.n1=nn.LayerNorm(d); s.qkv=nn.Linear(d,3*d,bias=False); s.o=nn.Linear(d,d,bias=False)
        s.n2=nn.LayerNorm(d); s.wi=nn.Linear(d,2*dff,bias=False); s.wo=nn.Linear(dff,d,bias=False)
    def a(s,x):
        B,T,_=x.shape; q,k,v=s.qkv(x).chunk(3,-1)
        q,k,v=[t.view(B,T,s.h,-1).transpose(1,2) for t in (q,k,v)]
        return s.o(F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(B,T,s.d))
    def f(s,x):
        a,b=s.wi(x).chunk(2,-1); return s.wo(F.gelu(a)*b)
    def forward(s,x):
        if s.par: n=s.n1(x); return x+s.a(n)+s.f(n)
        x=x+s.a(s.n1(x)); return x+s.f(s.n2(x))
class Enc(nn.Module):
    def __init__(s,L,d,h,dff,**k):
        super().__init__(); s.b=nn.ModuleList([Blk(d,h,dff,**k) for _ in range(L)]); s.n=nn.LayerNorm(d)
    def forward(s,x):
        for b in s.b: x=b(x)
        return s.n(x)
def gf(L,d,dff,T): return L*(8*T*d*d+4*T*T*d+2*T*(2*d*dff)+2*T*(dff*d))/1e9
def run(tag,L,d,h,dff,T,B,**k):
    m=Enc(L,d,h,dff,**k).cuda().to(torch.bfloat16).eval(); n=sum(p.numel() for p in m.parameters())
    x=torch.randn(B,T,d,device='cuda',dtype=torch.bfloat16)
    with torch.no_grad():
        for _ in range(12): m(x)
        torch.cuda.synchronize(); t0=time.perf_counter()
        for _ in range(50): m(x)
        torch.cuda.synchronize(); e=(time.perf_counter()-t0)/50*1e3
        g=torch.cuda.CUDAGraph(); st=torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3): m(x)
        torch.cuda.current_stream().wait_stream(st)
        with torch.cuda.graph(g): m(x)
        for _ in range(8): g.replay()
        torch.cuda.synchronize(); t0=time.perf_counter()
        for _ in range(50): g.replay()
        torch.cuda.synchronize(); gr=(time.perf_counter()-t0)/50*1e3
    print(f"{tag:<40}{L:>4}{d:>6}{T:>6}{B:>4}{n/1e6:>9.1f}{gf(L,d,dff,T)*B:>9.1f}{e:>10.3f}{gr:>10.3f}",flush=True)
    del m,x; torch.cuda.empty_cache()
print(f"{'tag':<40}{'L':>4}{'d':>6}{'T':>6}{'B':>4}{'body M':>9}{'GFLOP':>9}{'eager':>10}{'graph':>10}",flush=True)
print("--- sequence length sweep, recommended shape d1024 L16 dff1536, B=3 ---",flush=True)
for T in (256,384,512,640,768,1024): run(f"seq {T}",16,1024,16,1536,T,3)
print("--- candidate specs at B=3 T=512 ---",flush=True)
run("REC  d1024 L16 dff1536 parallel",16,1024,16,1536,512,3,parallel=True)
run("REC  d1024 L20 dff1536 parallel",20,1024,16,1536,512,3,parallel=True)
run("ALT  d768  L22 dff1152 (mmBERT)",22,768,12,1152,512,3)
run("ALT  d768  L22 dff1152 parallel",22,768,12,1152,512,3,parallel=True)
run("d1280 L12 dff1920 parallel",12,1280,20,1920,512,3,parallel=True)
print("--- batch scaling for REC d1024 L16 T512 ---",flush=True)
for B in (1,3,8,16): run(f"B={B}",16,1024,16,1536,512,B,parallel=True)
print("--- MICE-style top-layer cost: 64 claim tokens only ---",flush=True)
for T in (64,96): run(f"interaction layers T={T} (L6)",6,1024,16,1536,T,3,parallel=True)
