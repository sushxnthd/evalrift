#!/usr/bin/env python3
"""
ToxCast high-task-count atomicity stress test.

- MoleculeNet ToxCast: up to 617 binary assay tasks.
- Target chosen BEFORE training by a fixed label-statistics rule:
  among tasks with >=75% observed labels, maximize min(#positive,#negative).
- Shared tanh bottleneck + one logistic head per assay.
- Task-head gradients remain parent-level; only shared-trunk aggregation changes.
"""
import csv, gzip, io, json, math, os, time, urllib.request
from pathlib import Path
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import roc_auc_score

RDLogger.DisableLog("rdApp.*")
URL="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/toxcast_data.csv.gz"
SPLIT_MODE=os.environ.get("SPLIT_MODE","random").lower()
if SPLIT_MODE not in ("random","scaffold"): raise ValueError(SPLIT_MODE)

def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-30,30)))

def load(nbits=128):
    raw=gzip.decompress(urllib.request.urlopen(URL,timeout=120).read()).decode("utf-8")
    rows=list(csv.DictReader(io.StringIO(raw)))
    header=list(rows[0].keys())
    smiles_col=header[0]
    tasks=header[1:618]
    gen=AllChem.GetMorganGenerator(radius=2,fpSize=nbits)
    X=[];Y=[];scaff=[];bad=0
    for r in rows:
        m=Chem.MolFromSmiles(r[smiles_col])
        if m is None: bad+=1; continue
        X.append(gen.GetFingerprintAsNumPy(m).astype(np.float32))
        yy=np.full(len(tasks),np.nan,np.float32)
        for j,t in enumerate(tasks):
            v=(r[t] or "").strip()
            if v in ("0","1","0.0","1.0"): yy[j]=float(v)
        Y.append(yy)
        try: scaff.append(MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False))
        except Exception: scaff.append("")
    return np.asarray(X,np.float32),np.asarray(Y,np.float32),tasks,scaff,bad

def select_target(Y,tasks):
    n=len(Y); best=None
    for j,t in enumerate(tasks):
        v=Y[:,j];obs=~np.isnan(v);cnt=int(obs.sum())
        if cnt < .75*n: continue
        pos=int(np.sum(v[obs]==1));neg=int(np.sum(v[obs]==0))
        score=min(pos,neg)
        cand=(score,cnt,t,j,pos,neg)
        if best is None or cand[:2] > best[:2] or (cand[:2]==best[:2] and t<best[2]):
            best=cand
    if best is None: raise RuntimeError("No target satisfies coverage rule")
    score,cnt,t,j,pos,neg=best
    return j,{"task":t,"observed":cnt,"positive":pos,"negative":neg,"balance_score":score}

def split_idx(n,scaff,seed):
    if SPLIT_MODE=="random":
        p=np.random.default_rng(seed).permutation(n);c=int(.8*n);return p[:c],p[c:]
    groups={}
    for i,s in enumerate(scaff):groups.setdefault(s,[]).append(i)
    ordered=sorted(groups.items(),key=lambda kv:(-len(kv[1]),kv[0]))
    target=int(.8*n);tr=[];te=[]
    for _,idx in ordered:
        if len(tr)+len(idx)<=target or not tr:tr.extend(idx)
        else:te.extend(idx)
    return np.asarray(tr,int),np.asarray(te,int)

class Adam:
    def __init__(self,ps,lr=2e-3):
        self.lr=lr;self.t=0;self.m=[np.zeros_like(p) for p in ps];self.v=[np.zeros_like(p) for p in ps]
    def step(self,ps,gs):
        self.t+=1
        for i,(p,g) in enumerate(zip(ps,gs)):
            self.m[i]=.9*self.m[i]+.1*g;self.v[i]=.999*self.v[i]+.001*g*g
            p-=self.lr*(self.m[i]/(1-.9**self.t))/(np.sqrt(self.v[i]/(1-.999**self.t))+1e-8)

def init(seed,d,h,T):
    r=np.random.default_rng(seed)
    return [(r.normal(size=(d,h))*math.sqrt(2/d)).astype(np.float32),np.zeros(h,np.float32),
            (r.normal(size=(h,T))*math.sqrt(2/h)).astype(np.float32),np.zeros(T,np.float32)]

def posw(Y):
    out=np.ones(Y.shape[1],np.float32)
    for t in range(Y.shape[1]):
        v=Y[:,t];obs=~np.isnan(v);p=np.sum(v[obs]==1);n=np.sum(v[obs]==0)
        if p>0:out[t]=min(float(n/p),20)
    return out

def compute_grads(X,Y,ps,pw,ti,mode,k):
    W,b,V,c=ps;H=np.tanh(X@W+b);Z=H@V+c;B,T=Y.shape
    mask=~np.isnan(Y);Y0=np.nan_to_num(Y,nan=0.0)
    raw=(sigmoid(Z)-Y0)*np.where(Y0>.5,pw[None,:],1.0).astype(np.float32)*mask
    counts=mask.sum(0).astype(np.float32)
    denom=np.maximum(counts,1)[None,:]
    dl=raw/denom
    gV=H.T@dl;gc=dl.sum(0)
    dHc=dl@V.T
    dlr=dl.copy()
    obs=np.flatnonzero(mask[:,ti]);n=len(obs)
    if mode!="coarse" and n>0:
        dlr[:,ti]=0
        for child in np.array_split(np.arange(n),min(k,n)):
            if len(child):
                rows=obs[child];dlr[rows,ti]=raw[rows,ti]/len(child)
    dHr=dlr@V.T
    dAc=dHc*(1-H*H);dAr=dHr*(1-H*H)
    gWc=X.T@dAc;gbc=dAc.sum(0);gWr=X.T@dAr;gbr=dAr.sum(0)
    if mode=="norm":
        nc=math.sqrt(float(np.sum(gWc*gWc)+np.sum(gbc*gbc)));nr=math.sqrt(float(np.sum(gWr*gWr)+np.sum(gbr*gbr)))
        sc=nc/(nr+1e-12);gWr*=sc;gbr*=sc
    rec=0.0
    if n>0:
        parent=np.zeros(B,np.float32);parent[obs]=raw[obs,ti]/n
        can=np.zeros(B,np.float32)
        for child in np.array_split(np.arange(n),min(k,n)):
            if len(child):
                rows=obs[child];can[rows]=(raw[rows,ti]/len(child))*(len(child)/n)
        rec=float(np.linalg.norm(can-parent)/(np.linalg.norm(parent)+1e-12))
    return [gWr,gbr,gV,gc],rec

def evalm(X,Y,ps,ti):
    W,b,V,c=ps;P=sigmoid(np.tanh(X@W+b)@V+c);auc=[];dense=[]
    for t in range(Y.shape[1]):
        obs=~np.isnan(Y[:,t]);yy=Y[obs,t]
        if len(np.unique(yy))<2: auc.append(np.nan);dense.append(False);continue
        auc.append(float(roc_auc_score(yy,P[obs,t])))
        dense.append(np.sum(yy==1)>=20 and np.sum(yy==0)>=20)
    a=np.asarray(auc,float);dense=np.asarray(dense,bool);peer=np.ones(len(a),bool);peer[ti]=False
    return {"macro_auc":float(np.nanmean(a)),"target_auc":float(a[ti]),
            "peer_macro_auc":float(np.nanmean(a[peer])),
            "dense_peer_macro_auc":float(np.nanmean(a[peer&dense])),
            "valid_auc_tasks":int(np.sum(~np.isnan(a))),"dense_auc_tasks":int(np.sum(dense))}

def train(X,Y,tasks,scaff,ti,seed,mode,k,epochs=10,batch=1024,h=16):
    tr,te=split_idx(len(X),scaff,seed);Xtr,Ytr=X[tr],Y[tr];Xte,Yte=X[te],Y[te]
    ps=init(seed+3000,X.shape[1],h,Y.shape[1]);opt=Adam(ps,2e-3);pw=posw(Ytr);rng=np.random.default_rng(seed+999);maxrec=0
    for ep in range(epochs):
        order=rng.permutation(len(tr))
        for st in range(0,len(tr),batch):
            idx=order[st:st+batch];gs,rec=compute_grads(Xtr[idx],Ytr[idx],ps,pw,ti,mode,k);maxrec=max(maxrec,rec)
            total=math.sqrt(sum(float(np.sum(g*g)) for g in gs))
            if total>30:gs=[g*(30/total) for g in gs]
            opt.step(ps,gs)
    m=evalm(Xte,Yte,ps,ti);m.update(seed=seed,mode=mode,k=k,train_size=len(tr),test_size=len(te),max_canonical_reconstruction_error=maxrec);return m

def ci(x,r):
    x=np.asarray(x,float);z=[np.mean(r.choice(x,len(x),replace=True)) for _ in range(5000)]
    return [float(np.quantile(z,.025)),float(np.quantile(z,.975))]

def main():
    out=Path(os.environ.get("OUTDIR","results"));out.mkdir(parents=True,exist_ok=True);t0=time.time()
    X,Y,tasks,scaff,bad=load();ti,target=select_target(Y,tasks);seeds=[0,1,2,3,4];conds=[("coarse",1),("raw",2),("raw",4),("raw",8),("norm",8)];rows=[]
    print("TARGET_SELECTION",json.dumps(target),flush=True)
    for seed in seeds:
        for mode,k in conds:
            r=train(X,Y,tasks,scaff,ti,seed,mode,k);rows.append(r);print("DONE",SPLIT_MODE,seed,mode,k,r["target_auc"],r["peer_macro_auc"],flush=True)
    coarse={r["seed"]:r for r in rows if r["mode"]=="coarse"};rng=np.random.default_rng(20261111)
    summary={"dataset_rows":len(X),"bad_smiles":bad,"n_tasks":len(tasks),"target_selection":target,"split_mode":SPLIT_MODE,"conditions":{}}
    for mode,k in conds:
        rr=[r for r in rows if r["mode"]==mode and r["k"]==k]
        rec={z+"_mean":float(np.mean([r[z] for r in rr])) for z in ["macro_auc","target_auc","peer_macro_auc","dense_peer_macro_auc"]}
        rec["valid_auc_tasks"]=rr[0]["valid_auc_tasks"];rec["dense_auc_tasks"]=rr[0]["dense_auc_tasks"];rec["max_canonical_reconstruction_error"]=float(max(r["max_canonical_reconstruction_error"] for r in rr))
        if mode!="coarse":
            for z in ["macro_auc","target_auc","peer_macro_auc","dense_peer_macro_auc"]:
                d=[r[z]-coarse[r["seed"]][z] for r in rr];rec["delta_"+z+"_mean"]=float(np.mean(d));rec["delta_"+z+"_ci95"]=ci(d,rng)
        summary["conditions"][f"{mode}_k{k}"]=rec
    summary["runtime_sec"]=time.time()-t0
    (out/f"toxcast_{SPLIT_MODE}.json").write_text(json.dumps({"summary":summary,"rows":rows},indent=2))
    print("SUMMARY_JSON_BEGIN");print(json.dumps(summary,indent=2));print("SUMMARY_JSON_END")
if __name__=="__main__":main()
