#!/usr/bin/env python3
"""
SIDER measured-atomicity experiment for Project Zero.

Dataset: MoleculeNet SIDER, 27 binary side-effect tasks.
Features: RDKit Morgan fingerprints.
Model: shared tanh bottleneck + 27 task-specific logistic heads.

Only the shared trunk sees the representation intervention. Task-specific
head updates always use the parent semantic task loss.

TARGET_TASK and SPLIT_MODE are fixed by the workflow before training.
"""
import csv, io, json, math, os, re, time, urllib.request
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import roc_auc_score

RDLogger.DisableLog("rdApp.*")
DATA_URL="https://raw.githubusercontent.com/tencent-ailab/grover/main/exampledata/finetune/sider.csv"
TARGET=os.environ.get("TARGET_TASK","Reproductive system and breast disorders")
SPLIT_MODE=os.environ.get("SPLIT_MODE","random").lower()
if SPLIT_MODE not in ("random","scaffold"):
    raise ValueError(SPLIT_MODE)

def sigmoid(x):
    return 1.0/(1.0+np.exp(-np.clip(x,-30,30)))

def load_sider(nbits=256):
    raw=urllib.request.urlopen(DATA_URL,timeout=60).read().decode("utf-8")
    rows=list(csv.DictReader(io.StringIO(raw)))
    tasks=[x for x in rows[0].keys() if x!="smiles"]
    if TARGET not in tasks:
        raise ValueError(f"Unknown target {TARGET}")
    gen=AllChem.GetMorganGenerator(radius=2,fpSize=nbits)
    X=[];Y=[];scaffolds=[];bad=0
    for r in rows:
        mol=Chem.MolFromSmiles(r["smiles"])
        if mol is None:
            bad+=1; continue
        X.append(gen.GetFingerprintAsNumPy(mol).astype(np.float32))
        Y.append([float(r[t]) for t in tasks])
        try:
            scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol,includeChirality=False))
        except Exception:
            scaffolds.append("")
    return np.asarray(X,np.float32),np.asarray(Y,np.float32),tasks,scaffolds,bad

def split_indices(n,scaffolds,seed):
    if SPLIT_MODE=="random":
        p=np.random.default_rng(seed).permutation(n); c=int(.8*n)
        return p[:c],p[c:]
    groups={}
    for i,s in enumerate(scaffolds): groups.setdefault(s,[]).append(i)
    ordered=sorted(groups.items(),key=lambda kv:(-len(kv[1]),kv[0]))
    target=int(.8*n);tr=[];te=[]
    for _,idx in ordered:
        if len(tr)+len(idx)<=target or len(tr)==0: tr.extend(idx)
        else: te.extend(idx)
    return np.asarray(tr,int),np.asarray(te,int)

class Adam:
    def __init__(self,params,lr=2e-3):
        self.lr=lr;self.t=0;self.m=[np.zeros_like(p) for p in params];self.v=[np.zeros_like(p) for p in params]
    def step(self,params,grads):
        self.t+=1
        for i,(p,g) in enumerate(zip(params,grads)):
            self.m[i]=.9*self.m[i]+.1*g
            self.v[i]=.999*self.v[i]+.001*g*g
            mh=self.m[i]/(1-.9**self.t);vh=self.v[i]/(1-.999**self.t)
            p-=self.lr*mh/(np.sqrt(vh)+1e-8)

def init(seed,d,h,T):
    r=np.random.default_rng(seed)
    return [(r.normal(size=(d,h))*math.sqrt(2/d)).astype(np.float32),
            np.zeros(h,np.float32),
            (r.normal(size=(h,T))*math.sqrt(2/h)).astype(np.float32),
            np.zeros(T,np.float32)]

def pos_weights(Y):
    p=np.sum(Y==1,axis=0);n=np.sum(Y==0,axis=0)
    return np.minimum(n/np.maximum(p,1),20).astype(np.float32)

def grads(X,Y,params,pw,target_idx,mode,k):
    W,b,V,c=params;H=np.tanh(X@W+b);Z=H@V+c;B=len(X);T=Y.shape[1];h=W.shape[1]
    dHc=np.zeros((B,h),np.float32);dHr=np.zeros((B,h),np.float32)
    gV=np.zeros_like(V);gc=np.zeros_like(c)
    target_raw=None
    for t in range(T):
        raw=(sigmoid(Z[:,t])-Y[:,t])*np.where(Y[:,t]>.5,pw[t],1).astype(np.float32)
        dl=raw/B
        gV[:,t]=H.T@dl;gc[t]=dl.sum()
        con=dl[:,None]*V[:,t][None,:]
        dHc+=con
        if t!=target_idx:dHr+=con
        else:target_raw=raw
    if mode=="coarse":
        dl=target_raw/B;dHr+=dl[:,None]*V[:,target_idx][None,:]
    else:
        for child in np.array_split(np.arange(B),min(k,B)):
            dl=target_raw[child]/len(child)
            dHr[child]+=dl[:,None]*V[:,target_idx][None,:]
    dAc=dHc*(1-H*H);dAr=dHr*(1-H*H)
    gWc=X.T@dAc;gbc=dAc.sum(0);gWr=X.T@dAr;gbr=dAr.sum(0)
    if mode=="norm":
        nc=math.sqrt(float(np.sum(gWc*gWc)+np.sum(gbc*gbc)))
        nr=math.sqrt(float(np.sum(gWr*gWr)+np.sum(gbr*gbr)))
        s=nc/(nr+1e-12);gWr*=s;gbr*=s
    # canonical reconstruction check for target gradient
    dHcan=np.zeros((B,h),np.float32)
    for child in np.array_split(np.arange(B),min(k,B)):
        mass=len(child)/B
        dl=(target_raw[child]/len(child))*mass
        dHcan[child]+=dl[:,None]*V[:,target_idx][None,:]
    dHpar=(target_raw/B)[:,None]*V[:,target_idx][None,:]
    rec=float(np.linalg.norm(dHcan-dHpar)/(np.linalg.norm(dHpar)+1e-12))
    return [gWr,gbr,gV,gc],rec

def evaluate(X,Y,params,target_idx):
    W,b,V,c=params;P=sigmoid(np.tanh(X@W+b)@V+c)
    auc=np.array([roc_auc_score(Y[:,t],P[:,t]) if len(np.unique(Y[:,t]))>1 else np.nan for t in range(Y.shape[1])])
    return {"aucs":auc.tolist(),"macro_auc":float(np.nanmean(auc)),
            "target_auc":float(auc[target_idx]),
            "peer_macro_auc":float(np.nanmean(np.delete(auc,target_idx)))}

def train(X,Y,tasks,scaffolds,seed,mode,k,epochs=30,batch=256,h=32):
    ti=tasks.index(TARGET);tr,te=split_indices(len(X),scaffolds,seed)
    Xtr,Ytr=X[tr],Y[tr];Xte,Yte=X[te],Y[te];pw=pos_weights(Ytr)
    params=init(seed+2000,X.shape[1],h,len(tasks));opt=Adam(params,2e-3);rng=np.random.default_rng(seed+555)
    maxrec=0
    for ep in range(epochs):
        order=rng.permutation(len(tr))
        for st in range(0,len(tr),batch):
            idx=order[st:st+batch]
            gg,rec=grads(Xtr[idx],Ytr[idx],params,pw,ti,mode,k);maxrec=max(maxrec,rec)
            total=math.sqrt(sum(float(np.sum(g*g)) for g in gg))
            if total>25:gg=[g*(25/total) for g in gg]
            opt.step(params,gg)
    m=evaluate(Xte,Yte,params,ti)
    m.update(seed=seed,mode=mode,k=k,train_size=int(len(tr)),test_size=int(len(te)),max_canonical_reconstruction_error=maxrec)
    return m

def ci95(x,rng):
    x=np.asarray(x,float)
    z=[np.mean(rng.choice(x,len(x),replace=True)) for _ in range(5000)]
    return [float(np.quantile(z,.025)),float(np.quantile(z,.975))]

def main():
    out=Path(os.environ.get("OUTDIR","results"));out.mkdir(parents=True,exist_ok=True);t0=time.time()
    X,Y,tasks,scaf,bad=load_sider();seeds=[0,1,2,3,4]
    cond=[("coarse",1),("raw",2),("raw",4),("raw",8),("norm",8)]
    rows=[]
    for seed in seeds:
        for mode,k in cond:
            r=train(X,Y,tasks,scaf,seed,mode,k);rows.append(r)
            print("DONE",TARGET,SPLIT_MODE,seed,mode,k,r["macro_auc"],flush=True)
    coarse={r["seed"]:r for r in rows if r["mode"]=="coarse"}
    rng=np.random.default_rng(20261111+tasks.index(TARGET));summary={"dataset_rows":len(X),"bad_smiles":bad,"tasks":tasks,"target":TARGET,"split_mode":SPLIT_MODE,"conditions":{}}
    for mode,k in cond:
        rr=[r for r in rows if r["mode"]==mode and r["k"]==k]
        rec={"macro_auc_mean":float(np.mean([r["macro_auc"] for r in rr])),
             "target_auc_mean":float(np.mean([r["target_auc"] for r in rr])),
             "peer_macro_auc_mean":float(np.mean([r["peer_macro_auc"] for r in rr])),
             "train_size":rr[0]["train_size"],"test_size":rr[0]["test_size"],
             "max_canonical_reconstruction_error":float(max(r["max_canonical_reconstruction_error"] for r in rr))}
        if mode!="coarse":
            dm=[r["macro_auc"]-coarse[r["seed"]]["macro_auc"] for r in rr]
            dt=[r["target_auc"]-coarse[r["seed"]]["target_auc"] for r in rr]
            dp=[r["peer_macro_auc"]-coarse[r["seed"]]["peer_macro_auc"] for r in rr]
            rec.update(delta_macro_auc_mean=float(np.mean(dm)),delta_macro_auc_ci95=ci95(dm,rng),
                       delta_target_auc_mean=float(np.mean(dt)),delta_target_auc_ci95=ci95(dt,rng),
                       delta_peer_macro_auc_mean=float(np.mean(dp)),delta_peer_macro_auc_ci95=ci95(dp,rng))
        summary["conditions"][f"{mode}_k{k}"]=rec
    summary["runtime_sec"]=time.time()-t0
    stem=re.sub(r"[^A-Za-z0-9]+","_",f"sider_{TARGET}_{SPLIT_MODE}").strip("_")
    (out/f"{stem}.json").write_text(json.dumps({"summary":summary,"rows":rows},indent=2))
    print("SUMMARY_JSON_BEGIN");print(json.dumps(summary,indent=2));print("SUMMARY_JSON_END")

if __name__=="__main__":main()
