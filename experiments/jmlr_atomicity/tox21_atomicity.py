#!/usr/bin/env python3
"""
Tox21 measured-atomicity experiment for Project Zero.

Data: public Tox21 CSV (12 binary tasks, missing labels allowed).
Features: RDKit Morgan fingerprints.
Model: shared tanh bottleneck + 12 task-specific logistic heads.

Causal intervention:
- Task heads always receive parent semantic-task gradients.
- Only the shared trunk sees either:
  coarse parent task gradient,
  naive child-identity refinement, or
  a norm-controlled refined gradient.
- The selected target is fixed by TARGET_TASK before training.

Robustness modes:
- SPLIT_MODE=random: paired random 80/20 split per seed.
- SPLIT_MODE=scaffold: deterministic Bemis-Murcko scaffold 80/20 split,
  with seeds varying initialization and minibatch order only.

This script prioritizes causal isolation over SOTA accuracy.
"""
import csv, io, json, math, os, time, urllib.request
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import roc_auc_score

RDLogger.DisableLog("rdApp.*")

TASKS = [
    "NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase","NR-ER","NR-ER-LBD",
    "NR-PPAR-gamma","SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"
]
TARGET = os.environ.get("TARGET_TASK","SR-ARE")
if TARGET not in TASKS:
    raise ValueError(f"Unknown TARGET_TASK={TARGET}")
TARGET_IDX = TASKS.index(TARGET)
SPLIT_MODE = os.environ.get("SPLIT_MODE","random").lower()
if SPLIT_MODE not in ("random","scaffold"):
    raise ValueError(f"Unknown SPLIT_MODE={SPLIT_MODE}")

DATA_URL = "https://raw.githubusercontent.com/cumbof/hdlib/main/examples/tox21/tox21.csv"

def sigmoid(x):
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))

def load_tox21(nbits=256):
    raw = urllib.request.urlopen(DATA_URL, timeout=60).read().decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(raw)))
    X, Y, mol_ids, smiles, scaffolds = [], [], [], [], []
    bad = 0
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=nbits)
    for r in rows:
        smi=r["smiles"]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            bad += 1
            continue
        fp = gen.GetFingerprintAsNumPy(mol).astype(np.float32)
        y = np.full(len(TASKS), np.nan, dtype=np.float32)
        for j,t in enumerate(TASKS):
            v = r[t].strip()
            if v in ("0","1"):
                y[j] = float(v)
        try:
            scaf=MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            scaf=""
        X.append(fp); Y.append(y); mol_ids.append(r["mol_id"])
        smiles.append(smi); scaffolds.append(scaf)
    return (np.asarray(X,np.float32), np.asarray(Y,np.float32),
            mol_ids, smiles, scaffolds, bad)

def split_indices(n, scaffolds, seed, mode):
    if mode=="random":
        rng=np.random.default_rng(seed)
        perm=rng.permutation(n)
        ntr=int(.8*n)
        return perm[:ntr],perm[ntr:]

    groups={}
    for i,s in enumerate(scaffolds):
        groups.setdefault(s,[]).append(i)
    ordered=sorted(groups.items(), key=lambda kv:(-len(kv[1]),kv[0]))
    target=int(.8*n)
    tr=[]; te=[]
    for _,idxs in ordered:
        if len(tr)+len(idxs) <= target or len(tr)==0:
            tr.extend(idxs)
        else:
            te.extend(idxs)
    if len(te)==0:
        # deterministic safety fallback; should not occur for Tox21
        cut=int(.8*n)
        tr=list(range(cut)); te=list(range(cut,n))
    return np.asarray(tr,dtype=int),np.asarray(te,dtype=int)

class Adam:
    def __init__(self, params, lr=2e-3, b1=.9, b2=.999, eps=1e-8):
        self.lr=lr; self.b1=b1; self.b2=b2; self.eps=eps; self.t=0
        self.m=[np.zeros_like(p) for p in params]
        self.v=[np.zeros_like(p) for p in params]
    def step(self, params, grads):
        self.t += 1
        for i,(p,g) in enumerate(zip(params,grads)):
            self.m[i] = self.b1*self.m[i] + (1-self.b1)*g
            self.v[i] = self.b2*self.v[i] + (1-self.b2)*(g*g)
            mh = self.m[i]/(1-self.b1**self.t)
            vh = self.v[i]/(1-self.b2**self.t)
            p -= self.lr*mh/(np.sqrt(vh)+self.eps)

def init_model(seed, d, h, T):
    rng=np.random.default_rng(seed)
    W=(rng.normal(size=(d,h))*math.sqrt(2/d)).astype(np.float32)
    b=np.zeros(h,np.float32)
    V=(rng.normal(size=(h,T))*math.sqrt(2/h)).astype(np.float32)
    c=np.zeros(T,np.float32)
    return [W,b,V,c]

def pos_weights(Ytr):
    out=np.ones(Ytr.shape[1],np.float32)
    for t in range(Ytr.shape[1]):
        v=Ytr[:,t]
        obs=~np.isnan(v)
        p=np.sum(v[obs]==1); n=np.sum(v[obs]==0)
        if p>0: out[t]=min(float(n/p),20.0)
    return out

def task_dlogit(logits, y, posw):
    p=sigmoid(logits)
    w=np.where(y>0.5,posw,1.0).astype(np.float32)
    return (p-y)*w

def shared_and_head_grads(Xb,Yb,params,posw,mode="coarse",k=1):
    W,b,V,c=params
    A=Xb@W+b
    H=np.tanh(A)
    Z=H@V+c
    B=len(Xb); h=W.shape[1]; T=Yb.shape[1]
    dH_coarse=np.zeros((B,h),np.float32)
    dH_rep=np.zeros((B,h),np.float32)
    gV=np.zeros_like(V); gc=np.zeros_like(c)

    target_obs_idx=None
    target_raw=None
    target_parent_n=0

    for t in range(T):
        obs=np.flatnonzero(~np.isnan(Yb[:,t]))
        n=len(obs)
        if n==0:
            continue
        y=Yb[obs,t]
        raw=task_dlogit(Z[obs,t],y,posw[t]).astype(np.float32)
        dl=raw/n
        gV[:,t]=H[obs].T@dl
        gc[t]=np.sum(dl)
        contrib=dl[:,None]*V[:,t][None,:]
        dH_coarse[obs]+=contrib
        if t!=TARGET_IDX:
            dH_rep[obs]+=contrib
        else:
            target_obs_idx=obs; target_raw=raw; target_parent_n=n

    if target_obs_idx is not None:
        if mode=="coarse":
            dl=target_raw/target_parent_n
            dH_rep[target_obs_idx]+=dl[:,None]*V[:,TARGET_IDX][None,:]
        else:
            children=np.array_split(np.arange(target_parent_n), min(k,target_parent_n))
            for child in children:
                if len(child)==0: continue
                rows=target_obs_idx[child]
                dl=target_raw[child]/len(child)
                dH_rep[rows]+=dl[:,None]*V[:,TARGET_IDX][None,:]

    dA_coarse=dH_coarse*(1-H*H)
    dA_rep=dH_rep*(1-H*H)
    gWc=Xb.T@dA_coarse; gbc=np.sum(dA_coarse,axis=0)
    gWr=Xb.T@dA_rep; gbr=np.sum(dA_rep,axis=0)

    if mode=="norm":
        nc=math.sqrt(float(np.sum(gWc*gWc)+np.sum(gbc*gbc)))
        nr=math.sqrt(float(np.sum(gWr*gWr)+np.sum(gbr*gbr)))
        scale=nc/(nr+1e-12)
        gWr*=scale; gbr*=scale

    recon_err=0.0
    if target_obs_idx is not None and target_parent_n>0:
        children=np.array_split(np.arange(target_parent_n), min(k,target_parent_n))
        dHcan=np.zeros((B,h),np.float32)
        for child in children:
            if len(child)==0: continue
            rows=target_obs_idx[child]
            mass=len(child)/target_parent_n
            dl=(target_raw[child]/len(child))*mass
            dHcan[rows]+=dl[:,None]*V[:,TARGET_IDX][None,:]
        dHpar=np.zeros((B,h),np.float32)
        dl=target_raw/target_parent_n
        dHpar[target_obs_idx]+=dl[:,None]*V[:,TARGET_IDX][None,:]
        recon_err=float(np.linalg.norm(dHcan-dHpar)/(np.linalg.norm(dHpar)+1e-12))

    return [gWr,gbr,gV,gc], recon_err

def evaluate(X,Y,params):
    W,b,V,c=params
    H=np.tanh(X@W+b); Z=H@V+c; P=sigmoid(Z)
    aucs=[]
    counts=[]
    for t in range(Y.shape[1]):
        obs=~np.isnan(Y[:,t]); yy=Y[obs,t]
        counts.append(int(np.sum(obs)))
        if len(np.unique(yy))<2:
            aucs.append(np.nan)
        else:
            aucs.append(float(roc_auc_score(yy,P[obs,t])))
    a=np.asarray(aucs,float)
    return {
        "aucs":aucs,
        "test_label_counts":counts,
        "macro_auc":float(np.nanmean(a)),
        "target_auc":float(a[TARGET_IDX]),
        "peer_macro_auc":float(np.nanmean(np.delete(a,TARGET_IDX)))
    }

def train_one(X,Y,scaffolds,seed,mode,k,epochs=18,batch=768,h=24):
    tr,te=split_indices(len(X),scaffolds,seed,SPLIT_MODE)
    Xtr,Ytr=X[tr],Y[tr]; Xte,Yte=X[te],Y[te]
    pw=pos_weights(Ytr)
    params=init_model(seed+1000,X.shape[1],h,Y.shape[1])
    opt=Adam(params,lr=2e-3)
    rng=np.random.default_rng(seed+777)
    max_recon=0.0
    for ep in range(epochs):
        order=rng.permutation(len(tr))
        for st in range(0,len(tr),batch):
            idx=order[st:st+batch]
            grads,recon=shared_and_head_grads(Xtr[idx],Ytr[idx],params,pw,mode=mode,k=k)
            max_recon=max(max_recon,recon)
            total=math.sqrt(sum(float(np.sum(g*g)) for g in grads))
            if total>20.0:
                grads=[g*(20.0/total) for g in grads]
            opt.step(params,grads)
    met=evaluate(Xte,Yte,params)
    met.update({"seed":seed,"mode":mode,"k":k,
                "train_size":int(len(tr)),"test_size":int(len(te)),
                "max_canonical_reconstruction_error":max_recon})
    return met

def ci95(x, rng):
    x=np.asarray(x,float); means=[]
    for _ in range(5000):
        means.append(np.mean(rng.choice(x,size=len(x),replace=True)))
    return [float(np.quantile(means,.025)),float(np.quantile(means,.975))]

def main():
    outdir=Path(os.environ.get("OUTDIR","results"))
    outdir.mkdir(parents=True,exist_ok=True)
    t0=time.time()
    X,Y,mol_ids,smiles,scaffolds,bad=load_tox21(256)
    seeds=[0,1,2,3,4]
    conditions=[("coarse",1),("raw",2),("raw",4),("raw",8),("norm",8)]
    rows=[]
    for seed in seeds:
        for mode,k in conditions:
            rows.append(train_one(X,Y,scaffolds,seed,mode,k))
            print("DONE",TARGET,SPLIT_MODE,seed,mode,k,rows[-1]["macro_auc"],flush=True)

    summary={"dataset_rows":int(len(X)),"bad_smiles":bad,"tasks":TASKS,
             "target":TARGET,"split_mode":SPLIT_MODE,
             "conditions":{},"runtime_sec":time.time()-t0}
    rng=np.random.default_rng(20261111+TARGET_IDX)
    coarse={r["seed"]:r for r in rows if r["mode"]=="coarse"}

    for mode,k in conditions:
        key=f"{mode}_k{k}"
        rr=[r for r in rows if r["mode"]==mode and r["k"]==k]
        vals=np.array([r["macro_auc"] for r in rr])
        tar=np.array([r["target_auc"] for r in rr])
        peer=np.array([r["peer_macro_auc"] for r in rr])
        rec={
            "macro_auc_mean":float(np.mean(vals)),
            "target_auc_mean":float(np.mean(tar)),
            "peer_macro_auc_mean":float(np.mean(peer)),
            "train_size":rr[0]["train_size"],"test_size":rr[0]["test_size"],
            "max_canonical_reconstruction_error":float(max(r["max_canonical_reconstruction_error"] for r in rr))
        }
        if mode!="coarse":
            dm=np.array([r["macro_auc"]-coarse[r["seed"]]["macro_auc"] for r in rr])
            dt=np.array([r["target_auc"]-coarse[r["seed"]]["target_auc"] for r in rr])
            dp=np.array([r["peer_macro_auc"]-coarse[r["seed"]]["peer_macro_auc"] for r in rr])
            rec.update({
                "delta_macro_auc_mean":float(np.mean(dm)),
                "delta_macro_auc_ci95":ci95(dm,rng),
                "delta_target_auc_mean":float(np.mean(dt)),
                "delta_target_auc_ci95":ci95(dt,rng),
                "delta_peer_macro_auc_mean":float(np.mean(dp)),
                "delta_peer_macro_auc_ci95":ci95(dp,rng)
            })
        summary["conditions"][key]=rec

    payload={"summary":summary,"rows":rows}
    stem=f"tox21_{TARGET}_{SPLIT_MODE}".replace("/","_")
    (outdir/f"{stem}.json").write_text(json.dumps(payload,indent=2))
    with open(outdir/f"{stem}.csv","w",newline="") as f:
        fields=["seed","mode","k","macro_auc","target_auc","peer_macro_auc","train_size","test_size","max_canonical_reconstruction_error"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({x:r[x] for x in fields})
    print("SUMMARY_JSON_BEGIN")
    print(json.dumps(summary,indent=2))
    print("SUMMARY_JSON_END")

if __name__=="__main__":
    main()
