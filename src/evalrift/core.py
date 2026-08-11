from __future__ import annotations
from dataclasses import dataclass
import hashlib, random
from typing import Callable

@dataclass(frozen=True)
class MCQ:
    question:str
    choices:tuple[str,...]
    answer:int

def reorder(item:MCQ,seed:int)->MCQ:
    idx=list(range(len(item.choices))); random.Random(seed).shuffle(idx)
    choices=tuple(item.choices[i] for i in idx); answer=idx.index(item.answer)
    return MCQ(item.question,choices,answer)

def whitespace(item:MCQ,seed:int)->MCQ:
    pad=' '*(1+seed%3); return MCQ(pad+item.question+pad,item.choices,item.answer)

def stable_seed(text:str)->int:return int(hashlib.sha1(text.encode()).hexdigest()[:8],16)

def accuracy(model:Callable[[MCQ],int],items:list[MCQ])->float:
    return sum(model(x)==x.answer for x in items)/max(1,len(items))

def robustness(model:Callable[[MCQ],int],items:list[MCQ],rounds:int=20)->dict:
    base=accuracy(model,items); scores=[]
    for r in range(rounds):
        pert=[reorder(x,stable_seed(x.question)+r) for x in items]
        scores.append(accuracy(model,pert))
    mean=sum(scores)/len(scores) if scores else base
    return {'base_accuracy':base,'perturbed_mean':mean,'worst_accuracy':min(scores or [base]),'max_drop':base-min(scores or [base])}


def relabel(item:MCQ,labels:tuple[str,...]=('A','B','C','D')) -> dict:
    if len(labels)<len(item.choices): raise ValueError('not enough labels')
    return {'question':item.question,'choices':{labels[i]:c for i,c in enumerate(item.choices)},'answer':labels[item.answer]}

def ranking(models:dict[str,Callable[[MCQ],int]],items:list[MCQ])->list[str]:
    return sorted(models,key=lambda name:accuracy(models[name],items),reverse=True)

def ranking_instability(models:dict[str,Callable[[MCQ],int]],items:list[MCQ],rounds:int=20)->float:
    base=ranking(models,items); changes=0
    for r in range(rounds):
        pert=[reorder(x,stable_seed(x.question)+r) for x in items]
        changes += ranking(models,pert)!=base
    return changes/max(1,rounds)
