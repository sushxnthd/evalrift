#!/usr/bin/env python3
import csv, io, json, math, os, urllib.request, zipfile
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np
from scipy.optimize import minimize_scalar

URL = "https://aqs.epa.gov/aqsweb/airdata/hourly_88101_2020.zip"
OUTDIR = "research/safecast/results"
STATES = {"06":"California","41":"Oregon","53":"Washington"}
START = datetime(2020,8,1)
END = datetime(2020,11,1)
THRESHOLDS = [
    ("Moderate", 9.1),
    ("USG", 35.5),
    ("Unhealthy", 55.5),
    ("Very Unhealthy", 125.5),
    ("Hazardous", 225.5),
]

def official_nowcast(window):
    # window: newest first; None means missing. Negative raw values are allowed
    valid = [(i, v) for i, v in enumerate(window) if v is not None]
    if len([1 for i,v in valid if i < 3]) < 2 or not valid:
        return None
    vals = [v for _,v in valid]
    mx=max(vals); mn=min(vals)
    if mx <= 0:
        w=0.5
    else:
        w=max(0.5, mn/mx)
    num=sum((w**i)*v for i,v in valid)
    den=sum(w**i for i,_ in valid)
    return num/den

def safecast_lower_nonnegative(window):
    # Exact lower bound over all nonnegative completions of missing entries.
    obs={i:float(v) for i,v in enumerate(window) if v is not None}
    if not obs:
        return 0.0, "empty"
    if any(v < 0 for v in obs.values()):
        return None, "negative_observed"
    missing=[i for i in range(len(window)) if i not in obs]
    if not missing:
        return official_nowcast(window), "complete"
    a=min(obs.values()); b=max(obs.values())
    # w = 0.5 branch: all missing values can be 0.
    w=0.5
    den=sum(w**i for i in range(len(window)))
    c0=sum((w**i)*v for i,v in obs.items())/den
    if b <= 0 or a/b <= 0.5:
        return c0, "w0.5"
    hi=min(1.0,a/b)
    def F(w):
        numerator = sum((w**i)*v for i,v in obs.items())
        numerator += b*sum(w**(i+1) for i in missing)
        denominator = sum(w**i for i in range(len(window)))
        return numerator/denominator
    res=minimize_scalar(F,bounds=(0.5,hi),method="bounded",options={"xatol":1e-12})
    candidates=[(c0,0.5),(F(0.5),0.5),(F(hi),hi),(float(res.fun),float(res.x))]
    val,wstar=min(candidates,key=lambda z:z[0])
    return val, f"w={wstar:.8f}"

def category(c):
    if c is None: return None
    if c >= 225.5: return "Hazardous"
    if c >= 125.5: return "Very Unhealthy"
    if c >= 55.5: return "Unhealthy"
    if c >= 35.5: return "USG"
    if c >= 9.1: return "Moderate"
    return "Good"

def main():
    os.makedirs(OUTDIR,exist_ok=True)
    zpath="/tmp/hourly_88101_2020.zip"
    if not os.path.exists(zpath):
        urllib.request.urlretrieve(URL,zpath)
    series=defaultdict(dict)
    meta={}
    with zipfile.ZipFile(zpath) as z:
        csvname=next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(csvname) as raw:
            reader=csv.DictReader(io.TextIOWrapper(raw,encoding="utf-8-sig",errors="replace"))
            for row in reader:
                st=row.get("State Code","").zfill(2)
                if st not in STATES: continue
                try:
                    dt=datetime.strptime(row["Date Local"]+" "+row["Time Local"],"%Y-%m-%d %H:%M")
                    if not (START <= dt < END): continue
                    val=float(row["Sample Measurement"])
                except Exception:
                    continue
                # One continuous instrument stream = site + POC. Duplicate rows at same
                # hour are averaged below if encountered.
                key=(st,row.get("County Code","").zfill(3),row.get("Site Num","").zfill(4),row.get("POC",""))
                if dt in series[key]:
                    old=series[key][dt]
                    series[key][dt]=(old+val)/2
                else:
                    series[key][dt]=val
                meta[key]={
                    "state":STATES[st],
                    "county":row.get("County Name",""),
                    "site":f"{st}-{row.get('County Code','').zfill(3)}-{row.get('Site Num','').zfill(4)}",
                    "poc":row.get("POC",""),
                    "lat":row.get("Latitude",""),
                    "lon":row.get("Longitude",""),
                    "method":row.get("Method Name",""),
                }

    cases=[]
    totals=defaultdict(int)
    site_summaries=[]
    for key,data in series.items():
        if len(data)<100: continue
        # scan fixed study interval; window newest first
        t=START+timedelta(hours=11)
        local=defaultdict(int)
        while t<END:
            window=[data.get(t-timedelta(hours=i)) for i in range(12)]
            nvalid=sum(v is not None for v in window)
            nrecent=sum(v is not None for v in window[:3])
            nmiss=12-nvalid
            if nmiss:
                totals["windows_with_missing"]+=1; local["windows_with_missing"]+=1
            off=official_nowcast(window)
            low,mode=safecast_lower_nonnegative(window)
            if low is None:
                totals["negative_obs_excluded"]+=1
                t+=timedelta(hours=1); continue
            lowcat=category(low)
            offcat=category(off)
            if nrecent<2:
                totals["official_ineligible"]+=1; local["official_ineligible"]+=1
                # If a robust lower bound still proves a health-relevant category.
                for name,thr in THRESHOLDS[1:]:
                    if low>=thr:
                        totals[f"ineligible_but_guaranteed_{name}"]+=1
                        local[f"ineligible_but_guaranteed_{name}"]+=1
                if low>=35.5:
                    cases.append({**meta[key],"time":t.isoformat(),"kind":"official_ineligible_but_certified",
                                  "valid_12":nvalid,"valid_recent3":nrecent,"lower":round(low,3),
                                  "lower_category":lowcat,"official":None,"official_category":None,
                                  "mode":mode,"window":";".join("" if v is None else f"{v:.1f}" for v in window)})
            else:
                totals["official_eligible"]+=1
                if nmiss:
                    totals["eligible_with_missing"]+=1; local["eligible_with_missing"]+=1
                    if off is not None and offcat != lowcat:
                        totals["eligible_category_not_guaranteed"]+=1
                        local["eligible_category_not_guaranteed"]+=1
                        cases.append({**meta[key],"time":t.isoformat(),"kind":"official_category_not_robust",
                                      "valid_12":nvalid,"valid_recent3":nrecent,"lower":round(low,3),
                                      "lower_category":lowcat,"official":round(off,3),
                                      "official_category":offcat,"mode":mode,
                                      "window":";".join("" if v is None else f"{v:.1f}" for v in window)})
            if window[0] is None and nrecent>=2:
                totals["current_missing_but_2of3_eligible"]+=1
                local["current_missing_but_2of3_eligible"]+=1
            t+=timedelta(hours=1)
        site_summaries.append({**meta[key],"raw_hours":len(data),**local})

    cases.sort(key=lambda r:(r["lower"],r["time"]),reverse=True)
    site_summaries.sort(key=lambda r:r.get("official_ineligible",0),reverse=True)
    result={
        "source":URL,
        "study_window":[START.isoformat(),END.isoformat()],
        "states":STATES,
        "series_count":len(series),
        "totals":dict(sorted(totals.items())),
        "top_cases":cases[:100],
        "notes":[
            "Preliminary audit uses validated AQS hourly PM2.5 (parameter 88101), not the AirNow real-time feed.",
            "Series are site+POC streams; windows with observed negative concentrations are excluded from the nonnegative-completion theorem.",
            "Current 2024+ PM2.5 concentration category thresholds are used for descriptive category labels.",
            "No statistical imputation is used in the SafeCast lower certificate."
        ]
    }
    with open(f"{OUTDIR}/summary.json","w") as f: json.dump(result,f,indent=2)
    if cases:
        with open(f"{OUTDIR}/cases.csv","w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(cases[0].keys())); w.writeheader(); w.writerows(cases)
    if site_summaries:
        fields=sorted({k for r in site_summaries for k in r})
        with open(f"{OUTDIR}/sites.csv","w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(site_summaries)
    print(json.dumps(result["totals"],indent=2))
    print("series",len(series),"cases",len(cases))

if __name__=="__main__":
    main()
