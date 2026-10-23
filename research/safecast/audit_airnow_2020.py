#!/usr/bin/env python3
import json, os, urllib.request
from collections import defaultdict
import numpy as np
import pandas as pd
import pyreadr
from scipy.optimize import minimize_scalar

BASE="https://airfire-data-exports.s3.us-west-2.amazonaws.com/monitoring/v2/airnow/2020/data"
MONTHS=["202008","202009","202010"]
STATES={"CA","OR","WA"}
OUTDIR="research/safecast/results_airnow"

def official_nowcast(window):
    valid=[(i,float(v)) for i,v in enumerate(window) if v is not None and not pd.isna(v)]
    if sum(i<3 for i,_ in valid)<2 or not valid: return None
    vals=[v for _,v in valid]; mx=max(vals); mn=min(vals)
    w=.5 if mx<=0 else max(.5,mn/mx)
    return sum(w**i*v for i,v in valid)/sum(w**i for i,_ in valid)

def lower_bound(window):
    obs={i:float(v) for i,v in enumerate(window) if v is not None and not pd.isna(v)}
    if not obs: return 0.0
    if any(v<0 for v in obs.values()): return None
    M=[i for i in range(len(window)) if i not in obs]
    if not M: return official_nowcast(window)
    a=min(obs.values()); b=max(obs.values())
    c0=sum(.5**i*v for i,v in obs.items())/sum(.5**i for i in range(len(window)))
    if b<=0 or a/b<=.5: return c0
    hi=min(1.0,a/b)
    def F(w):
        return (sum(w**i*v for i,v in obs.items())+b*sum(w**(i+1) for i in M))/sum(w**i for i in range(len(window)))
    res=minimize_scalar(F,bounds=(.5,hi),method="bounded",options={"xatol":1e-11})
    return min(c0,F(.5),F(hi),float(res.fun))

def cat(x):
    if x is None: return None
    if x>=225.5:return "Hazardous"
    if x>=125.5:return "Very Unhealthy"
    if x>=55.5:return "Unhealthy"
    if x>=35.5:return "USG"
    if x>=9.1:return "Moderate"
    return "Good"

def read_rda(url, path):
    if not os.path.exists(path): urllib.request.urlretrieve(url,path)
    d=pyreadr.read_r(path)
    if not d: raise RuntimeError(f"No object in {path}")
    return next(iter(d.values()))

def main():
    os.makedirs(OUTDIR,exist_ok=True)
    totals=defaultdict(int); cases=[]; meta_rows=[]
    sitehour_inelig=set(); sitehour_notrobust=set()
    unique_by_severity={k:set() for k in ["USGplus","Unhealthyplus","VeryUnhealthyplus","Hazardous"]}
    strict_unique_by_severity={k:set() for k in ["USGplus","Unhealthyplus","VeryUnhealthyplus","Hazardous"]}
    strict_sitedays_by_severity={k:set() for k in ["USGplus","Unhealthyplus","VeryUnhealthyplus","Hazardous"]}
    strict_sites_by_severity={k:set() for k in ["USGplus","Unhealthyplus","VeryUnhealthyplus","Hazardous"]}
    for month in MONTHS:
        meta_url=f"{BASE}/airnow_PM2.5_{month}_meta.rda"
        data_url=f"{BASE}/airnow_PM2.5_{month}_data.rda"
        meta=read_rda(meta_url,f"/tmp/{month}_meta.rda")
        data=read_rda(data_url,f"/tmp/{month}_data.rda")
        data.columns=[str(c) for c in data.columns]
        if "datetime" not in data.columns:
            raise RuntimeError(f"datetime missing; cols={list(data.columns)[:10]}")
        meta=meta.copy()
        wanted=meta[(meta["stateCode"].astype(str).isin(STATES)) & (meta["countryCode"].astype(str)=="US")].copy()
        byid={str(r["deviceDeploymentID"]):r for _,r in wanted.iterrows()}
        ids=[i for i in byid if i in data.columns]
        totals["device_month_series"]+=len(ids)
        dt=pd.to_datetime(data["datetime"],utc=True)
        for did in ids:
            vals=pd.to_numeric(data[did],errors="coerce").to_numpy()
            r=byid[did]
            site=str(r.get("fullAQSID",r.get("AQSID",did)))
            state=str(r.get("stateCode",""))
            county=str(r.get("countyName",""))
            for k in range(11,len(vals)):
                wv=[None if pd.isna(vals[k-i]) else float(vals[k-i]) for i in range(12)]
                if all(v is None for v in wv): continue
                nrecent=sum(v is not None for v in wv[:3])
                nvalid=sum(v is not None for v in wv)
                nmiss=12-nvalid
                off=official_nowcast(wv)
                low=lower_bound(wv)
                if low is None:
                    totals["negative_obs_excluded"]+=1; continue
                timestamp=dt.iloc[k].isoformat()
                if nmiss: totals["windows_with_missing"]+=1
                if nrecent<2:
                    totals["official_ineligible"]+=1
                    sev=[]
                    if low>=35.5: totals["ineligible_certified_USGplus"]+=1; sev.append("USGplus")
                    if low>=55.5: totals["ineligible_certified_Unhealthyplus"]+=1; sev.append("Unhealthyplus")
                    if low>=125.5: totals["ineligible_certified_VeryUnhealthyplus"]+=1; sev.append("VeryUnhealthyplus")
                    if low>=225.5: totals["ineligible_certified_Hazardous"]+=1; sev.append("Hazardous")
                    for s in sev:
                        unique_by_severity[s].add((site,timestamp))
                        if site.isdigit() and len(site)==12:
                            strict_unique_by_severity[s].add((site,timestamp))
                            strict_sitedays_by_severity[s].add((site,timestamp[:10]))
                            strict_sites_by_severity[s].add(site)
                    if low>=35.5:
                        sitehour_inelig.add((site,timestamp))
                        cases.append({"month":month,"state":state,"county":county,"site":site,"device":did,
                                      "time":timestamp,"kind":"ineligible_but_certified","valid12":nvalid,
                                      "valid_recent3":nrecent,"lower":round(low,3),"lower_cat":cat(low),
                                      "official":None,"official_cat":None,
                                      "window":";".join("" if v is None else f"{v:.1f}" for v in wv)})
                else:
                    totals["official_eligible"]+=1
                    if nmiss: totals["eligible_with_missing"]+=1
                    if nmiss and off is not None and cat(off)!=cat(low):
                        totals["eligible_category_not_guaranteed"]+=1
                        sitehour_notrobust.add((site,timestamp))
                        cases.append({"month":month,"state":state,"county":county,"site":site,"device":did,
                                      "time":timestamp,"kind":"official_category_not_guaranteed","valid12":nvalid,
                                      "valid_recent3":nrecent,"lower":round(low,3),"lower_cat":cat(low),
                                      "official":round(off,3),"official_cat":cat(off),
                                      "window":";".join("" if v is None else f"{v:.1f}" for v in wv)})
                if wv[0] is None and nrecent>=2:
                    totals["current_missing_but_2of3_eligible"]+=1
    totals["unique_sitehours_ineligible_but_certified_USGplus"]=len(sitehour_inelig)
    totals["unique_sitehours_official_category_not_guaranteed"]=len(sitehour_notrobust)
    for s,v in unique_by_severity.items(): totals[f"unique_sitehours_ineligible_certified_{s}"]=len(v)
    for s,v in strict_unique_by_severity.items(): totals[f"strict_numeric_AQSID_unique_sitehours_ineligible_certified_{s}"]=len(v)
    for s,v in strict_sitedays_by_severity.items(): totals[f"strict_numeric_AQSID_unique_sitedays_ineligible_certified_{s}"]=len(v)
    for s,v in strict_sites_by_severity.items(): totals[f"strict_numeric_AQSID_unique_sites_ineligible_certified_{s}"]=len(v)
    cases.sort(key=lambda x:(x["lower"],x["time"]),reverse=True)
    out={
      "source":"USFS AirFire preprocessed AirNow monthly PM2.5 archive",
      "source_pattern":f"{BASE}/airnow_PM2.5_YYYYMM_[meta|data].rda",
      "months":MONTHS,"states":sorted(STATES),"totals":dict(totals),"top_cases":cases[:150],
      "notes":[
        "This is the AirNow feed rather than retrospectively validated AQS.",
        "SafeCast bound assumes physically nonnegative missing PM2.5 concentrations; windows containing observed negative values are excluded.",
        "Categories use current 2024+ PM2.5 concentration breakpoints.",
        "Unique site-hour counts deduplicate multiple AirNow device deployments sharing fullAQSID."
      ]
    }
    with open(f"{OUTDIR}/summary.json","w") as f: json.dump(out,f,indent=2)
    pd.DataFrame(cases).to_csv(f"{OUTDIR}/cases.csv",index=False)
    print(json.dumps(out["totals"],indent=2))
    print("cases",len(cases))

if __name__=="__main__": main()
