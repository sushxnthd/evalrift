#!/usr/bin/env python3
import json, math, os
from collections import defaultdict
import pandas as pd
from audit_airnow_2020 import BASE, MONTHS, STATES, read_rda, official_nowcast, lower_bound

OUTDIR="research/safecast/results_airnow"

def hav_km(a,b):
    lat1,lon1=a; lat2,lon2=b
    R=6371.0088
    p1=math.radians(lat1); p2=math.radians(lat2)
    dp=math.radians(lat2-lat1); dl=math.radians(lon2-lon1)
    h=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(min(1,math.sqrt(h)))

def main():
    eligible=defaultdict(dict)   # time -> site -> point NowCast concentration
    coords={}
    events={}                    # (site,time) -> robust lower
    for month in MONTHS:
        meta=read_rda(f"{BASE}/airnow_PM2.5_{month}_meta.rda",f"/tmp/{month}_meta.rda")
        data=read_rda(f"{BASE}/airnow_PM2.5_{month}_data.rda",f"/tmp/{month}_data.rda")
        data.columns=[str(c) for c in data.columns]
        wanted=meta[(meta["stateCode"].astype(str).isin(STATES)) & (meta["countryCode"].astype(str)=="US")].copy()
        byid={str(r["deviceDeploymentID"]):r for _,r in wanted.iterrows()}
        ids=[i for i in byid if i in data.columns]
        dt=pd.to_datetime(data["datetime"],utc=True)
        for did in ids:
            r=byid[did]
            site=str(r.get("fullAQSID",r.get("AQSID",did)))
            mt=str(r.get("airnow_monitorType",""))
            if not (site.isdigit() and len(site)==12 and mt.lower()=="permanent"): continue
            try:
                lat=float(r.get("latitude")); lon=float(r.get("longitude"))
                if not (math.isfinite(lat) and math.isfinite(lon)): continue
            except Exception: continue
            coords[site]=(lat,lon)
            vals=pd.to_numeric(data[did],errors="coerce").to_numpy()
            for k in range(11,len(vals)):
                wv=[None if pd.isna(vals[k-i]) else float(vals[k-i]) for i in range(12)]
                obs=[v for v in wv if v is not None]
                if not obs or min(obs)<0 or max(obs)>1000: continue
                ts=dt.iloc[k].isoformat()
                off=official_nowcast(wv)
                if off is not None:
                    eligible[ts][site]=max(eligible[ts].get(site,-1),float(off))
                elif sum(v is not None for v in wv[:3])<2:
                    low=lower_bound(wv)
                    if low is not None and low>=55.5:
                        key=(site,ts)
                        events[key]=max(events.get(key,-1),float(low))
    stats=defaultdict(int); detail=[]
    for (site,ts),low in events.items():
        if site not in coords: continue
        near=[]
        for s,v in eligible.get(ts,{}).items():
            if s==site or s not in coords: continue
            d=hav_km(coords[site],coords[s])
            near.append((d,s,v))
        near.sort()
        stats["events"]+=1
        if not near:
            stats["no_eligible_monitor_any_distance_same_hour"]+=1
            nearest=None
        else:
            nearest=near[0]
        for radius in (25,50,100):
            n=[x for x in near if x[0]<=radius]
            if not n:
                stats[f"no_eligible_within_{radius}km"]+=1
            else:
                mx=max(v for _,_,v in n)
                if mx<55.5: stats[f"all_eligible_within_{radius}km_below_Unhealthy"]+=1
                if low>mx: stats[f"robust_lower_exceeds_all_eligible_within_{radius}km"]+=1
        detail.append({
          "site":site,"time":ts,"lower":round(low,3),
          "nearest_eligible_km":None if nearest is None else round(nearest[0],2),
          "nearest_eligible_site":None if nearest is None else nearest[1],
          "nearest_eligible_nowcast":None if nearest is None else round(nearest[2],3)
        })
    detail.sort(key=lambda x:(float("inf") if x["nearest_eligible_km"] is None else x["nearest_eligible_km"]),reverse=True)
    out={"stats":dict(stats),"examples":detail[:100],
         "notes":["Events restricted to permanent 12-digit numeric site IDs, observed PM2.5 in [0,1000], robust lower >=55.5.",
                  "Distance is straight-line between monitor coordinates; nearby monitors are not assumed interchangeable.",
                  "This is a coverage stress test, not a claim about reporting-area AQI."]}
    os.makedirs(OUTDIR,exist_ok=True)
    with open(f"{OUTDIR}/spatial_summary.json","w") as f: json.dump(out,f,indent=2)
    print(json.dumps(out["stats"],indent=2))

if __name__=="__main__": main()
