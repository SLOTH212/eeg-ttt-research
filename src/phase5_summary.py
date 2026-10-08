from pathlib import Path
import json,numpy as np,pandas as pd
r=Path(__file__).resolve().parents[1];rows=[]
for phase in ["phase5","phase5_balanced"]:
 for f in (r/phase/"results").glob("*/evaluation.json"):
  d=json.loads(f.read_text());_,source,seed=d["name"].split("_")
  for v in d["results"]:rows.append(dict(source=source,seed=int(seed),subject=v["subject"],cohort=v["cohort"],method=v["method"],f1=v["macro_f1"]))
df=pd.DataFrame(rows);df.to_csv(r/"phase5/all_metrics.csv",index=False);a=df.groupby(["source","subject","cohort","method"]).f1.mean().reset_index();summary=[]
for (source,method),g in a.groupby(["source","method"]):
 seeds=df[(df.source==source)&(df.method==method)].seed.nunique()
 if seeds!=3:print("PENDING",source,method,seeds);continue
 base=a[(a.source==source)&(a.method=="frozen")].set_index("subject");delta=g.set_index("subject").f1-base.f1;summary.append(dict(source=source,method=method,f1=100*g.f1.mean(),delta_pp=100*delta.mean(),harm_rate=float((delta<-.01).mean()),n=len(g),seeds=seeds))
s=pd.DataFrame(summary);s.to_csv(r/"phase5/summary.csv",index=False);print(s.to_string(index=False,float_format=lambda x:f"{x:.3f}"))
# Pure source-state pulse gains, participant-averaged before model/seed aggregation.
pulses=[]
for f in (r/"phase5/results").glob("*/pulse.json"):
 for v in json.loads(f.read_text()):pulses.append(dict(model=f.parent.name,**v))
p=pd.DataFrame(pulses);p=p[p.h==4];p["oracle"]=p.gain.clip(lower=0);p["positive"]=p.gain>1e-4;p=p.groupby(["model","subject","lr"])[["gain","oracle","positive"]].mean().reset_index();p.to_csv(r/"phase5/source_pulse_participant.csv",index=False)
