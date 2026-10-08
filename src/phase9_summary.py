from pathlib import Path
import json,pandas as pd,numpy as np
r=Path(__file__).resolve().parents[1];out=r/"phase9";rows=[]
for folder,pattern in [(out,"*/evaluation.json"),(out/"fresh","*_s[0-9][0-9].json")]:
 for f in folder.glob(pattern):
  d=json.loads(f.read_text());name=d.get("case",f.parent.name)
  if name.startswith("context11"):
   _,source,seed=name.split("_");kind="context11_"+source
  else:kind,fold,seed=name.split("_")
  for v in d["results"]:rows.append(dict(kind=kind,case=name,seed=int(seed),cohort=d.get("cohort","development"),subject=v["subject"],method=v["method"],f1=v["macro_f1"]))
df=pd.DataFrame(rows);df.to_csv(out/"all_metrics.csv",index=False);a=df.groupby(["kind","cohort","subject","method"]).f1.mean().reset_index();a.to_csv(out/"participant_metrics.csv",index=False);summary=[]
for (kind,cohort,method),g in a.groupby(["kind","cohort","method"]):
 base=a[(a.kind==kind)&(a.cohort==cohort)&(a.method=="frozen")].set_index("subject").f1;delta=g.set_index("subject").f1-base;ref=a[(a.kind==kind)&(a.cohort==cohort)&(a.method=="base_only_head")].set_index("subject").f1;extra=g.set_index("subject").f1-ref
 summary.append(dict(kind=kind,cohort=cohort,method=method,n=len(g),f1=100*g.f1.mean(),delta_pp=100*delta.mean(),harm_rate=float((delta<-.01).mean()),worst_delta_pp=float(100*delta.min()),delta_vs_base_only_pp=float(100*extra.mean())))
s=pd.DataFrame(summary);s.to_csv(out/"summary.csv",index=False);print(s[(s.cohort!="development")&(s.method!="frozen")].to_string(index=False,float_format=lambda x:f"{x:.3f}"))
