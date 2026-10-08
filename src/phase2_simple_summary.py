from pathlib import Path
import json,numpy as np,pandas as pd,hashlib
r=Path(__file__).resolve().parents[1];out=r/"phase2";df=pd.read_csv(out/"fresh_all.csv");rows=[]
for f in (out/"fresh_simple").glob("*_s*.json"):
 d=json.loads(f.read_text());kind,fold,seed=d["name"].split("_")
 for v in d["results"]:rows.append(dict(kind=kind,fold=fold,seed=int(seed),subject=v["subject"],method=v["method"],f1=v["macro_f1"],transition=v["transition_f1"],ce=v["ce"]))
df=pd.concat([df,pd.DataFrame(rows)],ignore_index=True);a=df.groupby(["kind","subject","method"])[["f1","transition","ce"]].mean().reset_index();counts=df.groupby(["kind","subject","method"]).size();a["replicates"]=[counts.loc[(v.kind,v.subject,v.method)] for v in a.itertuples()];a.to_csv(out/"simple_participant.csv",index=False);summary=[];pairs=[]
for (kind,method),g in a.groupby(["kind","method"]):
 if len(g)!=19 or g.replicates.min()!=6:continue
 g=g.set_index("subject");base=a[(a.kind==kind)&(a.method=="frozen")].set_index("subject");delta=g.f1-base.f1;summary.append(dict(kind=kind,method=method,f1=100*g.f1.mean(),delta_pp=100*delta.mean(),harm_rate=float((delta<-.01).mean()),worst_delta_pp=100*delta.min(),transition_delta_pp=100*(g.transition-base.transition).mean(),posthoc_extension=method in ["continuous","extended_continuous","periodic","reset_periodic","source_selected_simple"]))
 if method=="identity_future":
  for comparator in ["periodic","reset_periodic","source_selected_simple","extended_continuous"]:
   ref=a[(a.kind==kind)&(a.method==comparator)].set_index("subject")
   if len(ref)!=19 or ref.replicates.min()!=6:continue
   d=(g.f1-ref.f1).to_numpy();rng=np.random.default_rng(2026);ci=np.quantile(d[rng.integers(0,19,(10000,19))].mean(1),[.025,.975]);pairs.append(dict(kind=kind,comparator=comparator,future_delta_pp=100*d.mean(),ci_low=100*ci[0],ci_high=100*ci[1]))
s=pd.DataFrame(summary);s.to_csv(out/"simple_summary.csv",index=False);pd.DataFrame(pairs).to_csv(out/"simple_paired.csv",index=False);print(s[s.method.isin(["frozen","identity_future","periodic","reset_periodic","extended_continuous","source_selected_simple"])].to_string(index=False));print(pd.DataFrame(pairs).to_string(index=False));print("FILES",len(rows)//5,"/228")
