from pathlib import Path
import json,numpy as np,pandas as pd
r=Path(__file__).resolve().parents[1];out=r/"phase2";rows=[]
for path in (out/"fresh").glob("*.json"):
 d=json.loads(path.read_text());kind,fold,seed=d["name"].split("_")
 for row in d["results"]:rows.append(dict(kind=kind,fold=fold,seed=int(seed),subject=row["subject"],method=row["method"],f1=row["macro_f1"],transition=row["transition_f1"],ce=row["ce"],**{f"recall_{k}":v for k,v in enumerate(row["recall"])}))
df=pd.DataFrame(rows);df.to_csv(out/"fresh_all.csv",index=False)
a=df.groupby(["kind","subject","method"])[["f1","transition","ce"]+[f"recall_{k}" for k in range(5)]].mean().reset_index();counts=df.groupby(["kind","subject","method"]).size();a["replicates"]=[counts.loc[(x.kind,x.subject,x.method)] for x in a.itertuples()];a.to_csv(out/"fresh_participant.csv",index=False)
summary=[];paired=[];rng=np.random.default_rng(20261003)
for kind,g in a.groupby("kind"):
 base=g[g.method=="frozen"].set_index("subject");weak=g[g.method=="ema_08"].set_index("subject")
 for method,h in g.groupby("method"):
  if len(h)!=19 or h.replicates.min()!=6:
   print("PENDING",kind,method,"subjects",len(h),"min_replicates",h.replicates.min());continue
  h=h.set_index("subject");ix=h.index.intersection(base.index);delta=h.loc[ix,"f1"]-base.loc[ix,"f1"];ci=np.quantile(delta.to_numpy()[rng.integers(0,len(delta),(10000,len(delta)))].mean(1),[.025,.975])
  summary.append(dict(kind=kind,method=method,n=len(h),min_replicates=h.replicates.min(),mean_f1=100*h.f1.mean(),delta_pp=100*delta.mean(),ci_low=100*ci[0],ci_high=100*ci[1],harm_rate=(delta<-.01).mean(),worst_delta_pp=100*delta.min(),transition_f1=100*h.transition.mean(),transition_delta_pp=100*(h.loc[ix,"transition"]-base.loc[ix,"transition"]).mean(),ce=h.ce.mean()))
  for comparator in ["frozen","ema_08","identity_confidence","identity_current","source_prototype"]:
   c=g[g.method==comparator].set_index("subject");ix=h.index.intersection(c.index)
   if len(ix)!=19 or h.loc[ix,"replicates"].min()!=6 or c.loc[ix,"replicates"].min()!=6:continue
   diff=(h.loc[ix,"f1"]-c.loc[ix,"f1"]).to_numpy();ci=np.quantile(diff[rng.integers(0,19,(10000,19))].mean(1),[.025,.975]);paired.append(dict(kind=kind,method=method,comparator=comparator,delta_pp=100*diff.mean(),ci_low=100*ci[0],ci_high=100*ci[1],wins=int((diff>0).sum()),n=19))
s=pd.DataFrame(summary);s.to_csv(out/"fresh_summary.csv",index=False);pd.DataFrame(paired).to_csv(out/"fresh_paired.csv",index=False)
print(s.to_string(index=False,float_format=lambda x:f"{x:.3f}"));print("FILES",len(list((out/"fresh").glob("*.json"))),"EXPECTED",456)
