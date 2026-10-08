from pathlib import Path
import json,pandas as pd,numpy as np
r=Path(__file__).resolve().parents[1]
for phase in [14,15]:
 out=r/f"phase{phase}";rows=[];files=list((out/"probes" if phase==14 else out).glob("*/evaluation.json"));assert len(files)==(10 if phase==14 else 12)
 for f in files:
  d=json.loads(f.read_text())
  for v in d["results"]:rows.append(dict(source=d["source"],representation=d["representation"],objective=d.get("objective","none"),cohort=v["cohort"],subject=v["subject"],method=v["method"],f1=v["macro_f1"],transition=v["transition_f1"]))
 a=pd.DataFrame(rows);a.to_csv(out/"participant_metrics.csv",index=False);dev=a[a.cohort!="older"].copy();dev.cohort="all_dev";a=pd.concat([a,dev]);cols=["source","representation","objective","cohort","method"];summary=a.groupby(cols)[["f1","transition"]].mean().mul(100).reset_index();summary.to_csv(out/"summary.csv",index=False);pairs=[]
 for key,g in a.groupby(cols):
  source,rep,obj,cohort,method=key
  comparators=[(c,method) for c in ["spectral52","random400","pretrained400","random_plus_spectral452"]] if phase==14 else [(rep,c) for c in ["frozen","weak_ema","continuous","periodic","reset_periodic","current","confidence"]]
  for crep,cmethod in comparators:
   b=a[(a.source==source)&(a.cohort==cohort)&(a.representation==crep)&(a.objective==obj)&(a.method==cmethod)].set_index("subject")
   if len(b)!=len(g):continue
   gg=g.set_index("subject");delta=100*(gg.f1-b.f1);rng=np.random.default_rng(17029);lo,hi=np.quantile(rng.choice(delta.to_numpy(),(10000,len(delta))).mean(1),[.025,.975]);pairs.append(dict(zip(cols,key))|dict(comparator_representation=crep,comparator_method=cmethod,n=len(delta),delta_pp=delta.mean(),ci_low=lo,ci_high=hi,harm_n=int((delta< -1).sum()),worst_delta_pp=delta.min(),transition_delta_pp=100*(gg.transition-b.transition).mean()))
 p=pd.DataFrame(pairs);p.to_csv(out/"paired.csv",index=False)
 if phase==14:show=p[(p.representation=="pretrained_plus_spectral452")&(p.comparator_representation=="random_plus_spectral452")&(p.method=="frozen")&(p.cohort.isin(["all_dev","older"]))]
 else:show=p[(p.method=="future")&(p.comparator_method.isin(["frozen","weak_ema"]))&(p.cohort.isin(["all_dev","older"]))]
 print("PHASE",phase);print(show.to_string(index=False,float_format=lambda x:f"{x:.3f}"))
