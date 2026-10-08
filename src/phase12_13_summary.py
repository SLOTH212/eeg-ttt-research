from pathlib import Path
import json,pandas as pd,numpy as np,hashlib
r=Path(__file__).resolve().parents[1]
for phase in ["phase12","phase13"]:
 rows=[]
 for f in (r/phase).glob("*/evaluation.json"):
  d=json.loads(f.read_text());name=d["name"];kind="_".join(name.split("_")[:2]) if name.startswith("context11") else name.split("_")[0]
  for v in d["results"]:rows.append(dict(kind=kind,case=name,subject=v["subject"],method=v["method"],f1=v["macro_f1"]))
 df=pd.DataFrame(rows);assert df.case.nunique()==18;df.to_csv(r/phase/"all_metrics.csv",index=False)
 counts=df.groupby(["kind","subject","method"]).size();assert (counts==3).all()
 a=df.groupby(["kind","subject","method"]).f1.mean().reset_index();a.to_csv(r/phase/"participant_metrics.csv",index=False);out=[];pairs=[]
 for (kind,method),g in a.groupby(["kind","method"]):
  for comparator in ["frozen","weak_ema","source_selected_ema","current","teacher_confidence"]:
   b=a[(a.kind==kind)&(a.method==comparator)].set_index("subject").f1
   if len(b)!=len(g):continue
   delta=g.set_index("subject").f1-b;rng=np.random.default_rng(17029);boot=rng.choice(delta.to_numpy(),(10000,len(delta)),replace=True).mean(1)*100;lo,hi=np.quantile(boot,[.025,.975]);row=dict(kind=kind,method=method,n=len(delta),f1=100*g.f1.mean(),delta_pp=100*delta.mean(),ci_low=lo,ci_high=hi,harm_n=int((delta<-.01).sum()))
   pairs.append(dict(**row,comparator=comparator))
   if comparator=="frozen":out.append(row)
 pd.DataFrame(out).to_csv(r/phase/"summary.csv",index=False);pd.DataFrame(pairs).to_csv(r/phase/"paired.csv",index=False)
 print(phase,df.case.nunique(),"complete cases")
