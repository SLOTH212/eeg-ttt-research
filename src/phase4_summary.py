from pathlib import Path
import json,numpy as np,pandas as pd
r=Path(__file__).resolve().parents[1];rows=[]
for phase,pattern in [("phase3","spectral_*/evaluation.json"),("phase4","*/evaluation.json"),("phase4_controls","*/evaluation.json")]:
 for f in (r/phase/"results").glob(pattern):
  d=json.loads(f.read_text())
  if phase=="phase3":_,source,seed=d["name"].split("_");variant="full52"
  else:source=d["source"];seed=d["seed"];variant=d["variant"]
  for v in d["results"]:
   if v["method"] not in ["frozen","ema_08"]:continue
   sid=v["subject"];pr=np.load(f.parent/f"predictions_s{sid:02d}.npz");y=pr["y"];p=pr[v["method"]];mask=np.zeros(len(y),bool)
   for b in np.flatnonzero((y[1:]!=y[:-1])&(y[1:]>=0)&(y[:-1]>=0))+1:mask[max(0,b-2):min(len(y),b+3)]=True
   k=mask&(y>=0);cm=np.bincount(y[k]*5+p[k].argmax(1),minlength=25).reshape(5,5);transition=float(np.mean(2*cm.diagonal()/np.maximum(cm.sum(0)+cm.sum(1),1)))
   rows.append(dict(variant=variant,source=source,seed=int(seed),subject=sid,cohort=v["cohort"],method=v["method"],f1=v["macro_f1"],transition=transition))
df=pd.DataFrame(rows);df.to_csv(r/"phase4/all_metrics.csv",index=False);a=df.groupby(["variant","source","subject","cohort","method"])[["f1","transition"]].mean().reset_index();a.to_csv(r/"phase4/participant_metrics.csv",index=False);summary=[]
for (variant,source,cohort,method),g in a.groupby(["variant","source","cohort","method"]):
 base=a[(a.variant=="full52")&(a.source==source)&(a.cohort==cohort)&(a.method=="frozen")].set_index("subject");gg=g.set_index("subject");delta=gg.f1-base.f1
 summary.append(dict(variant=variant,source=source,cohort=cohort,method=method,f1=100*g.f1.mean(),delta_pp=100*delta.mean(),harm_rate=float((delta<-.01).mean()),transition=100*g.transition.mean(),transition_delta_pp=100*(gg.transition-base.transition).mean(),seeds=df[(df.variant==variant)&(df.source==source)].seed.nunique(),n=len(g)))
s=pd.DataFrame(summary);s.to_csv(r/"phase4/summary.csv",index=False);print(s[s.method=="frozen"].to_string(index=False,float_format=lambda x:f"{x:.3f}"))
