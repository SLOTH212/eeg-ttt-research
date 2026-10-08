from pathlib import Path
import json,numpy as np,pandas as pd,hashlib
r=Path(__file__).resolve().parents[1];out=r/"phase6";rows=[]
for folder in ["fresh_baselines","fresh_controllers"]:
 for f in (out/folder).glob("*.json"):
  d=json.loads(f.read_text())
  for v in d["results"]:rows.append(dict(variant=d["variant"],source=d["source"],seed=d["seed"],subject=d["subject"],method=v["method"],f1=v["macro_f1"],transition=v["transition_f1"],ce=v["ce"]))
df=pd.DataFrame(rows);df.to_csv(out/"all_metrics.csv",index=False);a=df.groupby(["variant","source","subject","method"])[["f1","transition","ce"]].mean().reset_index();n=df.groupby(["variant","source","subject","method"]).size();a["seeds"]=[n.loc[(v.variant,v.source,v.subject,v.method)] for v in a.itertuples()];a.to_csv(out/"participant_metrics.csv",index=False)
summary=[];pairs=[]
for (variant,source,method),g in a.groupby(["variant","source","method"]):
 if len(g)!=19 or g.seeds.min()!=3:continue
 g=g.set_index("subject");base=a[(a.variant==variant)&(a.source==source)&(a.method=="frozen")].set_index("subject");delta=g.f1-base.f1;summary.append(dict(variant=variant,source=source,method=method,n=19,f1=100*g.f1.mean(),delta_vs_own_frozen_pp=100*delta.mean(),harm_vs_own_frozen=float((delta<-.01).mean()),transition=100*g.transition.mean(),ce=g.ce.mean()))
 for cv,cm in [("wide","frozen"),("full52","frozen"),(variant,"frozen"),(variant,"ema_08"),(variant,"periodic"),(variant,"source_selected"),(variant,"identity_confidence"),(variant,"identity_current")]:
  ref=a[(a.variant==cv)&(a.source==source)&(a.method==cm)].set_index("subject")
  if len(ref)!=19 or ref.seeds.min()!=3:continue
  d=(g.f1-ref.f1).to_numpy();seed=int.from_bytes(hashlib.sha256(str((variant,source,method,cv,cm)).encode()).digest()[:4],"little");rng=np.random.default_rng(seed);ci=np.quantile(d[rng.integers(0,19,(10000,19))].mean(1),[.025,.975]);pairs.append(dict(variant=variant,source=source,method=method,comparator_variant=cv,comparator_method=cm,delta_pp=100*d.mean(),ci_low=100*ci[0],ci_high=100*ci[1],wins=int((d>0).sum()),harm_rate=float((d<-.01).mean()),worst_delta_pp=float(100*d.min()),transition_delta_pp=float(100*(g.transition-ref.transition).mean())))
s=pd.DataFrame(summary);p=pd.DataFrame(pairs);s.to_csv(out/"summary.csv",index=False);p.to_csv(out/"paired.csv",index=False);primary=p[(p.variant=="context11")&(p.source=="young")&(p.method=="frozen")&(p.comparator_variant=="wide")&(p.comparator_method=="frozen")];print("PRIMARY",primary.to_string(index=False));print(s[s.method=="frozen"].to_string(index=False));print("FILES",len(list((out/"fresh_baselines").glob("*.json"))),len(list((out/"fresh_controllers").glob("*.json"))))
