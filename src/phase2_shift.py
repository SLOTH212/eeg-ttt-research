from pathlib import Path
import json,numpy as np,pandas as pd
r=Path(__file__).resolve().parents[1];out=r/"phase2";rows=[]
for f in (out/"fresh").glob("*_baselines_s*.json"):
 d=json.loads(f.read_text());kind,fold,seed=d["name"].split("_")
 for v in d["results"]:
  if v["method"] not in ["frozen","ema_08","source_selected_filter","head_temporal"]:continue
  cm=np.array(v["confusion"]);den=cm.sum(0)+cm.sum(1);f1=2*cm.diagonal()/np.maximum(den,1);present=cm.sum(1)>0
  rows.append(dict(kind=kind,fold=fold,seed=seed,subject=d["subject"],method=v["method"],present_macro_f1=float(f1[present].mean()),fixed_macro_f1=float(f1.mean()),**{f"f1_{k}":v for k,v in enumerate(f1)},**{f"recall_{k}":v for k,v in enumerate(cm.diagonal()/np.maximum(cm.sum(1),1))}))
df=pd.DataFrame(rows);a=df.groupby(["kind","method","subject"]).mean(numeric_only=True).reset_index();a.to_csv(out/"fresh_stage_participant.csv",index=False);s=a.groupby(["kind","method"]).mean(numeric_only=True).drop(columns="subject");s.to_csv(out/"fresh_stage_summary.csv");print(s.to_string(float_format=lambda x:f"{100*x:.2f}"))
metadata=pd.read_csv(out/"subject_metadata.csv").drop_duplicates("subject");details={}
for group,ids in [("development",range(20)),("fresh",range(20,39))]:
 a=metadata[metadata.subject.isin(ids)];counts=np.ones((5,5));dwell=[[] for _ in range(5)]
 for sid in ids:
  base=r/("data/processed" if sid<20 else "phase2/processed");y=np.load(base/f"s{sid:02d}_meta.npz")["y"];valid=(y[:-1]>=0)&(y[1:]>=0);counts+=np.bincount(y[:-1][valid]*5+y[1:][valid],minlength=25).reshape(5,5)
  boundaries=np.r_[0,np.flatnonzero(y[1:]!=y[:-1])+1,len(y)]
  for start,end in zip(boundaries[:-1],boundaries[1:]):
   if y[start]>=0:dwell[y[start]].append(int(end-start))
 details[group]=dict(n=len(a),age_min=int(a.age.min()),age_max=int(a.age.max()),age_median=float(a.age.median()),female=int((a["sex (F=1)"]==1).sum()),male=int((a["sex (F=1)"]==2).sum()),transition_matrix=(counts/counts.sum(1,keepdims=True)).tolist(),median_stage_run_windows=[float(np.median(v)) for v in dwell])
(out/"shift_analysis.json").write_text(json.dumps(details,indent=2));print(json.dumps(details))
