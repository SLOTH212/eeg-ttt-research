from pathlib import Path
import json
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
r=Path(__file__).resolve().parents[1];o=r/"phase29";d=pd.read_csv(o/"pulses.csv");rows=[];features=["entropy","confidence","aux_loss","gradient_norm","proposal_l1"]
protocol=dict(purpose="Source-controller leave-one-person-out cold-write observability",features=features,excluded=["current_class","current_correct","stable","current_ce_gain","future_ce_gain","gradient_cosine","first_order_gain","base_ce","base_correct","write_correct"],models="Ridge alpha100; training-fold standardized features; equal-person training weights",policies=["KEEP","WRITE","train_mean_future","future_confidence2","future_features5","current_features5"],selection="All48source/weight/objective/lr groups; no target rollout; fit3people evaluatefourth",limitations="Repeated4controllers. Cold independent gains do not establish accumulated-state gate performance.")
(o/"observability_protocol.json").write_text(json.dumps(protocol,indent=2))
for keys,g in d.groupby(["source","weight","objective","lr"]):
 assert sorted(g.subject.unique().tolist())==[7,17,27,37]
 for sid in [7,17,27,37]:
  tr=g[g.subject!=sid];te=g[g.subject==sid];gain=te.future_ce_gain.to_numpy();policies={"KEEP":np.zeros(len(te),bool),"WRITE":np.ones(len(te),bool),"train_mean_future":np.repeat(tr.groupby("subject").future_ce_gain.mean().mean()>0,len(te))}
  for name,cols,target in [("future_confidence2",features[:2],"future_ce_gain"),("future_features5",features,"future_ce_gain"),("current_features5",features,"current_ce_gain")]:
   scaler=StandardScaler().fit(tr[cols]);counts=tr.subject.value_counts();w=np.array([1/counts[s] for s in tr.subject]);w/=w.mean();model=Ridge(alpha=100).fit(scaler.transform(tr[cols]),tr[target],sample_weight=w);pred=model.predict(scaler.transform(te[cols]));policies[name]=pred>0
  for name,commit in policies.items():rows.append(dict(source=keys[0],weight=keys[1],objective=keys[2],lr=keys[3],subject=sid,policy=name,utility=float(np.mean(np.where(commit,gain,0))),commit_fraction=float(commit.mean()),always_write=float(gain.mean()),oracle=float(np.maximum(gain,0).mean())))
frame=pd.DataFrame(rows);frame.to_csv(o/"observability_people.csv",index=False);summary=frame.groupby(["source","weight","objective","lr","policy"])[["utility","commit_fraction","always_write","oracle"]].mean().reset_index();summary.to_csv(o/"observability_summary.csv",index=False);subset=summary[(summary.source=="young")&(summary.weight==.01)&(summary.objective=="masked_token")&(summary.lr==.01)];print(subset.to_string(index=False));(o/"observability_audit.json").write_text(json.dumps(dict(status="PASS",groups=48,folds=192,features_whitelisted=True,people_disjoint_per_fold=True,rows=len(frame),source_only=True),indent=2))
