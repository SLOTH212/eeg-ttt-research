from common import *
import pandas as pd,pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
OUT=ROOT/"phase24"
SPECS={"future_confidence4":("confidence4","future_gain","ridge"),"future20":("base20","future_gain","ridge"),"future35":("direction_history35","future_gain","ridge"),"future435":("with_embedding435","future_gain","ridge"),"current20":("base20","current_gain","ridge"),"current35":("direction_history35","current_gain","ridge"),"current435":("with_embedding435","current_gain","ridge"),"future_hgb35":("direction_history35","future_gain","hgb")}

def fit_gate(x,y,subjects,indices,kind):
 counts={s:int((subjects==s).sum()) for s in np.unique(subjects)};w=np.array([1/counts[s] for s in subjects]);w/=w.mean();scaler=StandardScaler().fit(x[:,indices]);z=scaler.transform(x[:,indices])
 if kind=="ridge":model=Ridge(alpha=100).fit(z,y,sample_weight=w)
 else:model=[HistGradientBoostingRegressor(max_iter=100,max_depth=2,min_samples_leaf=30,l2_regularization=10,learning_rate=.05,random_state=17).fit(z,y[:,i],sample_weight=w) for i in range(2)]
 return dict(indices=indices,scaler=scaler,model=model,kind=kind)
def values(gate,x):
 z=gate["scaler"].transform(x[:,gate["indices"]]);m=gate["model"];out=m.predict(z) if gate["kind"]=="ridge" else np.column_stack([v.predict(z) for v in m]);out[x[:,3]==0,0]=0.;return out

def run():
 protocol=json.loads((OUT/"protocol.json").read_text());files=list(OUT.glob("*/complete.json"));assert len(files)==16;bundles={};audit=[];raw=[]
 for f in files:
  d=json.loads(f.read_text());assert d["feature_count"]==435 and d["proposal_state_restored_exact"] and d["nonwritable_state_unchanged"];saved=np.load(f.parent/"pulses.npz");x=saved["features"];assert x.shape==(d["pulses"],435) and np.isfinite(x).all();gain=saved["future_gain"];assert gain.shape==(len(x),2) and np.isfinite(gain).all();assert np.all(gain[x[:,3]==0,0]==0)
  predictions=np.load(f.parent/"actual_predictions.npz");score=metrics(predictions["y"],predictions["predictions"]);assert abs(score["macro_f1"]-d["actual_trajectory_metrics"]["macro_f1"])<1e-12
  bundle={k:saved[k] for k in ["features","future_gain","current_gain","t","current_class","stable","current_correct"]};bundle["subject"]=np.full(len(x),d["subject"]);bundle["trajectory"]=np.full(len(x),d["trajectory"]);bundles.setdefault(d["source"],[]).append(bundle);audit.append(dict(source=d["source"],subject=d["subject"],trajectory=d["trajectory"],pulses=len(x),identity_reset_exact=True,trajectory_score_recomputed=True))
  for i in range(len(x)):raw.append(dict(source=d["source"],subject=d["subject"],trajectory=d["trajectory"],t=int(saved["t"][i]),source_state=x[i,3]==0,write_gain=gain[i,1],reset_gain=gain[i,0],current_class=int(saved["current_class"][i]),stable=bool(saved["stable"][i]),current_correct=bool(saved["current_correct"][i])))
 pulse=pd.DataFrame(raw);pulse.to_csv(OUT/"pulse_utilities.csv",index=False);mechanism=[]
 for keys,g in pulse.groupby(["source","trajectory","subject"]):
  mechanism.append(dict(source=keys[0],trajectory=keys[1],subject=keys[2],pulses=len(g),write_all=g.write_gain.mean(),reset_all=g.reset_gain.mean(),oracle_write=np.maximum(g.write_gain,0).mean(),oracle_all=np.maximum(np.maximum(g.write_gain,g.reset_gain),0).mean(),beneficial_write_fraction=(g.write_gain>1e-6).mean(),harmful_write_fraction=(g.write_gain< -1e-6).mean()))
 pd.DataFrame(mechanism).to_csv(OUT/"mechanism_people.csv",index=False);pd.DataFrame(mechanism).groupby(["source","trajectory"]).mean(numeric_only=True).drop(columns=["subject"]).to_csv(OUT/"mechanism_summary.csv")
 people=[];splits=[];(OUT/"gates").mkdir(exist_ok=True)
 for source,parts in bundles.items():
  data={k:np.concatenate([v[k] for v in parts]) for k in parts[0]};x=data["features"];ss=data["subject"];gains=data["future_gain"]
  for held in [7,17,27,37]:
   train=ss!=held;test=~train;assert not(set(ss[train])&set(ss[test]));target=np.column_stack([np.zeros(test.sum()),gains[test]]);signals=x[test];base_actions={"KEEP":np.zeros(test.sum(),int),"WRITE":np.full(test.sum(),2),"RESET":np.ones(test.sum(),int),"low_entropy_write":np.where(signals[:,0]<.5,2,0),"anchor_guard":np.where(signals[:,19]>.05,1,2)}
   for name,action in base_actions.items():people.append(dict(source=source,held_subject=held,policy=name,utility=float(target[np.arange(len(action)),action].mean()),commit_fraction=float((action==2).mean()),reset_fraction=float((action==1).mean())))
   for name,(group,which,kind) in SPECS.items():
    indices=protocol["feature_groups"][group];gate=fit_gate(x[train],data[which][train],ss[train],indices,kind);v=values(gate,signals);action=np.column_stack([np.zeros(len(v)),v]).argmax(1);people.append(dict(source=source,held_subject=held,policy=name,utility=float(target[np.arange(len(action)),action].mean()),commit_fraction=float((action==2).mean()),reset_fraction=float((action==1).mean())))
    for suffix,column,code in [("write_only",1,2),("reset_only",0,1)]:
     restricted=np.where(v[:,column]>0,code,0);people.append(dict(source=source,held_subject=held,policy=name+"_"+suffix,utility=float(target[np.arange(len(restricted)),restricted].mean()),commit_fraction=float((restricted==2).mean()),reset_fraction=float((restricted==1).mean())))
    splits.append(dict(source=source,held_subject=held,training_subjects=sorted(set(ss[train].tolist())),policy=name,features=group,target=which))
  for name,(group,which,kind) in SPECS.items():
   gate=fit_gate(x,data[which],ss,protocol["feature_groups"][group],kind);gate.update(source=source,policy=name,training_subjects=[7,17,27,37],feature_names=[protocol["features"][i] for i in gate["indices"]],target=which)
   with (OUT/"gates"/f"{source}_{name}.pkl").open("wb") as f:pickle.dump(gate,f)
 result=pd.DataFrame(people);result.to_csv(OUT/"loso_people.csv",index=False);summary=result.groupby(["source","policy"])[["utility","commit_fraction","reset_fraction"]].mean().reset_index();summary.to_csv(OUT/"loso_summary.csv",index=False);(OUT/"loso_splits.json").write_text(json.dumps(splits,indent=2));(OUT/"diagnostic_audit.json").write_text(json.dumps(dict(status="PASS",cases=16,pulses=len(pulse),feature_count=435,features_explicit_whitelist=True,subject_disjoint_folds=True,identity_reset_exact=True,actual_trajectory_metrics_recomputed=True,scope="Source-controller pulse diagnostics only; no online policy performance claim",cases_detail=audit),indent=2));print(summary.to_string(index=False));print("DIAGNOSTICS COMPLETE",len(pulse),flush=True)
if __name__=="__main__":run()
