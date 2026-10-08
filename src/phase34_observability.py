from common import *
from phase24_observability import values
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import pandas as pd
OUT=ROOT/"phase34"
if __name__=="__main__":
 files=sorted(OUT.glob("lr*/complete.json"));assert len(files)==8;parts=[];audits=[]
 for f in files:
  m=json.loads(f.read_text());assert m["trajectory_reference_max_error"]<2e-5 and m["trajectory_reference_argmax_disagreements"]==0 and m["proposal_restored_exact"] and m["nonwritable_unchanged"];saved=np.load(f.parent/"pulses.npz");old=np.load(ROOT/"phase32"/f.parent.name/"pulses.npz");idx=np.searchsorted(saved["t"],old["t"]);assert np.array_equal(saved["t"][idx],old["t"]);feature_err=float(abs(saved["features"][idx]-old["features"]).max());gain_err=float(abs(saved["future_gain"][idx]-old["future_gain"]).max());assert np.allclose(saved["features"][idx],old["features"],atol=3e-4,rtol=3e-5) and gain_err<3e-5
  predictions=np.load(f.parent/"actual_predictions.npz");assert abs(metrics(predictions["y"],predictions["p"])["macro_f1"]-m["actual_metrics"]["macro_f1"])<1e-12;x=saved["features"];assert x.shape[1]==436 and np.isfinite(x).all();assert np.isfinite(saved["future_gain"]).all();assert np.all(saved["future_gain"][x[:,3]==0,0]==0)
  parts.append(dict(features=x,future_gain=saved["future_gain"],current_gain=saved["current_gain"],t=saved["t"],subject=np.full(len(x),m["subject"]),trajectory=np.full(len(x),m["trajectory"])));audits.append(dict(subject=m["subject"],trajectory=m["trajectory"],pulses=len(x),max_common_feature_error=feature_err,max_common_gain_error=gain_err,trajectory_max_error=m["trajectory_reference_max_error"]))
 data={k:np.concatenate([p[k] for p in parts]) for k in parts[0]};x=data["features"];subjects=data["subject"];t=data["t"];target=data["future_gain"];rows=[];budgets=[];groups={"confidence4":[0,4,18,19],"features36":list(range(35))+[435],"features436":list(range(436))}
 def record(held,scheme,policy,subset,indices,actions):
  utility=np.column_stack([np.zeros(len(indices)),target[indices]])[np.arange(len(indices)),actions];rows.append(dict(held_subject=held,scheme=scheme,policy=policy,test_subset=subset,n=len(indices),utility=float(utility.mean()),commit_fraction=float((actions==2).mean()),reset_fraction=float((actions==1).mean())))
 for held in [7,17,27,37]:
  train=subjects!=held;test=~train;tests={"dense":np.flatnonzero(test),"common_sparse0":np.flatnonzero(test&(t%20==0))}
  for subset,ii in tests.items():
   obs=x[ii]
   for policy,action in {"KEEP":np.zeros(len(ii),int),"WRITE":np.full(len(ii),2),"RESET":np.ones(len(ii),int),"low_entropy_write":np.where(obs[:,0]<.5,2,0),"anchor_guard":np.where(obs[:,19]>.05,1,2)}.items():record(held,"fixed",policy,subset,ii,action)
  sparse_n=int((train&(t%20==0)).sum())
  for scheme in ["sparse0","sparse5","sparse10","sparse15","dense"]:
   tr=train if scheme=="dense" else train&(t%20==int(scheme.removeprefix("sparse")));ss=subjects[tr];assert set(ss).isdisjoint({held});alpha=100*tr.sum()/sparse_n if scheme=="dense" else 100.;counts={s:int((ss==s).sum()) for s in np.unique(ss)};w=np.array([1/counts[s] for s in ss]);w/=w.mean();future_labels={(int(s),int(tt+offset)) for s,tt in zip(ss,t[tr]) for offset in range(1,5)};budgets.append(dict(held_subject=held,scheme=scheme,training_rows=int(tr.sum()),unique_future_labels=len(future_labels),ridge_alpha=alpha))
   mean=np.mean([target[tr][ss==s].mean(0) for s in np.unique(ss)],axis=0)
   for subset,ii in tests.items():record(held,scheme,"train_mean_future",subset,ii,np.full(len(ii),int(np.argmax(np.r_[0,mean]))))
   for group,columns in groups.items():
    scaler=StandardScaler().fit(x[tr][:,columns]);z=scaler.transform(x[tr][:,columns])
    for goal,key in [("future","future_gain"),("current","current_gain")]:
     model=Ridge(alpha=alpha).fit(z,data[key][tr],sample_weight=w);gate=dict(indices=columns,scaler=scaler,model=model,kind="ridge")
     for subset,ii in tests.items():
      v=values(gate,x[ii]);policy=f"{goal}_{group}";record(held,scheme,policy,subset,ii,np.column_stack([np.zeros(len(ii)),v]).argmax(1));record(held,scheme,policy+"_write_only",subset,ii,np.where(v[:,1]>0,2,0));record(held,scheme,policy+"_reset_only",subset,ii,np.where(v[:,0]>0,1,0))
 frame=pd.DataFrame(rows);frame.to_csv(OUT/"loso_people.csv",index=False);summary=frame.groupby(["scheme","policy","test_subset"])[["utility","commit_fraction","reset_fraction"]].mean().reset_index();summary.to_csv(OUT/"loso_summary.csv",index=False);pd.DataFrame(budgets).to_csv(OUT/"label_budgets.csv",index=False);best_fixed=float(summary[(summary.scheme=="fixed")&(summary.test_subset=="dense")].utility.max());decisions=[]
 for _,row in summary[(summary.scheme=="dense")&(summary.test_subset=="dense")&summary.policy.str.startswith("future")].iterrows():
  pp=frame[(frame.scheme=="dense")&(frame.test_subset=="dense")&(frame.policy==row.policy)];positive=int((pp.utility>0).sum());counter=row.policy.replace("future","current",1);current=float(summary[(summary.scheme=="dense")&(summary.test_subset=="dense")&(summary.policy==counter)].utility.iloc[0]);decisions.append(dict(policy=row.policy,utility=float(row.utility),best_fixed_utility=best_fixed,positive_people=positive,matched_current_utility=current,advance=bool(row.utility>best_fixed and positive>=3)))
 (OUT/"advance_decisions.json").write_text(json.dumps(dict(decisions=decisions),indent=2));(OUT/"audit.json").write_text(json.dumps(dict(status="PASS",cases=8,pulses=len(x),unique_source_times=len(set(zip(subjects,t))),same_actual_trajectory=True,common_sparse_pulses_match=True,subject_disjoint_training=True,normalized_regularization=True,label_budget_explicit=True,details=audits),indent=2));print(summary[(summary.test_subset=="dense")&summary.policy.isin(["WRITE","RESET","future_features436","current_features436","future_features436_write_only","current_features436_write_only"])].to_string(index=False));print("ADVANCE",[d for d in decisions if d["advance"]],flush=True)
