from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from phase2_temporal import ema
import warnings,pickle,time,pandas as pd
OUT=ROOT/"phase63"
def data(sid,kind):
 folder=ROOT/("data/processed" if sid<20 else "phase2/processed")
 y=np.load(folder/f"s{sid:02d}_meta.npz")["y"];x=np.load(OUT/"features"/f"s{sid:02d}_{kind}.npz")["features"];assert len(x)==len(y)
 return x,y
def run():
 pr=json.loads((OUT/"protocol.json").read_text());assert (OUT/"features_complete.json").exists();allrows=[]
 for kind in ["pretrained","random"]:
  out=OUT/("probe_"+kind);out.mkdir(exist_ok=True);train=[(sid,*data(sid,kind)) for sid in pr["training_subjects"]];xx=[];yy=[];ss=[]
  for sid,x,y in train:
   valid=y>=0;xx.append(x[valid]);yy.append(y[valid]);ss.extend([sid]*int(valid.sum()))
  x=np.concatenate(xx);y=np.concatenate(yy);ss=np.array(ss);scaler=StandardScaler().fit(x);x=scaler.transform(x);counts=np.bincount(y,minlength=5);classw=np.sqrt(len(y)/np.maximum(counts,1));subject_n={s:int((ss==s).sum()) for s in np.unique(ss)};w=np.array([1/subject_n[s] for s in ss])*classw[y];w/=w.mean();vals={sid:data(sid,kind) for sid in pr["validation_subjects"]}
  best=-np.inf;history=[];start=time.perf_counter()
  for c in [.001,.01,.1,1.,10.]:
   model=LogisticRegression(C=c,max_iter=500,tol=1e-5,solver="lbfgs")
   for limit in [500,2000,10000]:
    model.set_params(max_iter=limit,warm_start=(limit==10000))
    with warnings.catch_warnings(record=True) as caught:
     warnings.simplefilter("always");model.fit(x,y,sample_weight=w)
    if not any(issubclass(v.category,ConvergenceWarning) for v in caught):break
   assert not any(issubclass(v.category,ConvergenceWarning) for v in caught)
   score=float(np.mean([metrics(vy,model.predict_proba(scaler.transform(vx)))["macro_f1"] for vx,vy in vals.values()]));history.append(dict(C=c,val_macro_f1=score,iterations=model.n_iter_.tolist()));print("CBRAMOD PROBE",kind,history[-1],flush=True)
   if score>best:best=score;best_model=model;best_c=c
  selection=dict(kind=kind,C=best_c,val_f1=best,history=history,training_subjects=pr["training_subjects"],validation_subjects=pr["validation_subjects"],feature_dim=400,training_valid_windows=len(y),seconds=time.perf_counter()-start)
  (out/"selection.json").write_text(json.dumps(selection,indent=2))
  with (out/"model.pkl").open("wb") as f:pickle.dump(dict(scaler=scaler,model=best_model,selection=selection),f)
  for sid,(vx,vy) in vals.items():
   p=best_model.predict_proba(scaler.transform(vx));np.savez_compressed(out/f"s{sid:02d}.npz",y=vy,p=p)
   for smooth in [1.,.8,.5]:
    mm=metrics(vy,p if smooth==1 else ema(p,smooth));allrows.append(dict(subject=sid,model="cbramod_"+kind,ema=smooth,macro_f1=100*mm["macro_f1"],balanced_accuracy=100*mm["balanced_accuracy"],cross_entropy=mm["ce"]))
 for kind in ["pretrained","random"]:
  with (ROOT/"phase14/probes"/("mixed_"+kind+"400")/"model.pkl").open("rb") as f:old=pickle.load(f)
  assert old["selection"]["training_subjects"]==pr["training_subjects"] and old["selection"]["validation_subjects"]==pr["validation_subjects"]
  for sid in pr["validation_subjects"]:
   _,y=data(sid,"pretrained");x=np.load(ROOT/"phase14/features"/f"s{sid:02d}.npz")[kind];p=old["model"].predict_proba(old["scaler"].transform(x))
   for smooth in [1.,.8,.5]:
    mm=metrics(y,p if smooth==1 else ema(p,smooth));allrows.append(dict(subject=sid,model="labram_frozen_"+kind,ema=smooth,macro_f1=100*mm["macro_f1"],balanced_accuracy=100*mm["balanced_accuracy"],cross_entropy=mm["ce"]))
 for sid in pr["validation_subjects"]:
  z=np.load(ROOT/"phase28/mixed_joint_0"/f"s{sid:02d}.npz")
  for smooth in [1.,.8,.5]:
   mm=metrics(z["y"],z["p"] if smooth==1 else ema(z["p"],smooth));allrows.append(dict(subject=sid,model="labram_source_finetuned_reference",ema=smooth,macro_f1=100*mm["macro_f1"],balanced_accuracy=100*mm["balanced_accuracy"],cross_entropy=mm["ce"]))
 df=pd.DataFrame(allrows);df.to_csv(OUT/"participants.csv",index=False);summary=df.groupby(["model","ema"])[["macro_f1","balanced_accuracy","cross_entropy"]].mean();summary.to_csv(OUT/"summary.csv");(OUT/"probe_complete.json").write_text(json.dumps(dict(complete=True,method_streams=len(df),source_only=True)));print(summary.to_string())
if __name__=="__main__":run()
