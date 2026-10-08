from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from phase2_temporal import ema
import argparse,warnings,pickle,time
OUT=ROOT/"phase14"
def subject(sid,representation):
 data=ROOT/("data/processed" if sid<20 else ("phase2/processed" if sid<40 else "phase6/processed"));meta=np.load(data/f"s{sid:02d}_meta.npz");hf=meta["features"]
 if representation=="spectral52":x=hf
 else:
  path=OUT/"features"/f"s{sid:02d}.npz";start=time.monotonic()
  while not path.with_suffix(".json").exists():
   if time.monotonic()-start>1800:raise TimeoutError(str(path))
   time.sleep(2)
  cache=np.load(path);x=cache["random" if representation.startswith("random") else "pretrained"]
  if representation.endswith("plus_spectral452"):x=np.concatenate([x,hf],axis=1)
 return x,meta["y"]
def run(source,representation):
 out=OUT/"probes"/(source+"_"+representation);out.mkdir(parents=True,exist_ok=True)
 if (out/"evaluation.json").exists():return
 split=json.loads((ROOT/"phase3/protocol.json").read_text())["folds"][source];xs=[];ys=[];ss=[]
 for sid in split["base_train"]:
  x,y=subject(sid,representation);mask=y>=0;xs.append(x[mask]);ys.append(y[mask]);ss.extend([sid]*int(mask.sum()))
 x=np.concatenate(xs);y=np.concatenate(ys);ss=np.array(ss);scaler=StandardScaler().fit(x);x=scaler.transform(x);counts=np.bincount(y,minlength=5);classw=np.sqrt(len(y)/np.maximum(counts,1));subject_n={s:int(np.sum(ss==s)) for s in np.unique(ss)};w=np.array([1/subject_n[s] for s in ss])*classw[y];w/=w.mean();vals={s:subject(s,representation) for s in split["base_val"]};best=-np.inf;history=[];start=time.perf_counter()
 for c in [.001,.01,.1,1.,10.]:
  with warnings.catch_warnings(record=True) as ws:
   warnings.simplefilter("always");model=LogisticRegression(C=c,max_iter=500,tol=1e-5,solver="lbfgs").fit(x,y,sample_weight=w)
  if any(issubclass(v.category,ConvergenceWarning) for v in ws):
   with warnings.catch_warnings(record=True) as ws:
    warnings.simplefilter("always");model=LogisticRegression(C=c,max_iter=2000,tol=1e-5,solver="lbfgs").fit(x,y,sample_weight=w)
   if any(issubclass(v.category,ConvergenceWarning) for v in ws):
    with warnings.catch_warnings(record=True) as ws:
     warnings.simplefilter("always");model.set_params(max_iter=10000,warm_start=True);model.fit(x,y,sample_weight=w)
   assert not any(issubclass(v.category,ConvergenceWarning) for v in ws),"Probe failed to converge"
  score=float(np.mean([metrics(vy,model.predict_proba(scaler.transform(vx)))["macro_f1"] for vx,vy in vals.values()]));row=dict(C=c,val_macro_f1=score,iterations=model.n_iter_.tolist());history.append(row);print("PROBE SELECT",source,representation,row,flush=True)
  if score>best:best=score;best_model=model;best_c=c
 selection=dict(source=source,representation=representation,C=best_c,val_f1=best,history=history,training_subjects=split["base_train"],validation_subjects=split["base_val"],feature_dim=x.shape[1],seconds=time.perf_counter()-start)
 (out/"selection.json").write_text(json.dumps(selection,indent=2))
 with (out/"model.pkl").open("wb") as f:pickle.dump(dict(scaler=scaler,model=best_model,selection=selection),f)
 results=[]
 for sid in split["evaluation"]+list(range(40,59)):
  vx,vy=subject(sid,representation);p=best_model.predict_proba(scaler.transform(vx));preds=dict(frozen=p,weak_ema=ema(p,.8));mask=np.zeros(len(vy),bool)
  for b in np.flatnonzero((vy[1:]!=vy[:-1])&(vy[1:]>=0)&(vy[:-1]>=0))+1:mask[max(0,b-2):min(len(vy),b+3)]=True
  for method,pr in preds.items():results.append(dict(subject=sid,cohort="older" if sid>=40 else ("middle_dev" if sid>=20 else "young_dev"),method=method,**metrics(vy,pr),transition_f1=metrics(vy[mask],pr[mask])["macro_f1"]))
  np.savez_compressed(out/f"s{sid:02d}.npz",y=vy,**preds)
 (out/"evaluation.json").write_text(json.dumps(dict(source=source,representation=representation,results=results,selection=selection,exploratory=True),indent=2));print("PROBE COMPLETE",source,representation,flush=True)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--source",required=True);p.add_argument("--representation",required=True);a=p.parse_args();run(a.source,a.representation)
