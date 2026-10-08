from phase14_features import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings,pickle
OUT=ROOT/"phase19";(OUT/"features").mkdir(exist_ok=True);split=json.loads((ROOT/"phase3/protocol.json").read_text())["folds"]["young"];subjects=split["base_train"]+split["base_val"]
for sid in subjects:
 dest=OUT/"features"/f"s{sid:02d}.npz"
 if dest.exists():continue
 data=ROOT/("data/processed" if sid<20 else "phase2/processed");raw=np.load(data/f"s{sid:02d}_x.npy",mmap_mode="r");sos=butter(4,[.1,45.],fs=100,btype="bandpass",output="sos");features={};start=time.perf_counter()
 for kind,pretrained in [("pretrained",True),("random",False)]:
  model=make_encoder(pretrained);zs=[]
  for i in range(0,len(raw),32):
   block=np.array(raw[i:i+32]);x=resample_poly(sosfiltfilt(sos,block,axis=-1),2,1,axis=-1).astype(np.float32);x=x.reshape(len(x),2,6,5,200).transpose(0,2,1,3,4).reshape(-1,2,5,200);z=encode(model,x);z=z.reshape(len(block),2,3,200).mean(2).reshape(len(block),400);zs.append(z)
  features[kind]=np.concatenate(zs);assert np.isfinite(features[kind]).all();del model
 np.savez_compressed(dest,**features);print("SHORT SOURCE FEATURES",sid,time.perf_counter()-start,flush=True)
def data(sid,representation):
 root=ROOT/("data/processed" if sid<20 else "phase2/processed");meta=np.load(root/f"s{sid:02d}_meta.npz");cache=np.load(OUT/"features"/f"s{sid:02d}.npz");x=cache["random" if representation.startswith("random") else "pretrained"]
 if "plus" in representation:x=np.concatenate([x,meta["features"]],1)
 return x,meta["y"]
results=[]
for representation in ["pretrained400","random400","pretrained_plus_spectral452","random_plus_spectral452"]:
 dest=OUT/f"{representation}.json"
 if dest.exists():results.append(json.loads(dest.read_text()));continue
 xs=[];ys=[];ss=[]
 for sid in split["base_train"]:
  x,y=data(sid,representation);valid=y>=0;xs.append(x[valid]);ys.append(y[valid]);ss.extend([sid]*int(valid.sum()))
 x=np.concatenate(xs);y=np.concatenate(ys);ss=np.array(ss);scaler=StandardScaler().fit(x);x=scaler.transform(x);counts=np.bincount(y,minlength=5);cw=np.sqrt(len(y)/np.maximum(counts,1));ns={s:int(np.sum(ss==s)) for s in np.unique(ss)};w=np.array([1/ns[s] for s in ss])*cw[y];w/=w.mean();val={s:data(s,representation) for s in split["base_val"]};best=-np.inf;grid=[]
 for c in [.001,.01,.1,1.,10.]:
  model=LogisticRegression(C=c,max_iter=2000,tol=1e-5,solver="lbfgs")
  with warnings.catch_warnings(record=True) as ws:warnings.simplefilter("always");model.fit(x,y,sample_weight=w)
  if any(issubclass(v.category,ConvergenceWarning) for v in ws):
   with warnings.catch_warnings(record=True) as ws:warnings.simplefilter("always");model.set_params(max_iter=10000,warm_start=True);model.fit(x,y,sample_weight=w)
   assert not any(issubclass(v.category,ConvergenceWarning) for v in ws)
  per={s:metrics(yv,model.predict_proba(scaler.transform(xv)))["macro_f1"] for s,(xv,yv) in val.items()};score=float(np.mean(list(per.values())));grid.append(dict(C=c,val_f1=score,participants=per));print("SHORT SOURCE PROBE",representation,c,score,flush=True)
  if score>best:best=score;selected=model;selectedc=c
 with (OUT/f"{representation}.pkl").open("wb") as f:pickle.dump(dict(scaler=scaler,model=selected),f)
 old=json.loads((ROOT/"phase14/probes"/f"young_{representation}/selection.json").read_text());result=dict(representation=representation,C=selectedc,val_f1=best,old15s_val_f1=old["val_f1"],delta_pp=100*(best-old["val_f1"]),grid=grid,source_validation_only=True);dest.write_text(json.dumps(result,indent=2));results.append(result)
(OUT/"complete.json").write_text(json.dumps(dict(source="young",results=results,source_validation_only=True),indent=2));print("SHORT SOURCE COMPLETE",flush=True)
