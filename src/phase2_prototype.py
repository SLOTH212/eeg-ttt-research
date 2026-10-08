"""Source-supervised prototype memory baseline, causal target updates after prediction."""
from common import *
import itertools,argparse,time
OUT=ROOT/"phase2"
def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def score(y,p):
 k=y>=0;cm=np.bincount(y[k]*5+p[k].argmax(1),minlength=25).reshape(5,5)
 return float(np.mean(2*cm.diagonal()/np.maximum(cm.sum(0)+cm.sum(1),1)))
def stream(z,source_p,centroids,temp,blend,rate=0.,threshold=.8):
 # source_p is a fixed classifier teacher; target labels are never accepted.
 z=unit(z.astype(float));memory=centroids.copy();out=np.empty_like(source_p,dtype=float)
 for t,(v,pr) in enumerate(zip(z,source_p)):
  active=unit(.5*centroids+.5*memory);a=active@v/temp;a=np.exp(a-a.max());a/=a.sum();out[t]=(1-blend)*pr+blend*a
  if rate and pr.max()>=threshold:
   c=pr.argmax();memory[c]=(1-rate)*memory[c]+rate*v
 return out
def run(seed=17):
 args=argparse.Namespace(seed=seed)
 for kind,fold in itertools.product(["spectral","eegnet"],["A","B"]):
  name=f"{kind}_{fold}_{args.seed}";dest=OUT/"results"/(name+"_prototype.json")
  if dest.exists():print("CACHED",name,flush=True);continue
  split=CFG["folds"][fold];data={}
  for sid in CFG["subject_ids"]:
   path=OUT/"cache"/f"{name}_s{sid:02d}.npz"
   while not path.exists():time.sleep(2)
   with np.load(path) as d:data[sid]={k:d[k] for k in d.files}
  # Equal total subject weight inside each class, avoid long records dominating prototypes.
  byclass=[[] for _ in range(5)]
  for s in split["base_train"]:
   d=data[s];zz=unit(d["z"].astype(float))
   for c in range(5):
    if np.any(d["y"]==c):byclass[c].append(zz[d["y"]==c].mean(0))
  centroids=unit(np.stack([np.mean(v,axis=0) for v in byclass]));grid=[]
  for temp,blend in itertools.product([.05,.1,.2,.5],[0.,.25,.5,1.]):
   val=np.mean([score(data[s]["y"],stream(data[s]["z"],data[s]["p"],centroids,temp,blend)) for s in split["base_val"]]);grid.append(dict(temp=temp,blend=blend,val_f1=float(val)))
  fixed=max(grid,key=lambda v:v["val_f1"]);opts={k:fixed[k] for k in ["temp","blend"]};online=[]
  for rate,threshold in itertools.product([0.,.001,.01,.05],[.6,.8,.95]):
   if rate==0 and threshold!=.6:continue
   val=np.mean([score(data[s]["y"],stream(data[s]["z"],data[s]["p"],centroids,**opts,rate=rate,threshold=threshold)) for s in split["base_val"]]);online.append(dict(rate=rate,threshold=threshold,val_f1=float(val)))
  best=max(online,key=lambda v:v["val_f1"]);onopts={k:best[k] for k in ["rate","threshold"]};results=[]
  for s in split["controller_train"]+split["evaluation"]:
   d=data[s];preds={"frozen":d["p"],"source_prototype":stream(d["z"],d["p"],centroids,**opts),"online_prototype":stream(d["z"],d["p"],centroids,**opts,**onopts)}
   assert np.array_equal(preds["online_prototype"][:31],stream(d["z"][:31],d["p"][:31],centroids,**opts,**onopts))
   for method,pr in preds.items():results.append(dict(subject=s,group="evaluation" if s in split["evaluation"] else "controller_train",method=method,**metrics(d["y"],pr)))
   np.savez_compressed(OUT/"results"/f"{name}_prototype_s{s:02d}.npz",y=d["y"],**preds)
  dest.write_text(json.dumps(dict(name=name,development_only=True,fixed_selection=fixed,online_selection=best,fixed_grid=grid,online_grid=online,centroids=centroids.tolist(),results=results,prefix_audit=True),indent=2));print("PROTOTYPE COMPLETE",name,fixed,best,flush=True)

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--seed",type=int,default=17)
 run(p.parse_args().seed)
