"""Development-only causal temporal baselines; no claim of independent confirmation."""
from common import *
import argparse, hashlib, itertools, time
OUT=ROOT/"phase2"
OUT=ROOT/"phase2"; rows=[]
def score(y,p):
 k=y>=0; a=y[k]*5+p[k].argmax(1);cm=np.bincount(a,minlength=25).reshape(5,5)
 return float(np.mean(2*cm.diagonal()/np.maximum(cm.sum(0)+cm.sum(1),1)))
def temporal(p,A,prior,strength=1.,temperature=1.,correction=0.):
 # q at t only uses posterior at t-1 and current frozen classifier probabilities.
 emit=np.maximum(p,1e-12)**(1/temperature)/np.maximum(prior,1e-12)**correction
 q=prior.copy();out=np.empty_like(p,dtype=np.float64)
 for t,e in enumerate(emit):
  pred=q@A if t else prior
  q=e*np.maximum(pred,1e-12)**strength;q/=q.sum();out[t]=q
 return out
def ema(p,alpha):
 out=np.empty_like(p);q=p[0].copy()
 for i,v in enumerate(p):q=alpha*v+(1-alpha)*q;out[i]=q
 return out
def run(seed=17):
 args=argparse.Namespace(seed=seed)
 for kind,fold in itertools.product(["spectral","eegnet"],["A","B"]):
  name=f"{kind}_{fold}_{args.seed}";dest=OUT/"results"/(name+"_temporal.json")
  if dest.exists():print("CACHED RESULT",name,flush=True);continue
  m,ck=load_model(ROOT/"results"/name/"model.pt");split=CFG["folds"][fold];data={}
  for sid in CFG["subject_ids"]:
   cache=OUT/"cache"/f"{name}_s{sid:02d}.npz"
   if not cache.exists():
    x,y=load_subject(sid,kind);zs=[]
    with torch.no_grad():
     for a in range(0,len(x),128):zs.append(m.features(torch.tensor(np.array(x[a:a+128]),dtype=torch.float32)).numpy())
    z=np.concatenate(zs)
    with torch.no_grad():pr=m.head(torch.tensor(z)).softmax(1).numpy()
    np.savez_compressed(cache,z=z,p=pr,y=y)
    # Stateless encoder prefix check, independent of future samples.
    direct=predict(m,x[:31]);assert np.allclose(direct,pr[:31],atol=2e-6),np.max(abs(direct-pr[:31]))
   d=np.load(cache);data[sid]=(d["y"],d["p"])
   print("CACHE",name,sid,flush=True)
  counts=np.ones((5,5),dtype=float);freq=np.ones(5)
  for sid in split["base_train"]:
   y=data[sid][0];freq+=np.bincount(y[y>=0],minlength=5)
   valid=(y[:-1]>=0)&(y[1:]>=0);counts+=np.bincount(y[:-1][valid]*5+y[1:][valid],minlength=25).reshape(5,5)
  A=counts/counts.sum(1,keepdims=True);prior=freq/freq.sum()
  candidates=[]
  for strength,temp,corr in itertools.product([0.,.25,.5,1.],[.75,1.,1.5],[0.,.5,1.]):
   opts=dict(strength=strength,temperature=temp,correction=corr)
   v=np.mean([score(data[s][0],temporal(data[s][1],A,prior,**opts)) for s in split["base_val"]])
   candidates.append(dict(**opts,val_f1=float(v)))
  best=max(candidates,key=lambda x:x["val_f1"]);opts={k:best[k] for k in ["strength","temperature","correction"]}
  pp=data[split["base_val"][0]][1];assert np.array_equal(temporal(pp,A,prior,**opts)[:31],temporal(pp[:31],A,prior,**opts))
  result=[]
  for group in ["base_val","controller_train","evaluation"]:
   for sid in split[group]:
    y,pr=data[sid];preds={"frozen":pr,"ema_08":ema(pr,.8),"ema_05":ema(pr,.5),"source_selected_filter":temporal(pr,A,prior,**opts)}
    transition=np.zeros(len(y),bool)
    for b in np.flatnonzero((y[1:]!=y[:-1])&(y[1:]>=0)&(y[:-1]>=0))+1:transition[max(0,b-2):min(len(y),b+3)]=True
    for method,pred in preds.items():
     result.append(dict(subject=sid,group=group,method=method,macro_f1=score(y,pred),transition_f1=score(y[transition],pred[transition]),**{"metrics":metrics(y,pred)}))
    np.savez_compressed(OUT/"results"/f"{name}_temporal_s{sid:02d}.npz",y=y,**preds)
  dest.write_text(json.dumps(dict(name=name,development_only=True,selection_subjects=split["base_val"],candidates=candidates,selected=best,transition_matrix=A.tolist(),prior=prior.tolist(),results=result,affinity=sorted(os.sched_getaffinity(0)),prefix_audit=True),indent=2))
  print("COMPLETE",name,best,flush=True)
 print("TEMPORAL PHASE DONE",args.seed,flush=True)

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--seed",type=int,default=17)
 run(p.parse_args().seed)
