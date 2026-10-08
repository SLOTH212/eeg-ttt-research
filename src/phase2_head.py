"""NumPy head adaptation, predict-before-update, source-only hyperparameter selection."""
from common import *
import argparse,itertools,time
OUT=ROOT/"phase2"
def f1(y,p):
 k=y>=0;cm=np.bincount(y[k]*5+p[k].argmax(1),minlength=25).reshape(5,5)
 return float(np.mean(2*cm.diagonal()/np.maximum(cm.sum(0)+cm.sum(1),1)))
def softmax(v):
 e=np.exp(v-v.max());return e/e.sum()
def stream(z,source_p,W0,b0,objective,lr,anchor,alpha=.8,radius=.1):
 # Labels never enter this function. Original source predictions provide a fixed teacher.
 W=W0.astype(np.float64).copy();b=b0.astype(np.float64).copy();out=np.empty((len(z),5));teacher=source_p[0].astype(float).copy();drifts=[]
 maxnorm=radius*max(np.linalg.norm(W0),1.);bnorm=radius*max(np.linalg.norm(b0),1.)
 for t,(v,src) in enumerate(zip(z,source_p)):
  q=softmax(W@v+b);out[t]=q
  teacher=alpha*src+(1-alpha)*teacher
  if objective=="temporal":g=q-teacher
  elif objective=="pseudo":g=q.copy();g[np.argmax(src)]-=1
  elif objective=="entropy":
   l=np.log(np.maximum(q,1e-12));g=q*(np.sum(q*l)-l)
  else:raise ValueError(objective)
  gw=np.outer(g,v)+anchor*(W-W0);gb=g+anchor*(b-b0)
  norm=np.sqrt(np.sum(gw*gw)+np.sum(gb*gb));scale=min(1.,1/max(norm,1e-12))
  W-=lr*scale*gw;b-=lr*scale*gb
  d=W-W0;W=W0+d*min(1.,maxnorm/max(np.linalg.norm(d),1e-12))
  d=b-b0;b=b0+d*min(1.,bnorm/max(np.linalg.norm(d),1e-12))
  if t%100==0:drifts.append(float(np.linalg.norm(W-W0)))
 return out,drifts
def run(seed=17):
 args=argparse.Namespace(seed=seed)
 for kind,fold in itertools.product(["spectral","eegnet"],["A","B"]):
  name=f"{kind}_{fold}_{args.seed}";dest=OUT/"results"/(name+"_head.json")
  if dest.exists():print("CACHED",name,flush=True);continue
  m,ck=load_model(ROOT/"results"/name/"model.pt");split=CFG["folds"][fold]
  W=m.head.weight.detach().numpy();b=m.head.bias.detach().numpy();data={}
  for s in split["base_val"]+split["controller_train"]+split["evaluation"]:
   path=OUT/"cache"/f"{name}_s{s:02d}.npz"
   while not path.exists():time.sleep(2)
   with np.load(path) as d:data[s]={k:d[k] for k in d.files}
  selections={};grids={}
  for objective in ["entropy","temporal","pseudo"]:
   grid=[]
   for lr,anchor in itertools.product([0.,.0001,.001,.01],[0.,.1,1.]):
    if lr==0 and anchor!=0:continue
    opts=dict(objective=objective,lr=lr,anchor=anchor)
    scores=[f1(data[s]["y"],stream(data[s]["z"],data[s]["p"],W,b,**opts)[0]) for s in split["base_val"]]
    grid.append(dict(**opts,val_f1=float(np.mean(scores))))
   best=max(grid,key=lambda r:r["val_f1"]);selections[objective]=best;grids[objective]=grid
   print("SELECT",name,best,flush=True)
  results=[]
  for s in split["controller_train"]+split["evaluation"]:
   d=data[s];preds={"frozen":d["p"]};drifts={}
   for objective,sel in selections.items():
    opts={k:v for k,v in sel.items() if k!="val_f1"};preds[objective],drifts[objective]=stream(d["z"],d["p"],W,b,**opts)
    prefix,_=stream(d["z"][:31],d["p"][:31],W,b,**opts);assert np.array_equal(prefix,preds[objective][:31])
    assert np.allclose(preds[objective][0],d["p"][0],atol=2e-6)
   for method,pr in preds.items():results.append(dict(subject=s,group="evaluation" if s in split["evaluation"] else "controller_train",method=method,**metrics(d["y"],pr),drift=drifts.get(method,[])))
   np.savez_compressed(OUT/"results"/f"{name}_head_s{s:02d}.npz",y=d["y"],**preds)
  dest.write_text(json.dumps(dict(name=name,development_only=True,selections=selections,grids=grids,results=results,prefix_audit=True,affinity=sorted(os.sched_getaffinity(0))),indent=2));print("HEAD COMPLETE",name,flush=True)

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--seed",type=int,default=17)
 run(p.parse_args().seed)
