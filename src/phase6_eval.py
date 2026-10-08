from common import *
from phase2_temporal import ema
from adaptation import stream
import argparse,pickle,hashlib,time
OUT=ROOT/"phase6"
def load_carrier(path):
 ck=torch.load(path,map_location="cpu",weights_only=False);state=ck["state"];dim=len(state["mean"]);width=state["body.0.weight"].shape[0];m=SpectralNet(dim)
 if width!=64:m.body=nn.Sequential(nn.Linear(dim,width),nn.BatchNorm1d(width),nn.ELU(),nn.Dropout(.25),nn.Linear(width,32),nn.BatchNorm1d(32),nn.ELU())
 m.load_state_dict(state);m.eval();return m

def transform(hf,variant):
 if variant in ["full52","wide"]:return hf
 if variant=="repeat11":return np.tile(hf,(1,11))
 k=int(variant.removeprefix("context"));pad=np.pad(hf,((k-1,0),(0,0)),mode="edge");return np.concatenate([pad[j:j+len(hf)] for j in range(k)],axis=1)
def run(name,mode):
 lock=json.loads((OUT/"evaluation_lock.json").read_text());spec=lock["models"][name];path=ROOT/spec["path"];assert hashlib.sha256(path.read_bytes()).hexdigest()==spec["sha256"];m=load_carrier(path);variant=spec["variant"];destdir=OUT/("fresh_"+mode);destdir.mkdir(exist_ok=True);gate_info=None
 if mode=="controllers":
  f=OUT/"controllers"/(name+".pkl");assert hashlib.sha256(f.read_bytes()).hexdigest()==lock["controllers"][name]["sha256"]
  with open(f,"rb") as fh:gate_info=pickle.load(fh)
 for sid in lock["cohort"]:
  dest=destdir/f"{name}_s{sid:02d}.json"
  if dest.exists():continue
  metadata=OUT/"processed"/f"s{sid:02d}_meta.npz"
  while not metadata.exists():time.sleep(2)
  with np.load(metadata) as d:hf=d["features"];y=d["y"]
  x=transform(hf,variant);assert np.array_equal(x[:31],transform(hf[:31],variant));timings={};start=time.perf_counter()
  if mode=="baselines":
   p=predict(m,x);preds=dict(frozen=p,ema_08=ema(p,.8))
  else:
   lr=gate_info["lr"];preds={}
   for method in ["continuous","periodic","reset_periodic"]:preds[method],_,timings[method]=stream(m,x,method,lr,history_features=hf)
   base=np.load(OUT/"fresh_baselines"/f"{name}_s{sid:02d}.npz")["frozen"]
   preds["source_selected"]=base if gate_info["best_lr"]==0 else preds["continuous"]
   for method,g in gate_info["gates"].items():
    def gate(f):
     values=g.values(f)[0].copy()
     if f[3]==0.:values[1]=0.
     return int(np.r_[0.,values].argmax())
    preds[method],_,timings[method]=stream(m,x,"controller",lr,gate,hf)
  mask=np.zeros(len(y),bool)
  for b in np.flatnonzero((y[1:]!=y[:-1])&(y[1:]>=0)&(y[:-1]>=0))+1:mask[max(0,b-2):min(len(y),b+3)]=True
  results=[]
  for method,pr in preds.items():
   assert np.isfinite(pr).all() and np.allclose(pr.sum(1),1,atol=2e-6)
   results.append(dict(subject=sid,method=method,**metrics(y,pr),transition_f1=metrics(y[mask],pr[mask])["macro_f1"]))
  np.savez_compressed(dest.with_suffix(".npz"),y=y,**preds);dest.write_text(json.dumps(dict(name=name,variant=variant,source=spec["source"],seed=spec["seed"],subject=sid,results=results,timings=timings,seconds=time.perf_counter()-start,affinity=sorted(os.sched_getaffinity(0)),lock_sha256=hashlib.sha256((OUT/"evaluation_lock.json").read_bytes()).hexdigest()),indent=2));print("NEW COHORT",name,mode,sid,flush=True)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--name",required=True);p.add_argument("--mode",required=True,choices=["baselines","controllers"]);args=p.parse_args();run(args.name,args.mode)
