from common import *
from phase2_temporal import temporal,ema
from phase2_head import stream as head_stream
from phase2_prototype import stream as prototype_stream
from adaptation import stream as adaptation_stream
from controllers import fit_gate
import argparse,time,hashlib
OUT=ROOT/"phase2";LOCK=json.loads((OUT/"evaluation_lock.json").read_text())
def run(name,mode):
 for f,h in LOCK["sha256"].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h,f
 saved=OUT/"fresh";saved.mkdir(exist_ok=True);old=ROOT/"results"/name
 m,ck=load_model(old/"model.pt");kind=ck["kind"];spec=LOCK["models"][name]
 gates={}
 if mode=="controllers":
  rows=json.loads((old/"controller_diagnostics.json").read_text())+json.loads((old/"source_replay_diagnostics.json").read_text())
  gates={typ:fit_gate(rows,typ) for typ in ["future","confidence","current"]}
 for sid in LOCK["cohort"]:
  dest=saved/f"{name}_{mode}_s{sid:02d}.json"
  if dest.exists():continue
  path=OUT/"processed"/f"s{sid:02d}_meta.npz"
  while not path.exists():time.sleep(2)
  # meta npz writer completes before loading; retries handle an actively written archive.
  for attempt in range(30):
   try:
    with np.load(path) as d:y=d["y"];hf=d["features"]
    break
   except (ValueError,EOFError,OSError):time.sleep(1)
  x=hf if kind=="spectral" else np.load(OUT/"processed"/f"s{sid:02d}_x.npy",mmap_mode="r")
  preds={};timings={};start=time.perf_counter()
  if mode=="baselines":
   zs=[]
   with torch.no_grad():
    for a in range(0,len(x),128):zs.append(m.features(torch.tensor(np.array(x[a:a+128]),dtype=torch.float32)).numpy())
    z=np.concatenate(zs);p=m.head(torch.tensor(z)).softmax(1).numpy()
   np.savez_compressed(OUT/"cache"/f"{name}_s{sid:02d}.npz",z=z,p=p,y=y)
   assert np.allclose(p[:31],predict(m,x[:31]),atol=2e-6)
   ts=spec["temporal"];opts={k:ts["selected"][k] for k in ["strength","temperature","correction"]}
   preds={"frozen":p,"ema_08":ema(p,.8),"ema_05":ema(p,.5),"source_selected_filter":temporal(p,np.array(ts["transition_matrix"]),np.array(ts["prior"]),**opts)}
   W=m.head.weight.detach().numpy();b=m.head.bias.detach().numpy()
   for objective,sel in spec["head"].items():
    opts={k:v for k,v in sel.items() if k!="val_f1"};preds["head_"+objective],_=head_stream(z,p,W,b,**opts)
   ps=spec["prototype"];opts={k:ps["fixed_selection"][k] for k in ["temp","blend"]};onopts={k:ps["online_selection"][k] for k in ["rate","threshold"]}
   preds["source_prototype"]=prototype_stream(z,p,np.array(ps["centroids"]),**opts)
   preds["online_prototype"]=prototype_stream(z,p,np.array(ps["centroids"]),**opts,**onopts)
  else:
   for typ,g in gates.items():
    def gate(f):
     values=g.values(f)[0].copy()
     if f[3]==0.:values[1]=0.
     return int(np.r_[0.,values].argmax())
    preds["identity_"+typ],_,timings[typ]=adaptation_stream(m,x,"controller",spec["controller_lr"],gate,hf)
  transition=np.zeros(len(y),bool)
  for b in np.flatnonzero((y[1:]!=y[:-1])&(y[1:]>=0)&(y[:-1]>=0))+1:transition[max(0,b-2):min(len(y),b+3)]=True
  results=[]
  for method,pr in preds.items():
   assert np.isfinite(pr).all() and np.allclose(pr.sum(1),1,atol=2e-6)
   # Canonicalize float32 sum rounding before log-loss calculations.
   pr=pr.astype(float);pr/=pr.sum(1,keepdims=True)
   results.append(dict(method=method,subject=sid,**metrics(y,pr),transition_f1=metrics(y[transition],pr[transition])["macro_f1"]))
  np.savez_compressed(saved/f"{name}_{mode}_s{sid:02d}.npz",y=y,**preds)
  dest.write_text(json.dumps(dict(name=name,mode=mode,subject=sid,results=results,seconds=time.perf_counter()-start,timings=timings,lock_sha256=hashlib.sha256((OUT/"evaluation_lock.json").read_bytes()).hexdigest(),affinity=sorted(os.sched_getaffinity(0))),indent=2))
  print("FRESH",name,mode,sid,round(time.perf_counter()-start,2),flush=True)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--name",required=True);p.add_argument("--mode",choices=["baselines","controllers"],required=True);args=p.parse_args();run(args.name,args.mode)
