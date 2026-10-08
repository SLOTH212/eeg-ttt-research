from common import *
from adaptation import Adapter,stream,score_diagnostics
from controllers import fit_gate
import itertools,time
OUT=ROOT/"phase5";OUT.mkdir(exist_ok=True);(OUT/"results").mkdir(exist_ok=True)
protocol=dict(purpose="Exploratory retention revisit on stronger strictly causal11-window model; all participants already examined",sources=["young","mixed"],seeds=[17,29,43],base_val=[5,15,25,35],controller_train=[7,17,27,37],evaluation=[6,8,9,16,18,19,26,28,29,36,38],lrs=[1e-5,1e-4,1e-3,.01],allowed_cpu_ids=list(range(8,24)),gpu_allowed=False)
(OUT/"protocol.json").write_text(json.dumps(protocol,indent=2))
def subj(s):
 base=ROOT/("data/processed" if s<20 else "phase2/processed");d=np.load(base/f"s{s:02d}_meta.npz");hf=d["features"];pad=np.pad(hf,((10,0),(0,0)),mode="edge");x=np.concatenate([pad[j:j+len(hf)] for j in range(11)],axis=1);return x,d["y"],hf
for source,seed in itertools.product(protocol["sources"],protocol["seeds"]):
 name=f"context11_{source}_{seed}";out=OUT/"results"/name;out.mkdir(exist_ok=True)
 if (out/"evaluation.json").exists():continue
 ck=torch.load(ROOT/"phase4/results"/name/"model.pt",map_location="cpu",weights_only=False);m=SpectralNet(ck["dim"]);m.load_state_dict(ck["state"]);m.eval();data={s:subj(s) for s in protocol["base_val"]+protocol["controller_train"]+protocol["evaluation"]}
 if not (out/"pulse.json").exists():
  pulses=[]
  for s in protocol["controller_train"]:
   x,y,hf=data[s]
   for lr in protocol["lrs"]:
    a=Adapter(m,lr)
    for t in range(0,len(x)-10,20):
     a.restore(a.source);pr,candidate,signals=a.candidate(torch.tensor(x[t:t+1]));future=torch.tensor(x[t+1:t+11]);
     with torch.no_grad():
      keep=a.model(future).softmax(1).numpy();a.restore(candidate);write=a.model(future).softmax(1).numpy()
     for h in [1,4,10]:
      yy=y[t+1:t+1+h];k=yy>=0
      if k.any():
       gain=float((np.log(np.maximum(write[:h][k,yy[k]],1e-9))-np.log(np.maximum(keep[:h][k,yy[k]],1e-9))).mean());pulses.append(dict(subject=s,t=t,h=h,lr=lr,gain=gain))
  (out/"pulse.json").write_text(json.dumps(pulses));print("PULSE",name,flush=True)
 if not (out/"selection.json").exists():
  selection=[dict(lr=0.,f1=float(np.mean([metrics(data[s][1],predict(m,data[s][0]))["macro_f1"] for s in protocol["base_val"]])))]
  for lr in protocol["lrs"]:
   scores=[]
   for s in protocol["base_val"]:
    x,y,hf=data[s];pr,_,_=stream(m,x,"continuous",lr,history_features=hf);scores.append(metrics(y,pr)["macro_f1"])
   selection.append(dict(lr=lr,f1=float(np.mean(scores))));print("LR",name,lr,selection[-1]["f1"],flush=True)
  best=max(selection,key=lambda v:v["f1"]);positive=max(selection[1:],key=lambda v:v["f1"]);(out/"selection.json").write_text(json.dumps(dict(grid=selection,best=best,best_positive=positive),indent=2))
 selection=json.loads((out/"selection.json").read_text());lr=selection["best_positive"]["lr"]
 if not (out/"controller_rows.json").exists():
  rows=[]
  for s in protocol["controller_train"]:
   x,y,hf=data[s]
   for mode in ["continuous","periodic"]:
    pr,ds,_=stream(m,x,mode,lr,history_features=hf,collect=True,subject=s);rows.extend(score_diagnostics(ds,y,s,mode))
  (out/"controller_rows.json").write_text(json.dumps(rows));print("CONTROLLER DATA",name,flush=True)
 rows=json.loads((out/"controller_rows.json").read_text());gates={typ:fit_gate(rows,typ) for typ in ["future","confidence","current"]};results=[]
 for s in protocol["evaluation"]:
  x,y,hf=data[s];path=out/f"predictions_s{s:02d}.npz"
  if path.exists():
   with np.load(path) as f:preds={k:f[k] for k in f.files if k!="y"}
  else:
   preds={"frozen":predict(m,x)}
   for mode in ["continuous","periodic","reset_periodic"]:preds[mode],_,_=stream(m,x,mode,lr,history_features=hf)
   preds["source_selected"]=preds["frozen"] if selection["best"]["lr"]==0 else preds["continuous"]
   for typ,g in gates.items():
    def gate(f):
     values=g.values(f)[0].copy()
     if f[3]==0.:values[1]=0.
     return int(np.r_[0.,values].argmax())
    preds["identity_"+typ],_,_=stream(m,x,"controller",lr,gate,hf)
   np.savez_compressed(path,y=y,**preds)
  for method,pr in preds.items():results.append(dict(subject=s,cohort="young" if s<20 else "middle",method=method,**metrics(y,pr)))
  print("CONTEXT EVAL",name,s,flush=True)
 (out/"evaluation.json").write_text(json.dumps(dict(name=name,results=results,selection=selection,development_only=True),indent=2));print("CONTEXT COMPLETE",name,flush=True)
