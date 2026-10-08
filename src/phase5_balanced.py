from common import *
from adaptation import stream
from controllers import fit_gate
import itertools,time
OUT=ROOT/"phase5_balanced";OUT.mkdir(exist_ok=True);(OUT/"results").mkdir(exist_ok=True)
protocol=json.loads((ROOT/"phase5/protocol.json").read_text());protocol.update(purpose="Exploratory loss-alignment test: controller learns source-frequency-weighted future/current CE gains; target labels never enter policy",powers=[.5,1.],weight_rule="w_c=source_prior_c^(-power), normalized so sum(source_prior*w)=1; source training labels only")
(OUT/"protocol.json").write_text(json.dumps(protocol,indent=2));cfg3=json.loads((ROOT/"phase3/protocol.json").read_text())
def subj(s):
 base=ROOT/("data/processed" if s<20 else "phase2/processed");d=np.load(base/f"s{s:02d}_meta.npz");hf=d["features"];pad=np.pad(hf,((10,0),(0,0)),mode="edge");return np.concatenate([pad[j:j+len(hf)] for j in range(11)],axis=1),d["y"],hf
for source,seed in itertools.product(protocol["sources"],protocol["seeds"]):
 name=f"context11_{source}_{seed}";out=OUT/"results"/name;out.mkdir(exist_ok=True)
 if (out/"evaluation.json").exists():continue
 selection=ROOT/"phase5/results"/name/"selection.json"
 while not selection.exists():time.sleep(5)
 lr=json.loads(selection.read_text())["best_positive"]["lr"];ck=torch.load(ROOT/"phase4/results"/name/"model.pt",map_location="cpu",weights_only=False);m=SpectralNet(ck["dim"]);m.load_state_dict(ck["state"]);m.eval();freq=np.ones(5)
 for s in cfg3["folds"][source]["base_train"]:
  _,y,_=subj(s);freq+=np.bincount(y[y>=0],minlength=5)
 prior=freq/freq.sum();cache=out/"source_counterfactuals.npz"
 if not cache.exists():
  collect={k:[] for k in ["subject","t","trajectory","features","keep","write","reset","current_keep","current_write","current_reset","future_y","current_y"]}
  for s in protocol["controller_train"]:
   x,y,hf=subj(s)
   for mode in ["continuous","periodic"]:
    _,ds,_=stream(m,x,mode,lr,history_features=hf,collect=True,subject=s)
    for d in ds:
     for key in ["t","features","keep","write","reset","current_keep","current_write","current_reset"]:collect[key].append(d[key])
     collect["subject"].append(s);collect["trajectory"].append(mode);collect["future_y"].append(y[d["t"]+1:d["t"]+11]);collect["current_y"].append(y[d["t"]])
  np.savez_compressed(cache,**{k:np.array(v) for k,v in collect.items()});print("COUNTERFACTUAL CACHE",name,flush=True)
 d=np.load(cache);gates={};audit=[]
 original=ROOT/"phase5/results"/name/"controller_rows.json"
 if original.exists():
  old={(v["subject"],v["t"],v["trajectory"]):v for v in json.loads(original.read_text()) if v["h"]==4}
 else:old={}
 for power in protocol["powers"]:
  weights=prior**(-power);weights/=prior@weights;rows=[]
  for i in range(len(d["t"])):
   yy=d["future_y"][i,:4];k=yy>=0
   if not k.any():continue
   losses={key:-np.log(np.maximum(d[key][i,:4][k,yy[k]],1e-9)) for key in ["keep","write","reset"]}
   key=(int(d["subject"][i]),int(d["t"][i]),str(d["trajectory"][i]))
   if key in old:
    err=abs(float((losses["keep"]-losses["write"]).mean())-old[key]["write_gain"]);assert err<2e-6;audit.append(err)
   cy=int(d["current_y"][i]);current={key:-np.log(max(float(d["current_"+key][i,cy]),1e-9)) if cy>=0 else None for key in ["keep","write","reset"]}
   rows.append(dict(subject=key[0] if isinstance(key,tuple) else int(d["subject"][i]),t=int(d["t"][i]),h=4,features=d["features"][i].tolist(),write_gain=float(((losses["keep"]-losses["write"])*weights[yy[k]]).mean()),reset_gain=float(((losses["keep"]-losses["reset"])*weights[yy[k]]).mean()),current_write_gain=(current["keep"]-current["write"])*weights[cy] if cy>=0 else None,current_reset_gain=(current["keep"]-current["reset"])*weights[cy] if cy>=0 else None))
  for typ in ["future","confidence","current"]:gates[f"balanced_{power}_{typ}"]=fit_gate(rows,typ)
 results=[]
 for s in protocol["evaluation"]:
  x,y,hf=subj(s);path=out/f"predictions_s{s:02d}.npz"
  if path.exists():
   with np.load(path) as p:preds={key:p[key] for key in p.files if key!="y"}
  else:
   preds={}
   for method,g in gates.items():
    def gate(f):
     values=g.values(f)[0].copy()
     if f[3]==0.:values[1]=0.
     return int(np.r_[0.,values].argmax())
    preds[method],_,_=stream(m,x,"controller",lr,gate,hf)
   np.savez_compressed(path,y=y,**preds)
  for method,pr in preds.items():results.append(dict(subject=s,cohort="young" if s<20 else "middle",method=method,**metrics(y,pr)))
  print("BALANCED EVAL",name,s,flush=True)
 (out/"evaluation.json").write_text(json.dumps(dict(name=name,results=results,source_prior=prior.tolist(),counterfactual_replay_max_error=max(audit,default=None),lr=lr,development_only=True),indent=2));print("BALANCED COMPLETE",name,flush=True)
