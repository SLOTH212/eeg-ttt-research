from common import *
from phase14_probe import subject
from phase2_head import stream as reference_stream,f1
from phase2_temporal import ema
from adaptation import score_diagnostics
from controllers import fit_gate
import argparse,pickle,itertools,time
OUT=ROOT/"phase15"
def probabilities(z,W,b):
 logits=z@W.T+b;exp=np.exp(logits-logits.max(axis=-1,keepdims=True));return exp/exp.sum(axis=-1,keepdims=True)
def ent(p):return float(-np.sum(p*np.log(np.maximum(p,1e-12))))
def stream(z,source_p,W0,b0,hf,objective,lr,anchor,mode="continuous",gate=None,collect=False):
 W=W0.copy();b=b0.copy();teacher=source_p[0].copy();out=[];diagnostics=[];actions=[];Wlimit=.1*max(np.linalg.norm(W0),1.);blimit=.1*max(np.linalg.norm(b0),1.);previous=None
 for t,(v,src) in enumerate(zip(z,source_p)):
  q=probabilities(v,W,b);teacher=.8*src+.2*teacher
  if objective=="temporal":g=q-teacher
  elif objective=="entropy":
   lp=np.log(np.maximum(q,1e-12));g=q*(np.sum(q*lp)-lp)
  else:raise ValueError(objective)
  gw=np.outer(g,v)+anchor*(W-W0);gb=g+anchor*(b-b0);norm=np.sqrt(np.sum(gw*gw)+np.sum(gb*gb));scale=min(1.,1/max(norm,1e-12));Wc=W-lr*scale*gw;bc=b-lr*scale*gb;d=Wc-W0;Wc=W0+d*min(1.,Wlimit/max(np.linalg.norm(d),1e-12));d=bc-b0;bc=b0+d*min(1.,blimit/max(np.linalg.norm(d),1e-12));after=probabilities(v,Wc,bc);drift=float(np.sqrt(np.sum((W-W0)**2)+np.sum((b-b0)**2)));distance=0. if previous is None else float(np.linalg.norm(hf[t]-previous)/np.sqrt(len(hf[t])));features=np.r_[ent(q),norm,np.abs(after-q).sum(),drift,q.max(),ent(after),q,distance,float(previous is None)]
  if objective=="temporal":features=np.r_[features,teacher,ent(teacher),np.abs(q-teacher).sum()]
  if collect and t%20==0 and t+10<len(z):
   future=z[t+1:t+11];diagnostics.append(dict(t=t,features=features,keep=probabilities(future,W,b),write=probabilities(future,Wc,bc),reset=probabilities(future,W0,b0),current_keep=q,current_write=after,current_reset=src))
  if mode=="continuous":action=1
  elif mode=="periodic":action=int(t%20==0)
  elif mode=="reset_periodic":action=2 if (t+1)%100==0 else 1
  elif mode=="controller":action=int(gate(features))
  elif mode=="frozen":action=0
  else:raise ValueError(mode)
  out.append(q)
  if action==1:W,b=Wc,bc
  elif action==2:W,b=W0.copy(),b0.copy()
  previous=hf[t].copy();actions.append(action)
 return np.array(out),diagnostics,np.bincount(actions,minlength=3).tolist()
def run(source,representation,objective):
 name=f"{source}_{representation}_{objective}";out=OUT/name;out.mkdir(exist_ok=True)
 if (out/"evaluation.json").exists():return
 probe=ROOT/"phase14/probes"/f"{source}_{representation}";start=time.monotonic()
 while not (probe/"evaluation.json").exists():
  if time.monotonic()-start>3600:raise TimeoutError(str(probe))
  time.sleep(2)
 with (probe/"model.pkl").open("rb") as f:carrier=pickle.load(f)
 scaler=carrier["scaler"];head=carrier["model"];W0=head.coef_.copy();b0=head.intercept_.copy();assert np.array_equal(head.classes_,np.arange(5));split=json.loads((ROOT/"phase3/protocol.json").read_text())["folds"][source];controller=[7,17,27,37];evaluation=[s for s in split["evaluation"] if s not in controller]+list(range(40,59));data={}
 for sid in split["base_val"]+controller+evaluation:
  x,y=subject(sid,representation);z=scaler.transform(x).astype(np.float64);p=head.predict_proba(z);assert np.allclose(p,probabilities(z,W0,b0),rtol=1e-12,atol=1e-12);root=ROOT/("data/processed" if sid<20 else ("phase2/processed" if sid<40 else "phase6/processed"));hf=np.load(root/f"s{sid:02d}_meta.npz")["features"];data[sid]=dict(z=z,y=y,p=p,hf=hf)
 for sid in split["base_val"][:1]:
  d=data[sid];opts=dict(objective=objective,lr=.001,anchor=.1);p,_,_=stream(d["z"][:51],d["p"][:51],W0,b0,d["hf"][:51],**opts);ref,_=reference_stream(d["z"][:51],d["p"][:51],W0,b0,**opts);assert np.allclose(p,ref,atol=2e-12,rtol=2e-12)
 selection_path=out/"selection.json"
 if not selection_path.exists():
  frozen=float(np.mean([f1(data[s]["y"],data[s]["p"]) for s in split["base_val"]]));grid=[dict(lr=0.,anchor=0.,val_f1=frozen)]
  for lr,anchor in itertools.product([1e-5,1e-4,.001,.01],[0.,.1]):
   scores=[]
   for sid in split["base_val"]:
    d=data[sid];p,_=reference_stream(d["z"],d["p"],W0,b0,objective,lr,anchor);scores.append(f1(d["y"],p))
   row=dict(lr=lr,anchor=anchor,val_f1=float(np.mean(scores)));grid.append(row);print("HEAD LR",name,row,flush=True)
  selection_path.write_text(json.dumps(dict(grid=grid,best=max(grid,key=lambda v:v["val_f1"]),positive=max(grid[1:],key=lambda v:v["val_f1"])),indent=2))
 sel=json.loads(selection_path.read_text());opts=dict(objective=objective,lr=sel["positive"]["lr"],anchor=sel["positive"]["anchor"]);diag=out/"controller_rows.json"
 if not diag.exists():
  rows=[]
  for sid in controller:
   d=data[sid]
   for mode in ["continuous","periodic"]:
    _,ds,_=stream(d["z"],d["p"],W0,b0,d["hf"],mode=mode,collect=True,**opts);rows.extend(score_diagnostics(ds,d["y"],sid,mode))
  diag.write_text(json.dumps(rows))
 rows=json.loads(diag.read_text());gates={typ:fit_gate(rows,typ) for typ in ["future","current","confidence"]};results=[];audits=[]
 for sid in evaluation:
  d=data[sid];path=out/f"predictions_s{sid:02d}.npz"
  if path.exists():
   saved=np.load(path);preds={k:saved[k] for k in saved.files if k!="y"}
  else:
   preds=dict(frozen=d["p"],weak_ema=ema(d["p"],.8))
   for mode in ["continuous","periodic","reset_periodic"]:preds[mode],_,_=stream(d["z"],d["p"],W0,b0,d["hf"],mode=mode,**opts)
   preds["source_selected_update"]=preds["frozen"] if sel["best"]["lr"]==0 else preds["continuous"]
   for typ,g in gates.items():
    def choose(features):
     values=g.values(features)[0].copy()
     if features[3]==0.:values[1]=0.
     return int(np.r_[0.,values].argmax())
    pr,_,actions=stream(d["z"],d["p"],W0,b0,d["hf"],mode="controller",gate=choose,**opts);preds[typ]=pr;prefix,_,_=stream(d["z"][:31],d["p"][:31],W0,b0,d["hf"][:31],mode="controller",gate=choose,**opts);assert np.array_equal(prefix,pr[:31]);audits.append(dict(subject=sid,method=typ,prefix_exact=True,actions=actions))
   np.savez_compressed(path,y=d["y"],**preds)
  mask=np.zeros(len(d["y"]),bool);y=d["y"]
  for boundary in np.flatnonzero((y[1:]!=y[:-1])&(y[1:]>=0)&(y[:-1]>=0))+1:mask[max(0,boundary-2):min(len(y),boundary+3)]=True
  for method,pr in preds.items():results.append(dict(subject=sid,cohort="older" if sid>=40 else ("middle_dev" if sid>=20 else "young_dev"),method=method,**metrics(y,pr),transition_f1=metrics(y[mask],pr[mask])["macro_f1"]))
  print("HEAD EVAL",name,sid,flush=True)
 (out/"evaluation.json").write_text(json.dumps(dict(source=source,representation=representation,objective=objective,selection=sel,results=results,audits=audits,exploratory=True,reference_stream_match=True),indent=2));print("HEAD RETENTION COMPLETE",name,flush=True)
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--source",required=True);parser.add_argument("--representation",required=True);parser.add_argument("--objective",required=True);a=parser.parse_args();run(a.source,a.representation,a.objective)
