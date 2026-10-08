from common import *
from adaptation import Adapter,entropy,score_diagnostics
from phase2_temporal import ema
from phase6_eval import load_carrier,transform
from controllers import fit_gate,Gate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import argparse,time
class TemporalAdapter(Adapter):
 def candidate_with_teacher(self,x,teacher):
  self.model.zero_grad(set_to_none=True);logits=self.model(x);p=logits.softmax(1);q=torch.tensor(teacher,dtype=torch.float32)[None];q=q/q.sum(1,keepdim=True);loss=-(q*logits.log_softmax(1)).sum();loss.backward();old=self.snapshot();gradnorm=float(sum(v.grad.square().sum() for v in self.params).sqrt());drift=self.drift()
  with torch.no_grad():
   for v,s,scale in zip(self.params,self.source,self.scales):
    v.add_(v.grad,alpha=-self.lr);d=v-s;limit=self.radius*scale*np.sqrt(v.numel());norm=float(d.norm())
    if norm>limit:v.copy_(s+d*(limit/norm))
   candidate=self.snapshot();after=self.model(x).softmax(1)
  sig=np.array([float(entropy(p).detach()),gradnorm,float((after-p.detach()).abs().sum()),drift,float(p.detach().max()),float(entropy(after)),*p.detach().numpy()[0]])
  self.restore(old);return p.detach().numpy()[0],candidate,sig

def temporal_stream(model,x,teacher,hf,lr,mode="continuous",gate=None,collect=False):
 a=TemporalAdapter(model,lr);pred=[];ds=[];previous=None;actions=[]
 for t in range(len(x)):
  xt=torch.tensor(np.array(x[t:t+1]),dtype=torch.float32);p,candidate,sig=a.candidate_with_teacher(xt,teacher[t]);distance=0. if previous is None else float(np.linalg.norm(hf[t]-previous)/np.sqrt(len(hf[t])));q=teacher[t];f=np.r_[sig,distance,float(previous is None),q,-np.sum(q*np.log(np.maximum(q,1e-9))),np.abs(p-q).sum()]
  if collect and t%20==0 and t+10<len(x):
   old=a.snapshot();future=torch.tensor(np.array(x[t+1:t+11]),dtype=torch.float32)
   with torch.no_grad():
    keep=a.model(future).softmax(1).numpy();current_keep=a.model(xt).softmax(1).numpy()[0];a.restore(candidate);write=a.model(future).softmax(1).numpy();current_write=a.model(xt).softmax(1).numpy()[0];a.restore(a.source);reset=a.model(future).softmax(1).numpy();current_reset=a.model(xt).softmax(1).numpy()[0]
   a.restore(old);ds.append(dict(t=t,features=f,keep=keep,write=write,reset=reset,current_keep=current_keep,current_write=current_write,current_reset=current_reset))
  action=1 if mode=="continuous" else (int(t%20==0) if mode=="periodic" else (2 if (t+1)%100==0 else 1))
  if mode=="controller":action=int(gate(f))
  pred.append(p)
  if action==1:a.restore(candidate)
  elif action==2:a.restore(a.source)
  previous=hf[t].copy();actions.append(action)
 return np.array(pred),ds,np.bincount(actions,minlength=3).tolist()

def teacher_confidence_gate(rows):
 r=[v for v in rows if v["h"]==4];indices=[0,4,18,19];x=np.array([v["features"] for v in r])[:,indices];y=np.array([[v["write_gain"],v["reset_gain"]] for v in r]);ss=np.array([v["subject"] for v in r]);w=np.array([len(ss)/(len(set(ss))*np.sum(ss==s)) for s in ss]);reg=make_pipeline(StandardScaler(),Ridge(alpha=100.));reg.fit(x,y,ridge__sample_weight=w);return Gate(reg,indices)

def run(name):
 OUT=ROOT/"phase12";OUT.mkdir(exist_ok=True);out=OUT/name;out.mkdir(exist_ok=True)
 if (out/"evaluation.json").exists():return
 context=name.startswith("context11")
 if context:
  _,source,seed=name.split("_");m=load_carrier(ROOT/"phase4/results"/name/"model.pt");split=json.loads((ROOT/"phase3/protocol.json").read_text())["folds"][source];controller=[7,17,27,37];evaluation=[6,8,9,16,18,19,26,28,29,36,38];kind="spectral"
 else:
  m,ck=load_model(ROOT/"results"/name/"model.pt");kind=ck["kind"];split=CFG["folds"][ck["fold"]];controller=split["controller_train"];evaluation=split["evaluation"]
 data={}
 for sid in split["base_val"]+controller+evaluation:
  root=ROOT/("data/processed" if sid<20 else "phase2/processed");meta=np.load(root/f"s{sid:02d}_meta.npz");hf=meta["features"];y=meta["y"];x=hf if kind=="spectral" else np.load(root/f"s{sid:02d}_x.npy",mmap_mode="r")
  if context:x=transform(x,"context11")
  data[sid]=dict(x=x,y=y,hf=hf,p=predict(m,x))
 lrs=[1e-6,1e-5,1e-4,1e-3] if kind=="eegnet" else [1e-4,1e-3,.01];alphas=[.2,.5,.8];rows=[]
 for alpha in alphas:
  for lr in lrs:
   path=out/f"selection_a{alpha}_lr{lr}.json"
   if path.exists():rows.append(json.loads(path.read_text()));continue
   scores=[]
   for sid in split["base_val"]:
    d=data[sid];q=ema(d["p"],alpha);pr,_,_=temporal_stream(m,d["x"],q,d["hf"],lr);scores.append(metrics(d["y"],pr)["macro_f1"])
   row=dict(alpha=alpha,lr=lr,val_f1=float(np.mean(scores)));path.write_text(json.dumps(row));rows.append(row);print("TEMPORAL BN SELECT",name,row,flush=True)
 selected=max(rows,key=lambda v:v["val_f1"]);alpha=selected["alpha"];lr=selected["lr"];frozen_val=float(np.mean([metrics(data[s]["y"],data[s]["p"])["macro_f1"] for s in split["base_val"]]));teacher_scores={a:float(np.mean([metrics(data[s]["y"],ema(data[s]["p"],a))["macro_f1"] for s in split["base_val"]])) for a in alphas};teacher_best=max(teacher_scores,key=teacher_scores.get);(out/"selection.json").write_text(json.dumps(dict(selected=selected,grid=rows,frozen_val=frozen_val,source_prefers_frozen=bool(frozen_val>=selected["val_f1"]),teacher_scores=teacher_scores,teacher_best=teacher_best),indent=2))
 diag=out/"controller_rows.json"
 if not diag.exists():
  rows=[]
  for sid in controller:
   d=data[sid];q=ema(d["p"],alpha)
   for mode in ["continuous","periodic"]:
    _,ds,_=temporal_stream(m,d["x"],q,d["hf"],lr,mode,collect=True);rows.extend(score_diagnostics(ds,d["y"],sid,mode))
  diag.write_text(json.dumps(rows))
 rows=json.loads(diag.read_text());gates={"future":fit_gate(rows,"future"),"current":fit_gate(rows,"current"),"teacher_confidence":teacher_confidence_gate(rows)};results=[]
 for sid in evaluation:
  d=data[sid];q=ema(d["p"],alpha);dest=out/f"predictions_s{sid:02d}.npz"
  if dest.exists():
   with np.load(dest) as f:preds={k:f[k] for k in f.files if k!="y"}
  else:
   preds=dict(frozen=d["p"],teacher_ema=q,source_selected_ema=ema(d["p"],teacher_best),weak_ema=ema(d["p"],.8))
   for mode in ["continuous","periodic","reset_periodic"]:preds[mode],_,_=temporal_stream(m,d["x"],q,d["hf"],lr,mode)
   preds["source_selected_update"]=preds["frozen"] if frozen_val>=selected["val_f1"] else preds["continuous"]
   for typ,g in gates.items():
    def choose(f):
     values=g.values(f)[0].copy()
     if f[3]==0.:values[1]=0.
     return int(np.r_[0.,values].argmax())
    preds[typ],_,_=temporal_stream(m,d["x"],q,d["hf"],lr,"controller",choose)
   prefix,_,_=temporal_stream(m,d["x"][:31],ema(d["p"][:31],alpha),d["hf"][:31],lr);assert np.array_equal(prefix,preds["continuous"][:31]);np.savez_compressed(dest,y=d["y"],**preds)
  for method,pr in preds.items():results.append(dict(subject=sid,method=method,**metrics(d["y"],pr)))
  print("TEMPORAL BN EVAL",name,sid,flush=True)
 (out/"evaluation.json").write_text(json.dumps(dict(name=name,results=results,selection=selected,prefix_audit=True,development_only=True),indent=2));print("TEMPORAL BN COMPLETE",name,flush=True)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--case",required=True);args=p.parse_args();run(args.case)
