from common import *
from adaptation import Adapter
from phase6_eval import load_carrier,transform
import argparse
p=argparse.ArgumentParser();p.add_argument("--case",required=True);args=p.parse_args();name=args.case;OUT=ROOT/"phase11";OUT.mkdir(exist_ok=True);context=name.startswith("context11")
if context:
 m=load_carrier(ROOT/"phase4/results"/name/"model.pt");kind="spectral";source=[7,17,27,37];middle=[s for s in range(20,39) if s not in [25,27,35,37]];lr=.001
else:
 m,ck=load_model(ROOT/"results"/name/"model.pt");kind=ck["kind"];source=CFG["folds"][ck["fold"]]["controller_train"];middle=list(range(20,39));lr=.0001 if kind=="eegnet" else .001
rows=[]
for group,subjects in [("controller_source",source),("middle",middle),("older",list(range(40,59)))]:
 for sid in subjects:
  saved=OUT/f"{name}_{group}_s{sid:02d}.json"
  if saved.exists():rows.extend(json.loads(saved.read_text()));continue
  root=ROOT/("phase6/processed" if sid>=40 else ("phase2/processed" if sid>=20 else "data/processed"));d=np.load(root/f"s{sid:02d}_meta.npz");y=d["y"];x=d["features"] if kind=="spectral" else np.load(root/f"s{sid:02d}_x.npy",mmap_mode="r")
  if context:x=transform(x,"context11")
  a=Adapter(m,lr);subjectrows=[]
  for t in range(0,len(x)-4,20):
   yy=y[t+1:t+5];mask=yy>=0
   if not mask.any() or y[t]<0:continue
   a.restore(a.source);prob,candidate,signals=a.candidate(torch.tensor(np.array(x[t:t+1]),dtype=torch.float32));gent=[v.grad.detach().clone() for v in a.params];a.model.zero_grad(set_to_none=True);future=torch.tensor(np.array(x[t+1:t+5][mask]),dtype=torch.float32);labels=torch.tensor(yy[mask]);loss=nn.functional.cross_entropy(a.model(future),labels);loss.backward();gtask=[v.grad.detach().clone() for v in a.params];dot=float(sum((g*h).sum() for g,h in zip(gent,gtask)));gn=float(sum(g.square().sum() for g in gent).sqrt());tn=float(sum(g.square().sum() for g in gtask).sqrt());keep=float(loss.detach());a.restore(candidate)
   with torch.no_grad():write=float(nn.functional.cross_entropy(a.model(future),labels))
   subjectrows.append(dict(case=name,cohort=group,subject=sid,t=t,stage=int(y[t]),prediction=int(prob.argmax()),confidence=float(prob.max()),current_correct=bool(prob.argmax()==y[t]),future_stable=bool(np.all(yy[mask]==y[t])),gradient_dot=dot,gradient_cosine=dot/max(gn*tn,1e-20),entropy_gradient_norm=gn,task_gradient_norm=tn,first_order_gain=lr*dot,actual_gain=keep-write,keep_ce=keep,lr=lr))
  saved.write_text(json.dumps(subjectrows));rows.extend(subjectrows);print("ALIGNMENT",name,group,sid,flush=True)
(OUT/f"{name}_complete.json").write_text(json.dumps(dict(rows=len(rows),source_state_only=True,horizon=4,stride=20,loss="Unclipped log-softmax CE for derivative analysis; target labels used offline only",affinity=sorted(os.sched_getaffinity(0))),indent=2))
