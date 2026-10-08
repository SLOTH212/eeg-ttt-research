from common import *
from phase6_eval import load_carrier,transform
import copy,itertools
OUT=ROOT/"phase9";OUT.mkdir(exist_ok=True)
cases=[f"{k}_{f}_{z}" for k,f,z in itertools.product(["spectral","eegnet"],["A","B"],[17,29,43])]+[f"context11_{s}_{seed}" for s,seed in itertools.product(["young","mixed"],[17,29,43])]
protocol=dict(purpose="Exploratory equal-source-label-access baseline: static classifier-head fine-tuning versus retention; same frozen encoder",cases=cases,methods=["base_only_head","matched_h4_head","all_controller_head"],matched_rule="Union of future t+1..t+4 labels actually used in controller H4 fitting, across source trajectories; add original base training labels already available to classifier",source_selection="Head AdamW lr0.001,weight_decay0.001,20epochs; include original frozen checkpoint; choose source validation participant MacroF1 only",allowed_cpu_ids=list(range(8,24)),gpu_allowed=False)
(OUT/"protocol.json").write_text(json.dumps(protocol,indent=2))
for name in cases:
 out=OUT/name;out.mkdir(exist_ok=True)
 if (out/"evaluation.json").exists():
  previous=json.loads((out/"evaluation.json").read_text())
  if any(v["method"]=="base_only_head" for v in previous["results"]):continue
  backup=out/"evaluation_before_base_control.json"
  if not backup.exists():backup.write_text(json.dumps(previous,indent=2))
 context=name.startswith("context11")
 if context:
  _,source,seed=name.split("_");m=load_carrier(ROOT/"phase4/results"/name/"model.pt");split=json.loads((ROOT/"phase3/protocol.json").read_text())["folds"][source];controller=[7,17,27,37];evaluation=[6,8,9,16,18,19,26,28,29,36,38];rowpaths=[ROOT/"phase5/results"/name/"controller_rows.json"];kind="spectral"
 else:
  m,ck=load_model(ROOT/"results"/name/"model.pt");seed=ck["seed"];split=CFG["folds"][ck["fold"]];controller=split["controller_train"];evaluation=split["evaluation"];kind=ck["kind"];rowpaths=[ROOT/"results"/name/"controller_diagnostics.json",ROOT/"results"/name/"source_replay_diagnostics.json"]
 labelled={s:set() for s in controller}
 for path in rowpaths:
  for row in json.loads(path.read_text()):
   if row["h"]==4:labelled[row["subject"]].update(range(row["t"]+1,row["t"]+5))
 data={}
 for s in sorted(set(split["base_train"]+split["base_val"]+controller+evaluation)):
  base=ROOT/("data/processed" if s<20 else "phase2/processed");d=np.load(base/f"s{s:02d}_meta.npz");x=d["features"] if kind=="spectral" else np.load(base/f"s{s:02d}_x.npy",mmap_mode="r");y=d["y"]
  if context:x=transform(x,"context11")
  with torch.no_grad():z=torch.cat([m.features(torch.tensor(np.array(x[a:a+128]),dtype=torch.float32)) for a in range(0,len(x),128)])
  data[s]=(z,torch.tensor(y))
 def head_prob(head,z):
  with torch.no_grad():return head(z).softmax(1).numpy()
 initial={s:metrics(data[s][1].numpy(),head_prob(m.head,data[s][0]))["macro_f1"] for s in split["base_val"]};bestheads={};details={}
 for method in protocol["methods"]:
  checkpoint=out/f"{method}.pt"
  if checkpoint.exists():
   saved=torch.load(checkpoint,map_location="cpu",weights_only=False);head=copy.deepcopy(m.head);head.load_state_dict(saved["head"]);bestheads[method]=head;details[method]=saved["selection"];continue
  seed_all(int(seed));head=copy.deepcopy(m.head);opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.001);zs=[];ys=[];counts_by_subject={}
  for s in split["base_train"]+([] if method=="base_only_head" else controller):
   z,y=data[s];mask=y>=0
   if s in controller and method=="matched_h4_head":
    available=torch.zeros(len(y),dtype=torch.bool);available[list(sorted(labelled[s]))]=True;mask=mask&available
   zs.append(z[mask]);ys.append(y[mask]);counts_by_subject[s]=int(mask.sum())
  z=torch.cat(zs);y=torch.cat(ys);counts=torch.bincount(y,minlength=5).float();weights=(counts.sum()/counts.clamp(min=1)).sqrt();weights/=weights.mean();best=float(np.mean(list(initial.values())));beststate=copy.deepcopy(head.state_dict());bestepoch=0;history=[]
  for epoch in range(20):
   for idx in torch.randperm(len(y)).split(256):
    opt.zero_grad();loss=nn.functional.cross_entropy(head(z[idx]),y[idx],weight=weights);loss.backward();opt.step()
   scores={s:metrics(data[s][1].numpy(),head_prob(head,data[s][0]))["macro_f1"] for s in split["base_val"]};value=float(np.mean(list(scores.values())));history.append(dict(epoch=epoch+1,val_f1=value))
   if value>best:best=value;beststate=copy.deepcopy(head.state_dict());bestepoch=epoch+1
  head.load_state_dict(beststate);bestheads[method]=head;details[method]=dict(best_epoch=bestepoch,val_f1=best,history=history,label_counts=counts_by_subject);torch.save(dict(head=beststate,method=method,source_model=name,selection=details[method]),out/f"{method}.pt");print("STATIC HEAD",name,method,bestepoch,best,flush=True)
 results=[]
 for s in evaluation:
  z,y=data[s];preds={"frozen":head_prob(m.head,z)}
  for method,head in bestheads.items():preds[method]=head_prob(head,z)
  for method,pr in preds.items():results.append(dict(subject=s,method=method,**metrics(y.numpy(),pr)))
  np.savez_compressed(out/f"predictions_s{s:02d}.npz",y=y.numpy(),**preds)
 (out/"evaluation.json").write_text(json.dumps(dict(name=name,results=results,selection=details,frozen_val=initial,development_only=True),indent=2));print("LABEL ACCESS COMPLETE",name,flush=True)
