from common import *
from adaptation import Adapter
from phase6_eval import load_carrier,transform
import argparse,copy
p=argparse.ArgumentParser();p.add_argument("--case",required=True);args=p.parse_args();name=args.case;OUT=ROOT/"phase13";OUT.mkdir(exist_ok=True);out=OUT/name;out.mkdir(exist_ok=True)
if (out/"evaluation.json").exists():raise SystemExit(0)
context=name.startswith("context11")
if context:
 _,source,seed=name.split("_");m=load_carrier(ROOT/"phase4/results"/name/"model.pt");split=json.loads((ROOT/"phase3/protocol.json").read_text())["folds"][source];controller=[7,17,27,37];evaluation=[6,8,9,16,18,19,26,28,29,36,38];kind="spectral";rowpaths=[ROOT/"phase5/results"/name/"controller_rows.json"]
else:
 m,ck=load_model(ROOT/"results"/name/"model.pt");kind=ck["kind"];seed=ck["seed"];split=CFG["folds"][ck["fold"]];controller=split["controller_train"];evaluation=split["evaluation"];rowpaths=[ROOT/"results"/name/"controller_diagnostics.json",ROOT/"results"/name/"source_replay_diagnostics.json"]
labelled={s:set() for s in controller}
for f in rowpaths:
 for v in json.loads(f.read_text()):
  if v["h"]==4:labelled[v["subject"]].update(range(v["t"]+1,v["t"]+5))
def subj(s):
 root=ROOT/("data/processed" if s<20 else "phase2/processed");d=np.load(root/f"s{s:02d}_meta.npz");x=d["features"] if kind=="spectral" else np.load(root/f"s{s:02d}_x.npy",mmap_mode="r")
 if context:x=transform(x,"context11")
 return x,d["y"]
xs=[];ys=[];counts_by_subject={}
for s in controller:
 x,y=subj(s);idx=np.array(sorted(labelled[s]));idx=idx[y[idx]>=0];xs.append(np.array(x[idx]));ys.append(y[idx]);counts_by_subject[s]=len(idx)
x=torch.tensor(np.concatenate(xs));y=torch.tensor(np.concatenate(ys));del xs,ys;val={s:subj(s) for s in split["base_val"]};frozen_val=float(np.mean([metrics(vy,predict(m,vx))["macro_f1"] for vx,vy in val.values()]));counts=torch.bincount(y,minlength=5).float();weights=(counts.sum()/counts.clamp(min=1)).sqrt();weights/=weights.mean();models={};selections={}
for scope in ["affine","head"]:
 path=out/f"{scope}.pt"
 if path.exists():
  saved=torch.load(path,map_location="cpu",weights_only=False);fitted=copy.deepcopy(m);fitted.load_state_dict(saved["state"]);models[scope]=fitted;selections[scope]=saved["selection"];continue
 seed_all(int(seed));source_model=copy.deepcopy(m);source_model.writable_scope=scope;a=Adapter(source_model,radius=.1);opt=torch.optim.AdamW(a.params,lr=.001,weight_decay=.001);best=frozen_val;bestepoch=0;beststate=copy.deepcopy(a.model.state_dict());history=[]
 for epoch in range(20):
  for idx in torch.randperm(len(y)).split(64 if kind=="eegnet" else 128):
   opt.zero_grad();loss=nn.functional.cross_entropy(a.model(x[idx]),y[idx],weight=weights);loss.backward();opt.step()
   with torch.no_grad():
    for v,s,scale in zip(a.params,a.source,a.scales):
     d=v-s;limit=.1*scale*np.sqrt(v.numel());norm=float(d.norm())
     if norm>limit:v.copy_(s+d*(limit/norm))
  score=float(np.mean([metrics(vy,predict(a.model,vx))["macro_f1"] for vx,vy in val.values()]));history.append(dict(epoch=epoch+1,val_f1=score))
  if score>best:best=score;bestepoch=epoch+1;beststate=copy.deepcopy(a.model.state_dict())
 a.model.load_state_dict(beststate);models[scope]=a.model;selections[scope]=dict(best_epoch=bestepoch,best_val_f1=best,frozen_val=frozen_val,history=history,writable_parameters=sum(v.numel() for v in a.params),label_counts=counts_by_subject,relative_radius=.1,final_drift=a.drift());torch.save(dict(state=beststate,selection=selections[scope]),path);print("STATIC SAME SCOPE",name,scope,bestepoch,best,flush=True)
results=[]
for s in evaluation:
 vx,vy=subj(s);preds={"frozen":predict(m,vx)}
 for scope,model in models.items():preds["matched_label_static_"+scope]=predict(model,vx)
 for method,pr in preds.items():results.append(dict(subject=s,method=method,**metrics(vy,pr)))
 np.savez_compressed(out/f"predictions_s{s:02d}.npz",y=vy,**preds)
(out/"evaluation.json").write_text(json.dumps(dict(name=name,results=results,selections=selections,development_only=True),indent=2));print("STATIC SCOPE COMPLETE",name,flush=True)
