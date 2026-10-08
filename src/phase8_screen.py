from common import *
from phase8_methods import robust_stream,fisher_diagonal
from phase6_eval import load_carrier,transform
import argparse,itertools
p=argparse.ArgumentParser();p.add_argument("--case",required=True);args=p.parse_args();name=args.case;OUT=ROOT/"phase8";OUT.mkdir(exist_ok=True);out=OUT/name;out.mkdir(exist_ok=True)
if (out/"evaluation.json").exists():raise SystemExit(0)
context=name.startswith("context11")
if context:
 _,source,seed=name.split("_");path=ROOT/"phase4/results"/name/"model.pt";m=load_carrier(path);split=json.loads((ROOT/"phase3/protocol.json").read_text())["folds"][source];kind="spectral";evaluation=[6,8,9,16,18,19,26,28,29,36,38]
else:
 m,ck=load_model(ROOT/"results"/name/"model.pt");kind=ck["kind"];split=CFG["folds"][ck["fold"]];evaluation=split["evaluation"]
def subj(s):
 base=ROOT/("data/processed" if s<20 else "phase2/processed");d=np.load(base/f"s{s:02d}_meta.npz");x=d["features"] if kind=="spectral" else np.load(base/f"s{s:02d}_x.npy",mmap_mode="r")
 if context:x=transform(x,"context11")
 return x,d["y"]
val={s:subj(s) for s in split["base_val"]};fp=out/"fisher.pt"
if not fp.exists():
 fisher,n=fisher_diagonal(m,[subj(s)[0] for s in split["base_train"]]);torch.save(dict(fisher=fisher,n=n),fp)
fisher=torch.load(fp,map_location="cpu",weights_only=False)["fisher"];base_scores={s:metrics(y,predict(m,x))["macro_f1"] for s,(x,y) in val.items()};candidates=[]
for lr,d in itertools.product([1e-4,1e-3,.01],[.05,.2,.5]):
 candidates.append(dict(method="eta",lr=lr,diversity=d))
 for alpha in [50.,2000.]:candidates.append(dict(method="eata",lr=lr,diversity=d,alpha=alpha))
for lr,reset in itertools.product([1e-4,1e-3,.01],[.05,.2]):candidates.append(dict(method="sar",lr=lr,reset_entropy=reset))
rows=[]
for i,opts in enumerate(candidates):
 saved=out/f"candidate_{i:02d}.json"
 if saved.exists():rows.append(json.loads(saved.read_text()));continue
 scores=[];timings={};error=None
 for s,(x,y) in val.items():
  try:
   pr,tm=robust_stream(m,x,fisher=fisher,**opts);scores.append(metrics(y,pr)["macro_f1"]);timings[s]=tm
  except FloatingPointError as e:error=str(e);break
 row=dict(options=opts,val_f1=float(np.mean(scores)) if error is None else None,timings=timings,error=error);saved.write_text(json.dumps(row,indent=2));rows.append(row);print("ROBUST SOURCE",name,i,row["val_f1"],flush=True)
selected={}
for method in ["eta","eata","sar"]:
 available=[v for v in rows if v["options"]["method"]==method and v["val_f1"] is not None];best=max(available,key=lambda v:v["val_f1"]);selected[method]=dict(best=best,source_prefers_frozen=bool(best["val_f1"]<=np.mean(list(base_scores.values()))))
(out/"selection.json").write_text(json.dumps(dict(frozen=base_scores,selected=selected,candidates=rows,scope="Fixed running statistics, BN affine only, batch1, predict-before-update, SGD momentum0.9, no projection; not an exact EATA/SAR reproduction; source-only hyperparameter selection",evaluation=evaluation),indent=2));results=[]
for s in evaluation:
 x,y=subj(s);preds={"frozen":predict(m,x)}
 for method,sel in selected.items():
  pr,tm=robust_stream(m,x,fisher=fisher,**sel["best"]["options"]);preds[method+"_scoped"]=pr;preds[method+"_source_selected"]=preds["frozen"] if sel["source_prefers_frozen"] else pr
  # Streaming causality audit on first31 windows.
  prefix,_=robust_stream(m,x[:31],fisher=fisher,**sel["best"]["options"]);assert np.array_equal(pr[:31],prefix)
 for method,pr in preds.items():results.append(dict(subject=s,method=method,**metrics(y,pr)))
 np.savez_compressed(out/f"predictions_s{s:02d}.npz",y=y,**preds)
(out/"evaluation.json").write_text(json.dumps(dict(results=results,development_only=True),indent=2));print("ROBUST COMPLETE",name,flush=True)
