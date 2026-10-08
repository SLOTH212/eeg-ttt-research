from common import *
from phase6_eval import load_carrier,transform
import copy
OUT=ROOT/"phase9/fresh";OUT.mkdir(exist_ok=True);protocol=json.loads((ROOT/"phase9/protocol.json").read_text());cases=protocol["cases"]
(OUT/"analysis_scope.json").write_text(json.dumps(dict(posthoc_extension=True,note="Heads already selected using source validation; apply to previously observed age50-60 and66-74 cohorts. These are not new sealed primary tests. Exclude all source training/validation/controller people for each model.",methods=protocol["methods"],gpu_allowed=False,allowed_cpu_ids=list(range(8,24))),indent=2))
for name in cases:
 context=name.startswith("context11")
 if context:
  _,source,seed=name.split("_");m=load_carrier(ROOT/"phase4/results"/name/"model.pt");split=json.loads((ROOT/"phase3/protocol.json").read_text())["folds"][source];excluded=set(split["base_train"]+split["base_val"]+[7,17,27,37]);kind="spectral"
 else:m,ck=load_model(ROOT/"results"/name/"model.pt");kind=ck["kind"];sp=CFG["folds"][ck["fold"]];excluded=set(sp["base_train"]+sp["base_val"]+sp["controller_train"])
 heads={};drifts={}
 for method in protocol["methods"]:
  ck=torch.load(ROOT/"phase9"/name/f"{method}.pt",map_location="cpu",weights_only=False);head=copy.deepcopy(m.head);head.load_state_dict(ck["head"]);heads[method]=head;drifts[method]=dict(weight_norm=float((head.weight-m.head.weight).norm().detach()),bias_norm=float((head.bias-m.head.bias).norm().detach()))
 for sid in list(range(20,39))+list(range(40,59)):
  if sid in excluded:continue
  dest=OUT/f"{name}_s{sid:02d}.json"
  if dest.exists():continue
  root=ROOT/("phase2/processed" if sid<40 else "phase6/processed");meta=np.load(root/f"s{sid:02d}_meta.npz");y=meta["y"];cache=ROOT/"phase2/cache"/f"{name}_s{sid:02d}.npz"
  if not context and cache.exists():
   with np.load(cache) as d:z=torch.tensor(d["z"]);base=d["p"]
  else:
   x=meta["features"] if kind=="spectral" else np.load(root/f"s{sid:02d}_x.npy",mmap_mode="r")
   if context:x=transform(x,"context11")
   with torch.no_grad():z=torch.cat([m.features(torch.tensor(np.array(x[a:a+128]),dtype=torch.float32)) for a in range(0,len(x),128)]);base=m.head(z).softmax(1).numpy()
  preds=dict(frozen=base)
  with torch.no_grad():
   for method,head in heads.items():preds[method]=head(z).softmax(1).numpy()
  rows=[dict(subject=sid,method=method,**metrics(y,p)) for method,p in preds.items()];np.savez_compressed(dest.with_suffix(".npz"),y=y,**preds);dest.write_text(json.dumps(dict(case=name,cohort="middle" if sid<40 else "older",results=rows,head_drift=drifts,posthoc_extension=True),indent=2))
 print("STATIC HEAD TRANSFER",name,flush=True)
