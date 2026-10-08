from common import *
from phase6_eval import load_carrier
from adaptation import Adapter
import hashlib
rows=[]
for folder in sorted((ROOT/"phase13").iterdir()):
 if not folder.is_dir() or not (folder/"evaluation.json").exists():continue
 name=folder.name
 if name.startswith("context11"):base=load_carrier(ROOT/"phase4/results"/name/"model.pt")
 else:base,_=load_model(ROOT/"results"/name/"model.pt")
 for scope in ["affine","head"]:
  base.writable_scope=scope;a=Adapter(base,radius=.1);writable={k for k,v in a.model.named_parameters() if v.requires_grad};source=a.model.state_dict();saved=torch.load(folder/f"{scope}.pt",map_location="cpu",weights_only=False);state=saved["state"];changed=[k for k in source if not torch.equal(source[k],state[k])];assert set(changed)<=writable
  ratios=[]
  for k in writable:
   scale=max(float(source[k].square().mean().sqrt()),1.);limit=.1*scale*np.sqrt(source[k].numel());ratio=float((state[k]-source[k]).norm())/limit;ratios.append(ratio);assert ratio<=1.00002
  preds=list(folder.glob("predictions_s*.npz"));err=0.
  evals=json.loads((folder/"evaluation.json").read_text())["results"]
  for f in preds:
   sid=int(f.stem.rsplit("s",1)[1]);d=np.load(f);method="matched_label_static_"+scope;expected=next(v["macro_f1"] for v in evals if v["subject"]==sid and v["method"]==method);err=max(err,abs(expected-metrics(d["y"],d[method])["macro_f1"]))
  assert err==0.
  rows.append(dict(case=name,scope=scope,changed_tensors=len(changed),max_radius_fraction=max(ratios),frozen_parameters_and_running_stats_exact=True,metric_recompute_error=err,checkpoint_sha256=hashlib.sha256((folder/f"{scope}.pt").read_bytes()).hexdigest()))
assert len(rows)==36
(ROOT/"phase13/final_audit.json").write_text(json.dumps(dict(status="PASS",models=len(rows),checks=rows),indent=2));print("PHASE13 AUDIT PASS",len(rows))
