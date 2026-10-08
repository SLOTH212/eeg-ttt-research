from common import *
from adaptation import stream
import argparse,time,hashlib
p=argparse.ArgumentParser();p.add_argument("--name",required=True);args=p.parse_args();name=args.name;out=ROOT/"phase2/fresh_simple";out.mkdir(exist_ok=True);old=ROOT/"results"/name;m,ck=load_model(old/"model.pt");kind=ck["kind"];selection=json.loads((old/"lr_sensitivity.json").read_text());lr=json.loads((old/"selection.json").read_text())["lr"]
(out/f"{name}_protocol.json").write_text(json.dumps(dict(posthoc_extension=True,note="Added after locked phase2 was inspected; all learning rates and reset/periodic rules already fixed in phase1 using source labels only; not part of phase2 preregistered primary comparison",original_lr=lr,extended_lr=selection["selected_lr"],source_prefers_frozen=selection["source_prefers_frozen"],selection_sha256=hashlib.sha256((old/"lr_sensitivity.json").read_bytes()).hexdigest()),indent=2))
for sid in range(20,39):
 dest=out/f"{name}_s{sid:02d}.json"
 if dest.exists():continue
 meta=np.load(ROOT/"phase2/processed"/f"s{sid:02d}_meta.npz");y=meta["y"];hf=meta["features"];x=hf if kind=="spectral" else np.load(ROOT/"phase2/processed"/f"s{sid:02d}_x.npy",mmap_mode="r");preds={};timings={}
 for method,mode,rate in [("continuous","continuous",lr),("extended_continuous","continuous",selection["selected_lr"]),("periodic","periodic",lr),("reset_periodic","reset_periodic",lr)]:
  if method=="extended_continuous" and rate==lr:preds[method]=preds["continuous"];continue
  preds[method],_,timings[method]=stream(m,x,mode,rate,history_features=hf)
 if selection["source_prefers_frozen"]:preds["source_selected_simple"]=np.load(ROOT/"phase2/fresh"/f"{name}_baselines_s{sid:02d}.npz")["frozen"]
 else:preds["source_selected_simple"]=preds["extended_continuous"]
 mask=np.zeros(len(y),bool)
 for b in np.flatnonzero((y[1:]!=y[:-1])&(y[1:]>=0)&(y[:-1]>=0))+1:mask[max(0,b-2):min(len(y),b+3)]=True
 results=[dict(subject=sid,method=method,**metrics(y,pr),transition_f1=metrics(y[mask],pr[mask])["macro_f1"]) for method,pr in preds.items()]
 np.savez_compressed(dest.with_suffix(".npz"),y=y,**preds);dest.write_text(json.dumps(dict(name=name,subject=sid,results=results,timings=timings,posthoc_extension=True),indent=2));print("SIMPLE",name,sid,flush=True)
