from common import *
from adaptation import stream
from controllers import fit_gate
from phase6_eval import load_carrier,transform
import argparse
p=argparse.ArgumentParser();p.add_argument("--case",required=True);args=p.parse_args();name=args.case;OUT=ROOT/"phase10";OUT.mkdir(exist_ok=True);out=OUT/name;out.mkdir(exist_ok=True)
if (out/"source_selection.json").exists():raise SystemExit(0)
context=name.startswith("context11")
if context:
 _,source,seed=name.split("_");m=load_carrier(ROOT/"phase4/results"/name/"model.pt");old=ROOT/"phase5/results"/name;paths=[old/"controller_rows.json"];lr=json.loads((old/"selection.json").read_text())["best_positive"]["lr"];validation=[5,15,25,35];kind="spectral"
else:
 old=ROOT/"results"/name;m,ck=load_model(old/"model.pt");paths=[old/"controller_diagnostics.json",old/"source_replay_diagnostics.json"];lr=json.loads((old/"selection.json").read_text())["lr"];validation=CFG["folds"][ck["fold"]]["base_val"];kind=ck["kind"]
raw=[]
for f in paths:raw+=json.loads(f.read_text())
val={}
for s in validation:
 base=ROOT/("data/processed" if s<20 else "phase2/processed");d=np.load(base/f"s{s:02d}_meta.npz");hf=d["features"];x=hf if kind=="spectral" else np.load(base/f"s{s:02d}_x.npy",mmap_mode="r")
 if context:x=transform(x,"context11")
 val[s]=(x,d["y"],hf)
results=[];loso=[];frozen={s:metrics(y,predict(m,x))["macro_f1"] for s,(x,y,hf) in val.items()}
for horizon in [1,4,10]:
 rows=[dict(v,h=4) for v in raw if v["h"]==horizon];gate=fit_gate(rows,"future")
 for s in sorted({v["subject"] for v in rows}):
  train=[v for v in rows if v["subject"]!=s];test=[v for v in rows if v["subject"]==s];g=fit_gate(train);f=np.array([v["features"] for v in test]);p=g.values(f);p[f[:,3]==0,1]=0;act=np.c_[np.zeros(len(p)),p].argmax(1);gain=np.array([[0.,v["write_gain"],v["reset_gain"]] for v in test]);loso.append(dict(horizon=horizon,subject=s,realized_gain=float(gain[np.arange(len(act)),act].mean()),oracle=float(gain.max(1).mean())))
 for s,(x,y,hf) in val.items():
  path=out/f"h{horizon}_s{s:02d}.json"
  if path.exists():results.append(json.loads(path.read_text()));continue
  def choose(f):
   p=gate.values(f)[0].copy()
   if f[3]==0.:p[1]=0.
   return int(np.r_[0.,p].argmax())
  pr,_,tm=stream(m,x,"controller",lr,choose,hf);row=dict(horizon=horizon,subject=s,**metrics(y,pr),timing=tm);path.write_text(json.dumps(row,indent=2));results.append(row);print("HORIZON",name,horizon,s,row["macro_f1"],flush=True)
means={h:float(np.mean([v["macro_f1"] for v in results if v["horizon"]==h])) for h in [1,4,10]};best=max(means,key=means.get);(out/"source_selection.json").write_text(json.dumps(dict(case=name,source_frozen=frozen,source_scores=means,best_horizon=best,source_prefers_frozen=bool(np.mean(list(frozen.values()))>=means[best]),lr=lr,results=results,source_loso=loso,source_only=True),indent=2));print("HORIZON COMPLETE",name,means,flush=True)
