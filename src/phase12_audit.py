from phase12_temporal_bn import *
checks=[]
for folder in sorted((ROOT/"phase12").iterdir()):
 if not folder.is_dir() or not (folder/"evaluation.json").exists():continue
 name=folder.name;context=name.startswith("context11")
 if context:model=load_carrier(ROOT/"phase4/results"/name/"model.pt");kind="spectral"
 else:model,ck=load_model(ROOT/"results"/name/"model.pt");kind=ck["kind"]
 original={k:v.clone() for k,v in model.state_dict().items()};sel=json.loads((folder/"selection.json").read_text())["selected"];rows=json.loads((folder/"controller_rows.json").read_text());gates={"future":fit_gate(rows,"future"),"current":fit_gate(rows,"current"),"teacher_confidence":teacher_confidence_gate(rows)};files=sorted(folder.glob("predictions_s*.npz"));evaluations=json.loads((folder/"evaluation.json").read_text())["results"];metric_error=0.
 for f in files:
  sid=int(f.stem.rsplit("s",1)[1]);saved=np.load(f)
  for method in saved.files:
   if method=="y":continue
   expected=next(v["macro_f1"] for v in evaluations if v["subject"]==sid and v["method"]==method);metric_error=max(metric_error,abs(metrics(saved["y"],saved[method])["macro_f1"]-expected))
 assert metric_error==0.
 for f in [files[0],files[-1]]:
  sid=int(f.stem.rsplit("s",1)[1]);data=ROOT/("data/processed" if sid<20 else "phase2/processed");hf=np.load(data/f"s{sid:02d}_meta.npz")["features"][:31];x=hf if kind=="spectral" else np.load(data/f"s{sid:02d}_x.npy",mmap_mode="r")[:31]
  if context:x=transform(x,"context11")
  saved=np.load(f);teacher=ema(saved["frozen"][:31],sel["alpha"])
  for method,gate in gates.items():
   def choose(features):
    values=gate.values(features)[0].copy()
    if features[3]==0.:values[1]=0.
    return int(np.r_[0.,values].argmax())
   pred,_,_=temporal_stream(model,x,teacher,hf,sel["lr"],"controller",choose);err=float(np.max(np.abs(pred-saved[method][:31])));assert err==0.
   checks.append(dict(case=name,subject=sid,method=method,prefix_max_error=err,metric_recompute_error=metric_error))
  assert all(torch.equal(v,model.state_dict()[k]) for k,v in original.items())
 print("TEMPORAL AUDIT",name,flush=True)
assert len(checks)==108
(ROOT/"phase12/final_audit.json").write_text(json.dumps(dict(status="PASS",prefix_runs=len(checks),source_models_unchanged=True,checks=checks),indent=2));print("PHASE12 AUDIT PASS")
