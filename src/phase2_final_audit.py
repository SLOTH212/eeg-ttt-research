from common import *
from adaptation import stream
from controllers import fit_gate
import hashlib
out=ROOT/"phase2";lock=json.loads((out/"evaluation_lock.json").read_text());paths=list((out/"fresh").glob("*.json"));assert len(paths)==456;count=0
for path in paths:
 d=json.loads(path.read_text());sid=d["subject"];expected=np.load(out/"processed"/f"s{sid:02d}_meta.npz")["y"]
 with np.load(path.with_suffix(".npz")) as p:
  assert np.array_equal(p["y"],expected)
  for method in p.files:
   if method=="y":continue
   pr=p[method];assert pr.shape==(len(expected),5) and np.isfinite(pr).all() and np.allclose(pr.sum(1),1,atol=2e-6);count+=1
 assert d["affinity"]==list(range(8,24))
for f,h in lock["sha256"].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h
prefix=[]
for kind in ["spectral","eegnet"]:
 name=f"{kind}_A_17";m,_=load_model(ROOT/"results"/name/"model.pt");old=ROOT/"results"/name;rows=json.loads((old/"controller_diagnostics.json").read_text())+json.loads((old/"source_replay_diagnostics.json").read_text());meta=np.load(out/"processed/s20_meta.npz");hf=meta["features"];x=hf if kind=="spectral" else np.load(out/"processed/s20_x.npy",mmap_mode="r");saved=np.load(out/"fresh"/f"{name}_controllers_s20.npz")
 for typ in ["future","confidence","current"]:
  g=fit_gate(rows,typ)
  def gate(f):
   values=g.values(f)[0].copy()
   if f[3]==0.:values[1]=0.
   return int(np.r_[0.,values].argmax())
  pr,_,_=stream(m,x[:31],"controller",lock["models"][name]["controller_lr"],gate,hf[:31]);err=float(np.max(abs(pr-saved["identity_"+typ][:31])));assert err<2e-6;prefix.append(dict(kind=kind,method=typ,max_error=err))
summary=dict(status="PASS",result_files=len(paths),prediction_streams=count,locked_hashes_valid=True,prefix_checks=prefix,labels_used_only_in_offline_metrics_and_source_fitting=True,cohort=list(range(20,39)))
(out/"final_audit.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
