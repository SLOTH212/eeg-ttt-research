from common import *
from phase6_eval import load_carrier,transform
from adaptation import Adapter,stream
import hashlib,pickle
OUT=ROOT/"phase6";lock=json.loads((OUT/"evaluation_lock.json").read_text());files=list((OUT/"fresh_baselines").glob("*.json"))+list((OUT/"fresh_controllers").glob("*.json"));assert len(files)==969;labels={s:np.load(OUT/"processed"/f"s{s:02d}_meta.npz")["y"] for s in lock["cohort"]};streams=0;max_metric_error=0.
for f in files:
 d=json.loads(f.read_text());sid=d["subject"];expected=labels[sid]
 with np.load(f.with_suffix(".npz")) as p:
  assert np.array_equal(p["y"],expected)
  for row in d["results"]:
   pr=p[row["method"]];assert pr.shape==(len(expected),5) and np.isfinite(pr).all() and np.allclose(pr.sum(1),1,atol=2e-6);k=expected>=0;cm=np.bincount(expected[k]*5+pr[k].argmax(1),minlength=25).reshape(5,5);f1=float(np.mean(2*cm.diagonal()/np.maximum(cm.sum(0)+cm.sum(1),1)));err=abs(f1-row["macro_f1"]);assert err<1e-12;max_metric_error=max(max_metric_error,err);streams+=1
 assert d["affinity"]==list(range(8,24))
for f,h in lock["sha256"].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h,f
for v in lock["models"].values():assert hashlib.sha256((ROOT/v["path"]).read_bytes()).hexdigest()==v["sha256"]
# Rule out tiny apparent gains caused solely by batch256-vs-batch1 floating-point argmax differences.
checks=[];destination=OUT/"streaming_frozen";destination.mkdir(exist_ok=True)
names=list(lock["controllers"])+[f"wide_young_{seed}" for seed in [17,29,43]]
for name in names:
 spec=lock["models"][name];m=load_carrier(ROOT/spec["path"]);a=Adapter(m)
 for sid in lock["cohort"]:
  path=destination/f"{name}_s{sid:02d}.json"
  if path.exists():checks.append(json.loads(path.read_text()));continue
  meta=np.load(OUT/"processed"/f"s{sid:02d}_meta.npz");x=transform(meta["features"],spec["variant"]);y=meta["y"];pr=[]
  for t in range(len(x)):
   pr.append(a.model(torch.tensor(np.array(x[t:t+1]),dtype=torch.float32)).softmax(1).detach().numpy()[0])
  pr=np.array(pr);batch=np.load(OUT/"fresh_baselines"/f"{name}_s{sid:02d}.npz")["frozen"];row=dict(name=name,subject=sid,max_probability_error=float(np.max(abs(pr-batch))),argmax_disagreements=int(np.sum(pr.argmax(1)!=batch.argmax(1))),streaming_f1=metrics(y,pr)["macro_f1"]);np.savez_compressed(path.with_suffix(".npz"),p=pr);path.write_text(json.dumps(row));checks.append(row)
 print("STREAM NUMERICS",name,flush=True)
prefix=[]
for name in ["context11_young_17","context11_mixed_43"]:
 spec=lock["models"][name];m=load_carrier(ROOT/spec["path"]);f=OUT/"controllers"/(name+".pkl");assert hashlib.sha256(f.read_bytes()).hexdigest()==lock["controllers"][name]["sha256"]
 with open(f,"rb") as fh:info=pickle.load(fh)
 d=np.load(OUT/"processed/s40_meta.npz");hf=d["features"];x=transform(hf,"context11");saved=np.load(OUT/"fresh_controllers"/f"{name}_s40.npz")
 for method,g in info["gates"].items():
  def choose(f):
   p=g.values(f)[0].copy()
   if f[3]==0.:p[1]=0.
   return int(np.r_[0.,p].argmax())
  pp,_,_=stream(m,x[:31],"controller",info["lr"],choose,hf[:31]);err=float(np.max(abs(pp-saved[method][:31])));assert err<2e-6;prefix.append(dict(name=name,method=method,max_error=err))
report=dict(status="PASS",result_files=len(files),prediction_streams=streams,max_metric_error=max_metric_error,hashes_valid=True,prefix_checks=prefix,streaming_frozen_checks=len(checks),total_batch_argmax_disagreements=sum(v["argmax_disagreements"] for v in checks),max_batch_probability_error=max(v["max_probability_error"] for v in checks));(OUT/"final_audit.json").write_text(json.dumps(report,indent=2));print(json.dumps(report))
