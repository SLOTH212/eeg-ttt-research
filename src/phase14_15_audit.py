from common import *
import hashlib
checks=[]
for f in sorted((ROOT/"phase14/features").glob("s[0-9][0-9].json")):
 d=json.loads(f.read_text());assert d["cpu_ids"]==list(range(8,24));assert d["no_labels_read"];assert all(v["other_sample_perturbation_exact"] for v in d["numerical_audit"].values());features=np.load(f.with_suffix(".npz"));assert all(features[k].shape==tuple(d["features_shape"]) and np.isfinite(features[k]).all() for k in ["pretrained","random"]);checks.append(dict(subject=d["subject"],windows=d["windows"],sha256=hashlib.sha256(f.with_suffix(".npz").read_bytes()).hexdigest()))
assert len(checks)==58 and sum(v["windows"] for v in checks)==158260
for phase,expected_cases in [(14,10),(15,12)]:
 folder=ROOT/f"phase{phase}";root=folder/"probes" if phase==14 else folder;cases=list(root.glob("*/evaluation.json"));assert len(cases)==expected_cases;streams=0;maxerr=0.;prefix=0
 for f in cases:
  d=json.loads(f.read_text());subjects=set(v["subject"] for v in d["results"]);assert len(subjects)==(34 if phase==14 else 30)
  if phase==15:
   assert not subjects.intersection([7,17,27,37]);assert d["reference_stream_match"];assert len(d["audits"])==90 and all(v["prefix_exact"] for v in d["audits"]);prefix+=len(d["audits"])
  for sid in subjects:
   path=f.parent/(f"s{sid:02d}.npz" if phase==14 else f"predictions_s{sid:02d}.npz");p=np.load(path);y=p["y"]
   for method in p.files:
    if method=="y":continue
    pr=p[method];assert len(pr)==len(y) and np.isfinite(pr).all() and np.allclose(pr.sum(1),1,atol=1e-6);expected=next(v["macro_f1"] for v in d["results"] if v["subject"]==sid and v["method"]==method);err=abs(metrics(y,pr)["macro_f1"]-expected);maxerr=max(maxerr,err);assert err==0.;streams+=1
  print("AUDITED",phase,f.parent.name,flush=True)
 audit=dict(status="PASS",cases=len(cases),prediction_streams=streams,max_metric_error=maxerr,recorded_exact_prefix_checks=prefix)
 if phase==14:audit["feature_checks"]=checks
 (folder/"final_audit.json").write_text(json.dumps(audit,indent=2));print("AUDIT COMPLETE",phase,streams,flush=True)
