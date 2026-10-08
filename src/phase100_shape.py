from phase79_amplitude import *
from phase99_covariance import transform
import time,datetime
O=ROOT/"phase100";O.mkdir(exist_ok=True);(O/"predictions").mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
cases=json.loads((ROOT/"phase85/protocol.json").read_text())["cases"]
specs=[s for s in json.loads((ROOT/"phase79/protocol.json").read_text())["models"] if s["model"]!="relative"]
pr=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=cases,models=specs,modes=["diagonal","full"],baseline="Existing scalar guarded_median predictions",primary="CBraMod three seed metrics averaged within person; full minus scalar guard, EMA .5, native SleepEDF and left ISRUC; cohorts separate",secondary="Diagonal ablation, LaBraM and context11, right montage, EMA1/.8",selection="No tuning or selection on these previously observed targets; exploratory characterization, not new blind validation.",reference_sha256=sha(ROOT/"phase99/source_reference.npz"),shape_protocol_sha256=sha(ROOT/"phase99/protocol.json"),scalar_protocol_sha256=sha(ROOT/"phase79/protocol.json"),cpu_ids=list(range(8,24)),gpu=True)
if (O/"protocol.json").exists():
 old=json.loads((O/"protocol.json").read_text());pr["created_utc"]=old["created_utc"];assert old==pr
else:
 (O/"protocol.json").write_text(json.dumps(pr,indent=2));(O/"protocol.sha256").write_text(sha(O/"protocol.json"))
models={};snaps={}
for s in specs:
 path=ROOT/s["path"];assert sha(path)==s["sha256"];ck=torch.load(path,map_location="cpu",weights_only=False);kind=s["model"]
 m=Carrier("pretrained") if kind=="cbramod" else FullCarrier("mixed","pretrained") if kind=="labram" else SpectralNet(ck["dim"])
 m.load_state_dict(ck["state"]);m.cuda().eval().requires_grad_(False);key=f'{kind}_seed{s["seed"]}';models[key]=m;snaps[key]=snapshot(m)
reference=np.load(ROOT/"phase99/source_reference.npz")["reference"];records=[]
def legacy(c):
 d,s,m=c["domain"],c["subject"],c["montage"]
 if d in ["source","healthy"]:return ROOT/"phase79/predictions"/f'{"source" if d=="source" else "external"}_{s}_{m}_guarded_median.npz'
 if d=="sleepedf":return ROOT/"phase81/predictions"/f"{s}_guarded_median.npz"
 return ROOT/"phase83/predictions"/f"{s}_{m}_guarded_median.npz"
for ci,c in enumerate(cases):
 name="_".join(c[k] for k in ["domain","subject","montage"]);ap=O/"predictions"/(name+"_audit.json")
 if ap.exists():records.append(json.loads(ap.read_text()));continue
 start=time.perf_counter();raw=np.load(c["x"],mmap_mode="r");lp=legacy(c);gp=lp.with_name(lp.stem+"_normalizer.npz");gains=np.load(gp)["gains"];base=np.load(lp)
 scalar=np.clip(raw*gains[:,None,None],-20,20).astype(np.float32)
 rec=dict(case=c,baseline_path=str(lp),baseline_sha256=sha(lp),gains_sha256=sha(gp),checks=[])
 for mode in pr["modes"]:
  x,aux,stats=transform(raw,gains,reference,mode);changed=np.any(x!=scalar,axis=(1,2));mask=np.convolve(changed.astype(int),np.ones(11,dtype=int))[:len(raw)]>0
  prefix=transform(raw[:65],gains[:65],reference,mode)[0];future=raw[:66].copy();future[65]+=3;assert np.array_equal(prefix,x[:65]) and np.array_equal(transform(future,gains[:66],reference,mode)[0][:65],x[:65])
  assert stats["max_relative_energy_error_beforeclip"]<1e-12 and np.array_equal(x[gains==1],raw[gains==1])
  probs={key:base[key].copy() for key in models};checks=[]
  if changed.any():
   ff=features(x);ct=context(ff);assert np.array_equal(ct[:65],context(ff[:65]))
   inputs={"context11":torch.from_numpy(ct[mask]),"cbramod":torch.from_numpy(prep(x[changed],[.1,45])),"labram":preprocess(x[changed]).reshape(-1,2,2,15,200)}
   for key,m in models.items():
    kind=key.rsplit("_seed",1)[0];ix=mask if kind=="context11" else changed;xx=inputs[kind];p=probabilities(m,xx);probs[key][ix]=p
    with torch.no_grad():one=m(xx[:1].cuda()).softmax(1).cpu().numpy()
    err=float(abs(one-p[:1]).max());assert err<2e-5;assert np.array_equal(probs[key][~ix],base[key][~ix])
    assert all(torch.equal(v.cpu(),snaps[key][k]) for k,v in m.state_dict().items())
    checks.append(dict(model=key,recomputed=int(ix.sum()),batch_one_error=err,state_exact=True,unaffected_predictions_exact=True))
  for key,p in probs.items():assert p.shape==(len(raw),5) and np.isfinite(p).all() and abs(p.sum(1)-1).max()<3e-6
  np.savez_compressed(O/"predictions"/f"{name}_{mode}.npz",**probs)
  np.savez_compressed(O/"predictions"/f"{name}_{mode}_normalizer.npz",**aux)
  rec["checks"].append(dict(mode=mode,changed_from_scalar=int(changed.sum()),context_affected=int(mask.sum()),prefix_and_future_perturbation_exact=True,models=checks,**stats))
 rec["seconds"]=time.perf_counter()-start;ap.write_text(json.dumps(rec,indent=2));records.append(rec);print("CASE",ci+1,len(cases),name,round(rec["seconds"],2),flush=True)
(O/"audit.json").write_text(json.dumps(dict(status="PASS",records=records,models_frozen=True,source_reference_fixed=True,classification_labels_not_read=True),indent=2))
(O/"complete.json").write_text(json.dumps(dict(cases=len(cases),models=len(models),modes=pr["modes"])))
print("COMPLETE",flush=True)
