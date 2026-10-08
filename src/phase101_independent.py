from phase79_amplitude import Carrier,FullCarrier,SpectralNet,ROOT,prep,preprocess,probabilities,snapshot,context,normalize
from phase99_covariance import transform as shape_transform
from phase83_prepare import checked_download,resolve,sha
from phase76_prepare import header,direct_samples,transform,features
from pathlib import Path
import numpy as np,torch,json,datetime,shutil,pyedflib,time
O=ROOT/"phase101"
for d in ["metadata","scratch","predictions","inputs"]: (O/d).mkdir(parents=True,exist_ok=True)
ids=[f"sub-I{i:03d}" for i in range(41,61)];old=json.loads((ROOT/"phase83/protocol.json").read_text())
manifest=[v for v in json.loads((ROOT/"phase76/metadata/manifest.json").read_text()) if v["path"].split("/")[0] in ids]
assert sum(v["path"].endswith(".edf") for v in manifest)==20
specs=json.loads((ROOT/"phase100/protocol.json").read_text())["models"]
pr=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),subjects=ids,selection="Next contiguous20ISRUC cohortI patients041-060, no choice from signals,labels,diagnoses,orperformance; no replacement for input incompatibility.",development_exposure="Methods developed on prior SleepEDF and ISRUC III001-010/I021-039. Full public participants summary was previously downloaded but these subjects' rows were not consulted for design; individual event labels not read. Independent-person fixed-method validation, not independent-dataset validation.",models=specs,montages=["left","right"],modes=["raw","scalar","floor","diagonal","full"],ema=[1.,.8,.5],primary="CBraMod full minus scalar, left montage, EMA.5; three seed metrics averaged within each person, paired person bootstrap10000 seed101017. Positive mean and95%CI lower>0 support mean improvement on this cohort; report individual harm, no safetyclaim.",secondary="Diagonal ablation, full-versusraw/floor, othermodels, rightmontage, otherEMA and input-triggered subgroup; never replace primary.",reference_sha256=sha(ROOT/"phase99/source_reference.npz"),shape_protocol_sha256=sha(ROOT/"phase99/protocol.json"),scalar_protocol_sha256=sha(ROOT/"phase79/protocol.json"),input_compatibility=old["input_compatibility"],input_exclusions="Only absence of verified shared-reference triplets or incompatible declared units/sampling; record before that person's inference, no replacement. Other technical errors stop for audit.",label_map=old["label_map"],timeline=old["timeline"],label_access="Download/read individualevents only after all eligible people have completed predictions and prediction hashes sealed.",storage="Process one verified new task-temporary EDF at a time; retain header/provenance/derived-array hashes and predictions; remove only that scratch EDF after successful sealed person output or documented input exclusion. No persistent waveform copies, reproducible from manifest URL/hash.",cpu_ids=list(range(8,24)),gpu=True)
if (O/"protocol.json").exists():
 previous=json.loads((O/"protocol.json").read_text());pr["created_utc"]=previous["created_utc"];assert pr==previous
else:
 (O/"protocol.json").write_text(json.dumps(pr,indent=2));(O/"protocol.sha256").write_text(sha(O/"protocol.json"));(O/"manifest.json").write_text(json.dumps(manifest,indent=2))
reference=np.load(ROOT/"phase99/source_reference.npz")["reference"];rmsref=json.loads((ROOT/"phase79/protocol.json").read_text())["source_reference_rms"]
models={};snaps={}
for s in specs:
 assert sha(Path(s["path"]))==s["sha256"];ck=torch.load(s["path"],map_location="cpu",weights_only=False);kind=s["model"];m=Carrier("pretrained") if kind=="cbramod" else FullCarrier("mixed","pretrained") if kind=="labram" else SpectralNet(ck["dim"]);m.load_state_dict(ck["state"]);m.cuda().eval().requires_grad_(False);key=f'{kind}_seed{s["seed"]}';models[key]=m;snaps[key]=snapshot(m)
included=[];excluded=[];records=[]
def remove_scratch(path):
 assert path.parent.resolve()==(O/"scratch").resolve() and path.name in [s+".edf" for s in ids];path.unlink()
for sid in ids:
 done=O/"inputs"/(sid+"_complete.json");bad=O/"inputs"/(sid+"_excluded.json")
 if bad.exists():excluded.append(json.loads(bad.read_text()));continue
 if done.exists():
  rec=json.loads(done.read_text())
  for name,digest in rec["prediction_hashes"].items():assert sha(O/"predictions"/name)==digest
  included.append(sid);records.append(rec);continue
 entries=[v for v in manifest if v["path"].split("/")[0]==sid];edf=next(v for v in entries if v["path"].endswith(".edf"));path=O/"scratch"/(sid+".edf")
 assert shutil.disk_usage(ROOT).free>edf["size"]+250_000_000
 for v in entries:
  if v["path"].endswith(("_eeg.json","_channels.tsv")):checked_download(v,O/"metadata"/v["path"])
 print("DOWNLOAD",sid,edf["size"],flush=True);checked_download(edf,path);h=header(path);(O/"inputs"/(sid+"_header.json")).write_text(json.dumps(h,indent=2))
 try:
  left,lref=resolve(h["labels"],["F3","C3","O1"],["A2","M2"]);right,rref=resolve(h["labels"],["F4","C4","O2"],["A1","M1"]);channels=left+right;hi=[h["labels"].index(c) for c in channels];units=[h["units"][i] for i in hi];factor={"uv":1.,"mv":1000.,"v":1e6}
  un=[u.replace("µ","u").replace("μ","u").lower() for u in units]
  if any(u not in factor for u in un):raise ValueError(("unsupported physical units",units))
  factors=np.array([factor[u] for u in un]);sf=[int(h["samples_per_record"][i])/h["record_seconds"] for i in hi]
  if sf!=[200.]*6:raise ValueError(("incompatible sampling",sf))
 except ValueError as e:
  rec=dict(subject=sid,reason=str(e),source_sha256=edf["checksum"],before_person_predictions=True,labels_not_read=True);bad.write_text(json.dumps(rec,indent=2));excluded.append(rec);remove_scratch(path);print("INPUT EXCLUDED",sid,str(e),flush=True);continue
 with pyedflib.EdfReader(str(path),annotations_mode=0) as f:
  labels=f.getSignalLabels();fi=[labels.index(c) for c in channels];ns=f.getNSamples();assert len(set(int(ns[i]) for i in fi))==1;assert all(f.getSampleFrequency(i)==200 for i in fi);n=int(ns[fi[0]])//6000;leftover=int(ns[fi[0]])%6000;parts=[];err=0.
  for a,b,scale in zip(fi,hi,factors):err=max(err,float(abs((f.readSignal(a,start=0,n=1000)-direct_samples(path,h,b))*scale).max()))
  assert err<1e-8
  for first in range(0,n,64):
   count=min(64,n-first);native=np.stack([f.readSignal(i,start=first*6000,n=count*6000) for i in fi])*factors[:,None];windows=native.reshape(6,count,6000).transpose(1,0,2);x=transform(windows);parts.append(x)
   if first==0:
    prefix=transform(windows[:1]);changed=windows[:2].copy();changed[1]+=1000;assert np.array_equal(prefix,x[:1]) and np.array_equal(transform(changed)[:1],prefix)
  xx=np.concatenate(parts);assert np.isfinite(xx).all()
 rec=dict(subject=sid,windows=n,source_samples=int(ns[fi[0]]),trailing_partial_samples=leftover,channels=channels,shared_references=[lref,rref],units=units,units_to_microvolt=factors.tolist(),direct_physical_calibration_error=err,source_sha256=edf["checksum"],source_bytes=edf["size"],prediction_hashes={},checks=[],target_event_labels_read=False,derived_array_sha256={})
 for montage,sl in [("left",slice(0,2)),("right",slice(2,4))]:
  raw=xx[:,sl].copy();rec["derived_array_sha256"][montage]=__import__("hashlib").sha256(raw.tobytes()).hexdigest()
  scalar,gains,logstate,rms=normalize(raw,rmsref,"guarded_median")
  for mode in pr["modes"]:
   stats={}
   if mode=="raw":x=raw
   elif mode=="scalar":x=scalar
   elif mode=="floor":x=normalize(raw,rmsref,"floor")[0]
   else:
    x,aux,stats=shape_transform(raw,gains,reference,mode)
    assert stats["max_relative_energy_error_beforeclip"]<1e-12
    assert np.array_equal(shape_transform(raw[:65],gains[:65],reference,mode)[0],x[:65])
    future=raw[:66].copy();future[65]+=3;assert np.array_equal(shape_transform(future,gains[:66],reference,mode)[0][:65],x[:65])
    np.savez_compressed(O/"predictions"/f"{sid}_{montage}_{mode}_normalizer.npz",**aux)
   ff=features(x);ct=context(ff);assert np.array_equal(ct[:65],context(ff[:65]));xs={"context11":torch.from_numpy(ct),"cbramod":torch.from_numpy(prep(x,[.1,45])),"labram":preprocess(x).reshape(-1,2,2,15,200)};probs={};checks=[]
   for key,m in models.items():
    xp=xs[key.rsplit("_seed",1)[0]];p=probabilities(m,xp)
    with torch.no_grad():one=m(xp[:1].cuda()).softmax(1).cpu().numpy()
    delta=float(abs(one-p[:1]).max());assert delta<2e-5 and p.shape==(n,5) and np.isfinite(p).all() and abs(p.sum(1)-1).max()<3e-6
    assert all(torch.equal(v.cpu(),snaps[key][k]) for k,v in m.state_dict().items());probs[key]=p;checks.append(dict(model=key,batch_one_error=delta,state_exact=True))
   name=f"{sid}_{montage}_{mode}.npz";np.savez_compressed(O/"predictions"/name,**probs);rec["prediction_hashes"][name]=sha(O/"predictions"/name);rec["checks"].append(dict(montage=montage,mode=mode,triggered_windows=int((gains>1).sum()),models=checks,**stats))
  np.savez_compressed(O/"predictions"/f"{sid}_{montage}_scalar_normalizer.npz",gains=gains,log_rms_state=logstate,current_rms=rms)
 done.write_text(json.dumps(rec,indent=2));included.append(sid);records.append(rec);remove_scratch(path);print("PERSON SEALED",sid,n,"included",len(included),"excluded",len(excluded),flush=True)
cohort=dict(planned=ids,included=included,excluded=excluded,no_replacements=True,individual_event_labels_not_read=True)
(O/"cohort_lock.json").write_text(json.dumps(cohort,indent=2))
seal={p.name:sha(p) for p in (O/"predictions").glob("*.npz")}
(O/"prediction_hashes.json").write_text(json.dumps(seal,indent=2))
(O/"predictions_complete.json").write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),included=included,excluded=excluded,prediction_hashes_sha256=sha(O/"prediction_hashes.json"),protocol_sha256=sha(O/"protocol.json"),labels_not_read=True),indent=2))
(O/"input_prediction_audit.json").write_text(json.dumps(dict(status="PASS",records=records,cohort=cohort),indent=2));print("ALL PREDICTIONS SEALED",len(included),len(excluded),flush=True)
