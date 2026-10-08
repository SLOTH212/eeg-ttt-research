from phase79_amplitude import *
from phase99_covariance import transform,moments
from scipy.signal import lfilter
O=ROOT/"phase100";pr=json.loads((O/"protocol.json").read_text());reference=np.load(ROOT/"phase99/source_reference.npz")["reference"];models={}
for s in pr["models"]:
 ck=torch.load(s["path"],map_location="cpu",weights_only=False);k=s["model"];m=Carrier("pretrained") if k=="cbramod" else FullCarrier("mixed","pretrained") if k=="labram" else SpectralNet(ck["dim"]);m.load_state_dict(ck["state"]);m.cuda().eval().requires_grad_(False);models[f'{k}_seed{s["seed"]}']=m
def closed(a):
 d=np.sqrt(np.linalg.det(a));return (a+d*np.eye(2))/np.sqrt(np.trace(a)+2*d)
records=[]
for name in ["source_s35_native","healthy_sub-III001_left","patient_sub-I022_left"]:
 rec=json.loads((O/"predictions"/(name+"_audit.json")).read_text());c=rec["case"];raw=np.load(c["x"],mmap_mode="r");lp=ROOT/rec["baseline_path"];gain=np.load(lp.with_name(lp.stem+"_normalizer.npz"))["gains"]
 scalar,replayed,_,_=normalize(raw,json.loads((ROOT/"phase79/protocol.json").read_text())["source_reference_rms"],"guarded_median");assert np.array_equal(gain,replayed)
 cov=moments(raw);tr=np.trace(cov,axis1=1,axis2=2);valid=tr>=2e-12;shape=2*cov[valid]/tr[valid,None,None];state,_=lfilter([.01],[1,-.99],shape,axis=0,zi=.99*shape[:1]);states=np.empty_like(cov);states[:]=np.nan;states[valid]=state
 for mode in ["scalar","diagonal","full"]:
  if mode=="scalar":x=scalar;saved=np.load(lp);normerr=0.
  else:
   x,aux,stats=transform(raw,gain,reference,mode);saved=np.load(O/"predictions"/f"{name}_{mode}.npz");assert np.max(abs(aux["shape_states"][valid]-states[valid]))<1e-12
   manual=raw.copy();source=.99*reference+.01*np.eye(2)
   for t in np.flatnonzero((gain>1)&valid):
    target=.99*states[t]+.01*np.eye(2);mat=closed(source)@np.linalg.inv(closed(target)) if mode=="full" else np.diag(np.sqrt(np.diag(source)/np.diag(target)))
    z=mat@raw[t].astype(np.float64);z*=np.sqrt(np.mean(raw[t].astype(np.float64)**2)/np.mean(z*z))*gain[t];manual[t]=np.clip(z,-20,20)
   normerr=float(abs(manual-x).max());assert normerr<2e-6
  ff=features(x);xs={"context11":torch.from_numpy(context(ff)),"cbramod":torch.from_numpy(prep(x,[.1,45])),"labram":preprocess(x).reshape(-1,2,2,15,200)}
  for key,m in models.items():
   q=probabilities(m,xs[key.rsplit("_seed",1)[0]]);err=float(abs(q-saved[key]).max());assert err<2e-5 and np.array_equal(q.argmax(1),saved[key].argmax(1));records.append(dict(case=name,mode=mode,model=key,full_replay_probability_error=err,argmax_exact=True,independent_transform_error=normerr))
  print("REPLAY",name,mode,flush=True)
out=dict(status="PASS",streams=len(records),independent_closedform_transform=True,independent_scipy_state=True,scalar_gain_replay_exact=True,whole_record_inference_matches_selective_recompute=True,records=records)
(O/"independent_replay_audit.json").write_text(json.dumps(out,indent=2));print("PASS",len(records),flush=True)
