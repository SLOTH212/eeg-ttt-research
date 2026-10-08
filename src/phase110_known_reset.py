from phase79_amplitude import *
import datetime
O=ROOT/"phase110";O.mkdir(exist_ok=True);(O/"predictions").mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
parent=json.loads((ROOT/"phase109/protocol.json").read_text());pr=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),purpose="Known synthetic-boundary reset diagnostic for inputstate lag. Not gradientTTT, not deployable changepointdetector or performancebound.",subjects=parent["subjects"],models=parent["models"],methods=["reference_reset_guard","corrupted_reset_guard","reference_reset_floor","corrupted_reset_floor"],reset="Reset existing logRMS estimator at both fixed middlethird boundaries; normalize each ofthreechunks with unchanged phase79rule. Applysame shamresets to uncorruptedsource to separate resetitself.",comparisons="Reset-versus-persistent within same signal andmethod; remainingcorruption penalty under resets. No parametersearch.",ema=parent["ema"],normalizer_reference=parent["source_reference_rms"],parent_protocol_sha256=sha(ROOT/"phase109/protocol.json"),scope="Four repeatedlyseen sourcepeople, factor0.1 sameknownschedule. Labelsnotusedforreset/prediction. Doesnot validate a real changepointpolicy.")
if (O/"protocol.json").exists():
 old=json.loads((O/"protocol.json").read_text());pr["created_utc"]=old["created_utc"];assert pr==old
else:(O/"protocol.json").write_text(json.dumps(pr,indent=2));(O/"protocol.sha256").write_text(sha(O/"protocol.json"))
models={};snaps={}
for s in pr["models"]:
 path=ROOT/s["path"];assert sha(path)==s["sha256"];ck=torch.load(path,map_location="cpu",weights_only=False);m=Carrier("pretrained");m.load_state_dict(ck["state"]);m.cuda().eval().requires_grad_(False);key=f'cbramod_seed{s["seed"]}';models[key]=m;snaps[key]=snapshot(m)
def reset_normalize(x,mode,a,b):
 parts=[normalize(x[lo:hi],pr["normalizer_reference"],mode) for lo,hi in [(0,a),(a,b),(b,len(x))]];return tuple(np.concatenate([p[k] for p in parts]) for k in range(4))
records=[]
for sid in pr["subjects"]:
 ap=O/"predictions"/f"s{sid:02d}_audit.json"
 if ap.exists():records.append(json.loads(ap.read_text()));continue
 folder=ROOT/("data/processed" if sid<20 else "phase2/processed");raw=np.load(folder/f"s{sid:02d}_x.npy",mmap_mode="r");n=len(raw);a=n//3;b=2*n//3;factor=np.ones(n);factor[a:b]=.1;corrupt=(raw*factor[:,None,None]).astype(np.float32);old=np.load(ROOT/"phase109/predictions"/f"s{sid:02d}.npz");probs={};traces={};checks=[]
 for signal,inputx in [("reference",raw),("corrupted",corrupt)]:
  for short,mode in [("guard","guarded_median"),("floor","floor")]:
   base=normalize(inputx,pr["normalizer_reference"],mode)[0];x,g,states,rms=reset_normalize(inputx,mode,a,b);method=f"{signal}_reset_{short}";traces[method+"_gain"]=g;traces[method+"_log_state"]=states;assert np.array_equal(x[:a],base[:a])
   # Each reset segment starts from its first currentwindow, no laterwindow information.
   for lo,hi in [(0,a),(a,b),(b,n)]:
    pref=normalize(inputx[lo:min(lo+33,hi)],pr["normalizer_reference"],mode)[0];assert np.array_equal(pref,x[lo:min(lo+33,hi)])
   changed=np.any(x!=base,axis=(1,2));xp=torch.from_numpy(prep(x[changed],[.1,45])) if changed.any() else None
   for key,m in models.items():
    q=old[f"{signal}_{short}_{key}"].copy()
    if changed.any():
     p=probabilities(m,xp);q[changed]=p
     with torch.no_grad():one=m(xp[:1].cuda()).softmax(1).cpu().numpy()
     err=float(abs(one-p[:1]).max());assert err<2e-5
    else:err=0.
    assert np.array_equal(q[~changed],old[f"{signal}_{short}_{key}"][~changed]) and np.isfinite(q).all();assert all(torch.equal(v.cpu(),snaps[key][k]) for k,v in m.state_dict().items());probs[f"{method}_{key}"]=q;checks.append(dict(method=method,model=key,changed_windows=int(changed.sum()),batch_one_error=err,state_exact=True,unaffected_prediction_exact=True))
 np.savez_compressed(O/"predictions"/f"s{sid:02d}.npz",**probs);np.savez_compressed(O/"predictions"/f"s{sid:02d}_traces.npz",**traces);rec=dict(subject=sid,checks=checks,reset_boundaries=[a,b],warmup_exact=True,segment_prefix_exact=True);ap.write_text(json.dumps(rec,indent=2));records.append(rec);print("KNOWN RESET",sid,flush=True)
(O/"audit.json").write_text(json.dumps(dict(status="PASS",records=records,no_labels_for_reset=True,no_gradient_updates=True),indent=2));(O/"complete.json").write_text(json.dumps(dict(people=4,models=3,methods=4)));print("COMPLETE",flush=True)
