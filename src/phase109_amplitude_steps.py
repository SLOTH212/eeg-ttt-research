from phase79_amplitude import *
import datetime,time
O=ROOT/"phase109";O.mkdir(exist_ok=True);(O/"predictions").mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((ROOT/"phase79/protocol.json").read_text());specs=[s for s in old["models"] if s["model"]=="cbramod"]
pr=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),purpose="Source-only fixed synthetic amplitude downstep/recovery response of existing causal scalar controls. No learning or threshold search.",subjects=[5,15,25,35],models=specs,perturbation="Multiply bothrawchannels by0.1 only middlethird [floor(N/3),floor(2N/3)); original signal before/after; boundaries dependonlyrecordlength, notlabels.",methods=["reference_raw","corrupted_raw","reference_guard","corrupted_guard","reference_floor","corrupted_floor"],ema=[1.,.8,.5],bins="Warmup; downstep offsets0-9,10-49,50-199,>=200; recovery sameoffsetbins. Fixed before metrics; binsdiagnostic notonlinegates.",primary_scope="Descriptive shift penalty perexistingmethod against its own uncorruptedreference, plus corruptedguard/floor versuscorruptedraw. Four repeatedlyobserved sourcepeople; no realdomainvalidation or efficacyclaim.",normalizer_protocol_sha256=sha(ROOT/"phase79/protocol.json"),source_reference_rms=old["source_reference_rms"],cpu_ids=list(range(8,24)),gpu=True)
if (O/"protocol.json").exists():
 previous=json.loads((O/"protocol.json").read_text());pr["created_utc"]=previous["created_utc"];assert pr==previous
else:(O/"protocol.json").write_text(json.dumps(pr,indent=2));(O/"protocol.sha256").write_text(sha(O/"protocol.json"))
models={};snaps={}
for s in specs:
 path=ROOT/s["path"];assert sha(path)==s["sha256"];ck=torch.load(path,map_location="cpu",weights_only=False);m=Carrier("pretrained");m.load_state_dict(ck["state"]);m.cuda().eval().requires_grad_(False);key=f'cbramod_seed{s["seed"]}';models[key]=m;snaps[key]=snapshot(m)
records=[]
for sid in pr["subjects"]:
 dest=O/"predictions"/f"s{sid:02d}.npz";ap=O/"predictions"/f"s{sid:02d}_audit.json"
 if ap.exists():records.append(json.loads(ap.read_text()));continue
 folder=ROOT/("data/processed" if sid<20 else "phase2/processed");raw=np.load(folder/f"s{sid:02d}_x.npy",mmap_mode="r");n=len(raw);a=n//3;b=2*n//3;factor=np.ones(n);factor[a:b]=.1;corrupt=(raw*factor[:,None,None]).astype(np.float32);assert np.array_equal(corrupt[:a],raw[:a]) and np.array_equal(corrupt[b:],raw[b:]);probs={};traces=dict(factor=factor,downstep=a,recovery=b);checks=[];references={}
 for short,mode in [("raw","raw"),("guard","guarded_median"),("floor","floor")]:
  path=ROOT/"phase79/predictions"/f"source_s{sid:02d}_native_{mode}.npz";references[short]=dict(path=str(path),sha256=sha(path));legacy=np.load(path)
  if short=="raw":base=raw;x=corrupt
  else:
   base,bg,bs,br=normalize(raw,pr["source_reference_rms"],mode);x,cg,cs,cr=normalize(corrupt,pr["source_reference_rms"],mode);traces.update({short+"_reference_gain":bg,short+"_corrupted_gain":cg,short+"_reference_log_state":bs,short+"_corrupted_log_state":cs,short+"_reference_rms":br,short+"_corrupted_rms":cr})
   for end in [a+5,b+5]:
    xp,gp,_,_=normalize(corrupt[:end],pr["source_reference_rms"],mode);assert np.array_equal(xp,x[:end]) and np.array_equal(gp,cg[:end])
  changed=np.any(x!=base,axis=(1,2));prepared=torch.from_numpy(prep(x[changed],[.1,45])) if changed.any() else None
  for key,m in models.items():
   q=legacy[key].copy();probs[f"reference_{short}_{key}"]=q.copy()
   if changed.any():
    p=probabilities(m,prepared);q[changed]=p
    with torch.no_grad():one=m(prepared[:1].cuda()).softmax(1).cpu().numpy()
    error=float(abs(one-p[:1]).max());assert error<2e-5
   else:error=0.
   assert np.array_equal(q[~changed],legacy[key][~changed]) and np.isfinite(q).all();assert all(torch.equal(v.cpu(),snaps[key][k]) for k,v in m.state_dict().items());probs[f"corrupted_{short}_{key}"]=q;checks.append(dict(method=short,model=key,changed_inputs=int(changed.sum()),batch_one_error=error,state_exact=True,unaffected_predictions_exact=True))
 np.savez_compressed(dest,**probs);np.savez_compressed(O/"predictions"/f"s{sid:02d}_traces.npz",**traces);rec=dict(subject=sid,windows=n,downstep=a,recovery=b,raw_sha256=sha(folder/f"s{sid:02d}_x.npy"),references=references,checks=checks,prefix_replay_exact=True,boundaries_label_independent=True);ap.write_text(json.dumps(rec,indent=2));records.append(rec);print("STEP STRESS",sid,n,a,b,flush=True)
(O/"audit.json").write_text(json.dumps(dict(status="PASS",records=records,labels_not_read_for_perturbation_or_prediction=True,original_methods_unchanged=True),indent=2));(O/"complete.json").write_text(json.dumps(dict(people=4,models=3,methods=6)));print("COMPLETE",flush=True)
