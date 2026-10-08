from phase77_predict import Carrier,FullCarrier,SpectralNet,ROOT,prep,preprocess,probabilities,snapshot,context
from phase79_amplitude import normalize
from phase76_prepare import features
from pathlib import Path
import torch,numpy as np,json,hashlib,datetime
OUT81=ROOT/"phase81"
def old_probability(model,seed,sid):
 if model=="cbramod":p=ROOT/"phase70"/f"s{sid:02d}.npz" if seed==17 else ROOT/"phase71/evaluation"/f"seed{seed}"/f"s{sid:02d}.npz";return np.load(p)["pretrained" if seed==17 else "p"]
 if model=="labram":p=ROOT/"phase21/evaluation/mixed_pretrained"/f"s{sid:02d}.npz" if seed==17 else ROOT/"phase27/evaluation"/f"mixed_pretrained_seed{seed}"/f"s{sid:02d}.npz";return np.load(p)["source_tuned"]
 p=ROOT/"phase4/results"/f"context11_mixed_{seed}"/f"predictions_s{sid:02d}.npz" if sid<40 else ROOT/"phase6/fresh_baselines"/f"context11_mixed_{seed}_s{sid:02d}.npz";return np.load(p)["frozen"]
def run():
 OUT81.mkdir(exist_ok=True);base=json.loads((ROOT/"phase79/protocol.json").read_text());subjects=json.loads((ROOT/"phase70/protocol.json").read_text())["subjects"];modelspec=[v for v in base["models"] if v["model"]!="relative"];pr=dict(locked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope="Posthoc fixed amplitude-guard risk check on30 alreadyobserved SleepEDF transferpeople; notfreshblindtest",subjects=subjects,models=modelspec,modes=["raw","guarded_median","floor"],ema=[1,.8,.5],primary="CBraMod guarded_median vsraw, sameEMA.5, withinperson3seedmean",normalizer_protocol=str(ROOT/"phase79/protocol.json"),normalizer_protocol_sha256=hashlib.sha256((ROOT/"phase79/protocol.json").read_bytes()).hexdigest(),no_new_threshold_or_model_selection=True,identity_reuse="Unchanged currentwindow predictions reused exact. Context11 propagates changed featurewindow mask through current+next10windows; only affected predictions recomputed.")
 (OUT81/"protocol.json").write_text(json.dumps(pr,indent=2));out=OUT81/"predictions";out.mkdir(exist_ok=True);models={};states={};checks=[];nchecks=[]
 for spec in modelspec:
  path=Path(spec["path"]);assert hashlib.sha256(path.read_bytes()).hexdigest()==spec["sha256"];ck=torch.load(path,map_location="cpu",weights_only=False);kind=spec["model"];m=Carrier("pretrained") if kind=="cbramod" else FullCarrier("mixed","pretrained") if kind=="labram" else SpectralNet(ck["dim"]);m.load_state_dict(ck["state"]);m.cuda().eval().requires_grad_(False);key=f'{kind}_seed{spec["seed"]}';models[key]=m;states[key]=snapshot(m)
 for sid in subjects:
  folder=ROOT/("data/processed" if sid<20 else "phase2/processed" if sid<40 else "phase6/processed");raw=np.load(folder/f"s{sid:02d}_x.npy");ff=np.load(folder/f"s{sid:02d}_meta.npz")["features"];baseline={f'{s["model"]}_seed{s["seed"]}':old_probability(s["model"],s["seed"],sid) for s in modelspec};np.savez_compressed(out/f"s{sid:02d}_raw.npz",**baseline)
  for mode in ["guarded_median","floor"]:
   xx,gain,state,rms=normalize(raw,base["source_reference_rms"],mode);prefix=normalize(raw[:65],base["source_reference_rms"],mode)[0];assert np.array_equal(prefix,xx[:65]);changed=np.any(xx!=raw,axis=(1,2));contextchanged=np.convolve(changed.astype(int),np.ones(11,dtype=int),mode="full")[:len(raw)]>0;newff=ff.copy()
   if changed.any():newff[changed]=features(xx[changed])
   ct=context(newff);assert np.array_equal(ct[:65],context(newff[:65]));preds={}
   for key,m in models.items():
    kind=key.rsplit("_seed",1)[0];mask=contextchanged if kind=="context11" else changed;idx=np.flatnonzero(mask);p=baseline[key].copy()
    if len(idx):
     x=torch.from_numpy(ct[idx]) if kind=="context11" else torch.from_numpy(prep(xx[idx],[.1,45])) if kind=="cbramod" else preprocess(xx[idx]).reshape(-1,2,2,15,200);p[idx]=probabilities(m,x)
    assert np.array_equal(p[~mask],baseline[key][~mask]);t=int(np.flatnonzero(changed)[0]) if changed.any() else len(raw)//2;checkx=torch.from_numpy(ct[t:t+1]) if kind=="context11" else torch.from_numpy(prep(xx[t:t+1],[.1,45])) if kind=="cbramod" else preprocess(xx[t:t+1]).reshape(-1,2,2,15,200)
    with torch.no_grad():single=m(checkx.cuda()).softmax(1).cpu().numpy()[0];err=float(abs(single-p[t]).max());assert err<2e-5 and single.argmax()==p[t].argmax()
    assert all(torch.equal(v.cpu(),states[key][n]) for n,v in m.state_dict().items());preds[key]=p;checks.append(dict(subject=sid,mode=mode,model=key,recomputed_windows=len(idx),reused_windows=int((~mask).sum()),point_replay_error=err,unchanged_predictions_exact=True,state_unchanged=True))
   np.savez_compressed(out/f"s{sid:02d}_{mode}.npz",**preds);np.savez_compressed(out/f"s{sid:02d}_{mode}_normalizer.npz",gains=gain,current_rms=rms,log_rms_state=state,changed=changed,context_changed=contextchanged);nchecks.append(dict(subject=sid,mode=mode,windows=len(raw),changed_windows=int(changed.sum()),context_changed_windows=int(contextchanged.sum()),gain_max=float(gain.max()),prefix_exact=True));print("SLEEPEDF AMPLITUDE RISK",sid,mode,int(changed.sum()),len(raw),flush=True)
 (OUT81/"execution_audit.json").write_text(json.dumps(dict(status="PASS",prediction_checks=checks,normalizer_checks=nchecks,all_states_unchanged=True,source_parameters_fixed=True,no_targetlabelsread=True),indent=2));(OUT81/"complete.json").write_text(json.dumps(dict(complete=True,people=30,classifier_seeds=3,modeltypes=3,modes=3)))
if __name__=="__main__":run()
