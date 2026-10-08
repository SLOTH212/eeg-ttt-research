from phase77_predict import Carrier,FullCarrier,SpectralNet,ROOT,prep,preprocess,probabilities,snapshot,context
from phase78_amplitude import normalize as always_normalize
from phase79_amplitude import normalize as bounded_normalize
from phase76_prepare import features
from pathlib import Path
import torch,numpy as np,json,hashlib,time
OUT83=ROOT/"phase83"
def run():
 pr=json.loads((OUT83/"protocol.json").read_text());assert hashlib.sha256((OUT83/"protocol.json").read_bytes()).hexdigest()==(OUT83/"protocol.sha256").read_text();assert hashlib.sha256((ROOT/"phase79/protocol.json").read_bytes()).hexdigest()==pr["normalizer_protocol_sha256"];cohort=json.loads((OUT83/"cohort_lock_after_input_audit.json").read_text());eligible=cohort["included"];base=json.loads((ROOT/"phase79/protocol.json").read_text());models={};states={};out=OUT83/"predictions";out.mkdir(exist_ok=True)
 for spec in pr["models"]:
  path=Path(spec["path"]);assert hashlib.sha256(path.read_bytes()).hexdigest()==spec["sha256"];ck=torch.load(path,map_location="cpu",weights_only=False);kind=spec["model"];m=Carrier("pretrained") if kind=="cbramod" else FullCarrier("mixed","pretrained") if kind=="labram" else SpectralNet(ck["dim"]);m.load_state_dict(ck["state"]);m.cuda().eval().requires_grad_(False);key=f'{kind}_seed{spec["seed"]}';models[key]=m;states[key]=snapshot(m)
 for sid in eligible:
  done=out/(sid+"_audit.json")
  if done.exists():continue
  ready=OUT83/"processed"/(sid+"_input_audit.json")
  while not ready.exists():time.sleep(2)
  input_audit=json.loads(ready.read_text());checks=[];normal=[]
  for montage in pr["montages"]:
   raw=np.load(OUT83/"processed"/(sid+"_"+montage+"_x.npy"));rawfeatures=np.load(OUT83/"processed"/(sid+"_"+montage+"_features.npy"))
   for mode in pr["input_modes"]:
    if mode=="raw":xx=raw;ff=rawfeatures
    else:
     fn=always_normalize if mode=="logema01" else bounded_normalize;arg=.01 if mode=="logema01" else mode;xx,gains,logstate,rms=fn(raw,base["source_reference_rms"],arg);prefix=fn(raw[:65],base["source_reference_rms"],arg)[0];assert np.array_equal(prefix,xx[:65]);future=raw[:66].copy();future[65]+=7;assert np.array_equal(fn(future,base["source_reference_rms"],arg)[0][:65],prefix);ff=features(xx);np.savez_compressed(out/f"{sid}_{montage}_{mode}_normalizer.npz",gains=gains,log_rms_state=logstate,current_rms=rms);normal.append(dict(montage=montage,mode=mode,prefix_exact=True,future_perturbation_exact=True,gain_quantiles=np.quantile(gains,[0,.05,.5,.95,1]).tolist()))
    ct=context(ff);assert np.array_equal(ct[:65],context(ff[:65]));xs={"cbramod":torch.from_numpy(prep(xx,[.1,45])),"labram":preprocess(xx).reshape(-1,2,2,15,200),"context11":torch.from_numpy(ct),"relative":torch.from_numpy(np.concatenate([ff[:,24:48],ff[:,50:52]],axis=1))};preds={}
    for key,m in models.items():
     kind=key.rsplit("_seed",1)[0]
     if kind=="relative" and mode!="raw":continue
     x=xs[kind];p=probabilities(m,x);assert p.shape==(len(raw),5) and np.isfinite(p).all();preds[key]=p
     with torch.no_grad():t=len(x)//2;single=m(x[t:t+1].cuda()).softmax(1).cpu().numpy()[0];err=float(abs(single-p[t]).max());assert err<2e-5 and single.argmax()==p[t].argmax()
     assert all(torch.equal(v.cpu(),states[key][n]) for n,v in m.state_dict().items());checks.append(dict(montage=montage,mode=mode,model=key,windows=len(raw),batch_one_error=err,model_state_unchanged=True))
    np.savez_compressed(out/f"{sid}_{montage}_{mode}.npz",**preds);print("PATIENT FIXED PREDICTION",sid,montage,mode,len(raw),flush=True)
  done.write_text(json.dumps(dict(subject=sid,input_source_sha256=input_audit["source_sha256"],checks=checks,normalizer_checks=normal,no_event_labels_read=True),indent=2))
 allchecks=[json.loads((out/(sid+"_audit.json")).read_text()) for sid in eligible];(OUT83/"prediction_audit.json").write_text(json.dumps(dict(status="PASS",records=allchecks,all_model_states_unchanged=True,all_event_labels_withheld=True,planned_subjects=20,subjects=len(eligible),excluded=cohort["excluded"]),indent=2));(OUT83/"predictions_complete.json").write_text(json.dumps(dict(complete=True,planned_subjects=20,subjects=len(eligible),excluded=cohort["excluded"],allpredictions_beforeeventlabels=True)))
if __name__=="__main__":run()
