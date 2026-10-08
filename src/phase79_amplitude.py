from phase77_predict import Carrier,FullCarrier,SpectralNet,ROOT,prep,preprocess,probabilities,snapshot,context
from phase76_prepare import features
import torch,numpy as np,json,hashlib
OUT79=ROOT/"phase79"
def normalize(raw,reference,mode):
 pr=json.loads((OUT79/"protocol.json").read_text());lower=pr["lower_source_rms_quantile"];rms=np.sqrt(np.mean(raw.astype(np.float64)**2,axis=(1,2)));estimate=None;gains=[];states=[]
 for value in rms:
  if value<1e-6:gains.append(1.);states.append(np.nan if estimate is None else estimate);continue
  log=float(np.log(value));estimate=log if estimate is None else .99*estimate+.01*log;estimated=float(np.exp(estimate))
  if mode=="guarded_median":gain=reference/estimated if estimated<lower else 1.
  else:assert mode=="floor";gain=max(1.,lower/estimated)
  gains.append(float(np.clip(gain,1,100)));states.append(estimate)
 gains=np.array(gains);result=np.clip(raw*gains[:,None,None],-20,20).astype(np.float32);return result,gains,np.array(states),rms
def run():
 pr=json.loads((OUT79/"protocol.json").read_text());assert hashlib.sha256((OUT79/"protocol.json").read_bytes()).hexdigest()==(OUT79/"protocol.sha256").read_text();models={};states={};checks=[];nchecks=[]
 for spec in pr["models"]:
  p=ROOT/spec["path"];assert hashlib.sha256(p.read_bytes()).hexdigest()==spec["sha256"];ck=torch.load(p,map_location="cpu",weights_only=False);kind=spec["model"]
  m=Carrier("pretrained") if kind=="cbramod" else FullCarrier("mixed","pretrained") if kind=="labram" else SpectralNet(ck["dim"]);m.load_state_dict(ck["state"]);m.cuda().eval().requires_grad_(False);key=f'{kind}_seed{spec["seed"]}';models[key]=m;states[key]=snapshot(m)
 out=OUT79/"predictions";out.mkdir(exist_ok=True);cases=[("source",f"s{s:02d}","native") for s in pr["source_validation"]]+[("external",s,m) for s in pr["external_participants"] for m in pr["montages"]]
 for domain,sid,montage in cases:
  if domain=="source":
   number=int(sid[1:]);folder=ROOT/("data/processed" if number<20 else "phase2/processed");raw=np.load(folder/(sid+"_x.npy"));basefeatures=np.load(folder/(sid+"_meta.npz"))["features"]
  else:
   folder=ROOT/"phase76/processed";raw=np.load(folder/(sid+"_"+montage+"_x.npy"));basefeatures=np.load(folder/(sid+"_"+montage+"_features.npy"))
  for mode in ["raw"]+list(pr["normalizers"]):
   dest=out/f"{domain}_{sid}_{montage}_{mode}.npz";preds={}
   if mode=="raw":
    xx=raw;ff=basefeatures
    old=np.load(ROOT/"phase78/predictions"/f"{domain}_{sid}_{montage}_raw.npz");preds={key:old[key] for key in old.files}
   else:
    xx,gain,logstate,rms=normalize(raw,pr["source_reference_rms"],pr["normalizers"][mode]);first,fg,fs,fr=normalize(raw[:65],pr["source_reference_rms"],pr["normalizers"][mode]);assert np.array_equal(first,xx[:65]) and np.array_equal(fg,gain[:65]);changed=raw[:66].copy();changed[65]+=3;replay=normalize(changed,pr["source_reference_rms"],pr["normalizers"][mode])[0];assert np.array_equal(replay[:65],xx[:65]);ff=features(xx)
    np.savez_compressed(out/f"{domain}_{sid}_{montage}_{mode}_normalizer.npz",gains=gain,log_rms_state=logstate,current_rms=rms)
    nchecks.append(dict(domain=domain,subject=sid,montage=montage,mode=mode,windows=len(raw),prefix_exact=True,future_perturbation_exact=True,gain_quantiles=np.quantile(gain,[0,.05,.5,.95,1]).tolist(),source_reference=pr["source_reference_rms"],no_labels_read=True))
   ct=context(ff);assert np.array_equal(ct[:33],context(ff[:33]));xs={"context11":torch.from_numpy(ct),"relative":torch.from_numpy(np.concatenate([ff[:,24:48],ff[:,50:52]],axis=1))}
   if mode!="raw" or domain=="source":xs.update(cbramod=torch.from_numpy(prep(xx,[.1,45])),labram=preprocess(xx).reshape(-1,2,2,15,200))
   for key,m in models.items():
    kind=key.rsplit("_seed",1)[0]
    if kind=="relative" and mode!="raw":continue
    if key in preds:continue
    x=xs[kind];p=probabilities(m,x);assert p.shape==(len(raw),5) and np.isfinite(p).all();preds[key]=p
    with torch.no_grad():t=len(x)//2;single=m(x[t:t+1].cuda()).softmax(1).cpu().numpy()[0];err=float(abs(single-p[t]).max());assert err<2e-5 and single.argmax()==p[t].argmax()
    assert all(torch.equal(v.cpu(),states[key][n]) for n,v in m.state_dict().items());checks.append(dict(domain=domain,subject=sid,montage=montage,mode=mode,model=key,batch_one_error=err,state_unchanged=True))
   np.savez_compressed(dest,**preds);print("CAUSAL AMPLITUDE",domain,sid,montage,mode,len(preds),flush=True)
 (OUT79/"execution_audit.json").write_text(json.dumps(dict(status="PASS",prediction_checks=checks,normalizer_checks=nchecks,all_model_states_unchanged=True,source_reference_only=True,no_external_labels_read=True),indent=2));(OUT79/"complete.json").write_text(json.dumps(dict(complete=True,cases=len(cases))))
if __name__=="__main__":run()
