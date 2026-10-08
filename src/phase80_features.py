from pathlib import Path
from phase77_predict import Carrier,FullCarrier,SpectralNet,ROOT,prep,preprocess,probabilities,snapshot,context
from phase79_amplitude import normalize
from phase76_prepare import features
import torch,numpy as np,json,hashlib,datetime
OUT80=ROOT/"phase80"
def run():
 base=json.loads((ROOT/"phase79/protocol.json").read_text());pr=dict(locked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),purpose="Source-only no-gradient prototype-memory baseline on existing frozen checkpoints",subjects=[5,15,25,35],models=base["models"],input_modes=["raw","guarded_median"],relative_model_input="raw only",filter_K=20,first_window="exact original frozen classifier prediction",causal_order="predict with prior supports, then append current representation/pseudolabel from frozen head, retain20lowestentropy perclass",methods=["frozen","source_prototype_static","t3a20_causal"],ema=[1,.8,.5],primary="CBraMod raw input t3a20_causal vsfrozen andsourceprototype atsameEMA.5, withinperson3seedmean",advance_rule="meanF1 greaterthan both staticcontrols and>=3/4people improve overfrozen; no targetevaluationwithoutsourcepass",official_protocol_difference="Official T3A inserts current before currentprediction; ours moves insertion afterprediction, andfirstwindow uses originalfrozen probabilities. Scoped causal version, notexactoriginalpaperprotocol.",upstream=json.loads((OUT80/"upstream_provenance.json").read_text()),extra_source_training_labels=0,no_target_labels=True,all_backbone_classifier_parameters_frozen=True)
 if not (OUT80/"protocol.json").exists():(OUT80/"protocol.json").write_text(json.dumps(pr,indent=2))
 out=OUT80/"cache";out.mkdir(exist_ok=True);checks=[]
 for spec in pr["models"]:
  path=Path(spec["path"]);assert hashlib.sha256(path.read_bytes()).hexdigest()==spec["sha256"];ck=torch.load(path,map_location="cpu",weights_only=False);kind=spec["model"];m=Carrier("pretrained") if kind=="cbramod" else FullCarrier("mixed","pretrained") if kind=="labram" else SpectralNet(ck["dim"]);m.load_state_dict(ck["state"]);m.cuda().eval().requires_grad_(False);state=snapshot(m);key=f'{kind}_seed{spec["seed"]}';np.savez_compressed(out/(key+"_head.npz"),weight=m.head.weight.detach().cpu().numpy(),bias=m.head.bias.detach().cpu().numpy())
  for sid in pr["subjects"]:
   folder=ROOT/("data/processed" if sid<20 else "phase2/processed");raw=np.load(folder/f"s{sid:02d}_x.npy")
   for mode in pr["input_modes"]:
    if kind=="relative" and mode!="raw":continue
    x=raw if mode=="raw" else normalize(raw,base["source_reference_rms"],mode)[0]
    if kind=="cbramod":xx=torch.from_numpy(prep(x,[.1,45]))
    elif kind=="labram":xx=preprocess(x).reshape(-1,2,2,15,200)
    else:
     ff=np.load(folder/f"s{sid:02d}_meta.npz")["features"] if mode=="raw" else features(x);xx=torch.from_numpy(context(ff) if kind=="context11" else np.concatenate([ff[:,24:48],ff[:,50:52]],axis=1))
    chunks=[]
    def hook(module,args):chunks.append(args[0].detach().cpu().clone())
    handle=m.head.register_forward_pre_hook(hook);p=probabilities(m,xx);handle.remove();z=torch.cat(chunks).numpy();old=np.load(ROOT/"phase79/predictions"/f"source_s{sid:02d}_native_{mode}.npz")[key];err=float(abs(p-old).max());assert err<2e-5 and np.array_equal(p.argmax(1),old.argmax(1))
    with torch.no_grad():cpu=torch.nn.functional.linear(torch.from_numpy(z),m.head.weight.cpu(),m.head.bias.cpu()).softmax(1).numpy()
    cpuerr=float(abs(cpu-p).max());assert cpuerr<2e-5 and np.array_equal(cpu.argmax(1),p.argmax(1));assert all(torch.equal(v.cpu(),state[n]) for n,v in m.state_dict().items())
    np.savez_compressed(out/f"{key}_s{sid:02d}_{mode}.npz",z=z,p=old);checks.append(dict(model=key,subject=sid,input_mode=mode,windows=len(raw),classifier_input_dim=z.shape[1],saved_probability_error=err,cpu_head_error=cpuerr,state_unchanged=True));print("PROTOTYPE FEATURES",key,sid,mode,z.shape,flush=True)
  del m;torch.cuda.empty_cache()
 (OUT80/"cache_audit.json").write_text(json.dumps(dict(status="PASS",cases=checks,head_inputs_verified=True,no_targetdata=True,all_states_unchanged=True),indent=2))
if __name__=="__main__":run()
