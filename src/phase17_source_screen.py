from phase16_masked_pulses import *
from phase2_temporal import ema
OUT=ROOT/"phase17"
def run(source,objective,lr):
 out=OUT/f"{source}_{objective}_{lr}";out.mkdir(exist_ok=True)
 if (out/"complete.json").exists():return
 model=make_encoder(True)
 for module in model.modules():
  if isinstance(module,nn.LayerNorm):
   for p in module.parameters():p.requires_grad_(True)
 params=[p for p in model.parameters() if p.requires_grad];initial=[p.detach().clone() for p in params];writable={n for n,p in model.named_parameters() if p.requires_grad};fixed={n:v.clone() for n,v in model.state_dict().items() if n not in writable};sd=torch.load(ROOT/"phase14/weights_only.pt",map_location="cpu",weights_only=True);mask_token=sd["student.mask_token"];lmw=sd["student.lm_head.weight"];lmb=sd["student.lm_head.bias"]
 with (ROOT/"phase14/probes"/f"{source}_pretrained400/model.pkl").open("rb") as f:carrier=pickle.load(f)
 scale=torch.tensor(carrier["scaler"].scale_,dtype=torch.float32);mean=torch.tensor(carrier["scaler"].mean_,dtype=torch.float32);W=torch.tensor(carrier["model"].coef_,dtype=torch.float32);b=torch.tensor(carrier["model"].intercept_,dtype=torch.float32)
 def logits(x):
  z=model.forward_features(x,input_chans=[0,127,128],return_patch_tokens=True).mean(1).reshape(-1,400);return nn.functional.linear((z-mean)/scale,W,b)
 results=[]
 for sid in [5,15,25,35]:
  dest=out/f"s{sid:02d}.json"
  if dest.exists():results.append(json.loads(dest.read_text()));continue
  cache=OUT/"cache";startwait=time.monotonic()
  while not (cache/f"s{sid:02d}.json").exists():
   if time.monotonic()-startwait>1800:raise TimeoutError("source cache")
   time.sleep(2)
  xs=np.load(cache/f"s{sid:02d}_x.npy",mmap_mode="r");targets=np.load(cache/f"s{sid:02d}_tokens.npy",mmap_mode="r");data=ROOT/("data/processed" if sid<20 else "phase2/processed");y=np.load(data/f"s{sid:02d}_meta.npz")["y"];frozen_features=np.load(ROOT/"phase14/features"/f"s{sid:02d}.npz")["pretrained"];frozen=carrier["model"].predict_proba(carrier["scaler"].transform(frozen_features));predictions=[];losses=[];drifts=[];start=time.perf_counter()
  with torch.no_grad():
   for p,v in zip(params,initial):p.copy_(v)
  for t in range(len(xs)):
   x=torch.from_numpy(np.array(xs[t]));model.zero_grad(set_to_none=True)
   if objective=="entropy":
    lg=logits(x);lp=lg.log_softmax(1);loss=-(lp.exp()*lp).sum(1).mean();p=lp.exp().detach().numpy()[0]
   else:
    with torch.no_grad():p=logits(x).softmax(1).numpy()[0]
    rng=np.random.default_rng(100000*sid+t);mask=torch.zeros(2,30,dtype=torch.bool)
    for j in range(2):mask[j,rng.permutation(30)[:15]]=True
    target=torch.from_numpy(np.array(targets[t]).astype(np.int64));terms=[]
    for mk in [mask,~mask]:terms.append(nn.functional.cross_entropy(nn.functional.linear(masked_tokens(model,x,mk,mask_token)[mk],lmw,lmb),target[mk]))
    loss=(terms[0]+terms[1])/2
   predictions.append(p);loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm) and torch.isfinite(loss)
   with torch.no_grad():
    for param,v in zip(params,initial):
     param.add_(param.grad,alpha=-lr);delta=param-v;limit=.1*max(float(v.square().mean().sqrt()),1.)*np.sqrt(v.numel());length=float(delta.norm())
     if length>limit:param.copy_(v+delta*(limit/length))
   if t%100==0:losses.append([t,float(loss.detach())]);drifts.append([t,float(torch.sqrt(sum((p-v).square().sum() for p,v in zip(params,initial))))])
  predictions=np.array(predictions);assert np.isfinite(predictions).all();assert np.max(np.abs(predictions[0]-frozen[0]))<.002;assert all(torch.equal(v,model.state_dict()[n]) for n,v in fixed.items());np.savez_compressed(out/f"s{sid:02d}.npz",y=y,frozen=frozen,weak_ema=ema(frozen,.8),adapted=predictions)
  result=dict(subject=sid,source=source,objective=objective,lr=lr,frozen=metrics(y,frozen),weak_ema=metrics(y,ema(frozen,.8)),adapted=metrics(y,predictions),seconds=time.perf_counter()-start,loss_trace=losses,drift_trace=drifts,first_prediction_max_error=float(np.max(np.abs(predictions[0]-frozen[0]))),fixed_state_unchanged=True);dest.write_text(json.dumps(result,indent=2));results.append(result);print("LN SOURCE SCREEN",source,objective,lr,sid,result["adapted"]["macro_f1"],result["seconds"],flush=True)
 (out/"complete.json").write_text(json.dumps(dict(source=source,objective=objective,lr=lr,source_validation_only=True,results=results,mean_frozen=float(np.mean([v["frozen"]["macro_f1"] for v in results])),mean_adapted=float(np.mean([v["adapted"]["macro_f1"] for v in results]))),indent=2));print("SOURCE SCREEN COMPLETE",source,objective,lr,flush=True)
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--source",required=True);parser.add_argument("--objective",required=True);parser.add_argument("--lr",type=float,required=True);a=parser.parse_args();run(a.source,a.objective,a.lr)
