from phase86_meta_waveform import *
from phase27_train import FullCarrier,prepared
from phase119_models import ResidualLogitHead,transform
import datetime
O=ROOT/"phase121";O.mkdir(exist_ok=True);C=O/"logits";C.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
@torch.inference_mode()
def hp(m,x):
 m.eval();return np.concatenate([m(x[a:a+1024]).softmax(1).cpu().numpy() for a in range(0,len(x),1024)])
def run():
 train=[0,1,2,10,11,20,21,30,31,32];val=[5,15,25,35];seeds=[17,29,43];variants=["current","repeat11","history11"]
 specs=[s for s in json.loads((ROOT/"phase79/protocol.json").read_text())["models"] if s["model"]=="labram"]
 pr=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),train=train,validation=val,seeds=seeds,variants=variants,parents=specs,fixed_recipe_reference_sha256=sha(ROOT/"phase119/protocol.json"),residual_implementation_sha256=sha(ROOT/"src/phase119_models.py"),features="Frozen LaBraM5class pre-softmaxlogits, subtract samewindow classmean for translationinvariance. Current5dim, repeatcurrent11times55dim, orcurrent+past10windows55dim; edgepad atstart. No labels in features.",architecture="Residual SpectralNet64/32 BN/ELU/dropout.25, source-onlymean/std floor.1. Add MLPoutput to originalcurrent centeredlogits. Lastlinear initializedexactzero, soepoch0 equalsparent. Repeat/history55dim have identicalinitial trainableparameters.",training="30epochs,AdamWlr.001wd.001,batch128,sqrtinverseclassweights normalizedmean1; same dedicated seedpermutations and dropout seed100000+seed afterconstructor. Parent frozen, onlyresidualheadtrained.",selection="Earliest best rawmean4sourceMacroF1 amongepoch0..30 withnativeguards: meanwithin1pp and eachpersonwithin2pp of parent. Epoch0fallback. EMAsecondaryonly.",advance="Onlyconsider existingtargetexploratoryreplication if3seed-mean withinperson history selected beats bothrepeat11 andfrozen mean, history-repeat positive inatleast3/4sourcepeople, allsource/checkpoint/prefixauditsPASS. No online or target adaptation; failedgate means no targetinference.",scope="Sourcepeoplealreadyobserved; extra30headepochs supervisedlabels and frozenparenttrainingbudget disclosed. Static causalreadout, notTTT or teacher distillation; no newrawsamples.",cpu_ids=list(range(8,24)),gpu=True)
 p=O/"protocol.json"
 if p.exists():
  old=json.loads(p.read_text());pr["created_utc"]=old["created_utc"];assert old==pr
 else:p.write_text(json.dumps(pr,indent=2));(O/"protocol.sha256").write_text(sha(p))
 audits=[]
 for sp in specs:
  seed=sp["seed"];ip=Path(sp["path"]);assert sha(ip)==sp["sha256"];m=FullCarrier("mixed","pretrained").cuda();base=torch.load(ip,map_location="cpu",weights_only=False)["state"];m.load_state_dict(base);m.eval().requires_grad_(False);state=snapshot(m)
  for sid in train+val:
   dest=C/f"seed{seed}_s{sid:02d}.npz";ap=dest.with_name(dest.stem+"_audit.json")
   if dest.exists() and ap.exists():
    a=json.loads(ap.read_text());assert sha(dest)==a["sha256"];audits.append(a);continue
   x,y=prepared(sid);parts=[]
   with torch.inference_mode():
    for bx in x.split(128):
     z=m(bx.cuda());parts.append((z-z.mean(1,keepdim=True)).cpu().numpy())
   logits=np.concatenate(parts);assert np.isfinite(logits).all()
   with torch.inference_mode():
    zz=m(x[:17].cuda());prefix=(zz-zz.mean(1,keepdim=True)).cpu().numpy()
   err=float(abs(prefix-logits[:17]).max());assert err<1e-4
   prob=torch.from_numpy(logits).softmax(1).numpy();legacyerr=None
   if sid in val:
    legacy=np.load(ROOT/"phase79/predictions"/f"source_s{sid:02d}_native_raw.npz")[f"labram_seed{seed}"];legacyerr=float(abs(prob-legacy).max());assert legacyerr<2e-5 and np.array_equal(prob.argmax(1),legacy.argmax(1))
   np.savez_compressed(dest,logits=logits,y=y);a=dict(seed=seed,subject=sid,sha256=sha(dest),parent_sha256=sha(ip),prefix_logit_error=err,legacy_probability_error=legacyerr);ap.write_text(json.dumps(a,indent=2));audits.append(a)
  assert all(torch.equal(v.cpu(),state[k]) for k,v in m.state_dict().items());del m;torch.cuda.empty_cache();print("LOGITS",seed,flush=True)
 (O/"feature_audit.json").write_text(json.dumps(dict(status="PASS",records=audits,parent_states_unchanged_exact=True),indent=2))
 records=[]
 for seed in seeds:
  data={sid:dict(np.load(C/f"seed{seed}_s{sid:02d}.npz")) for sid in train+val}
  for variant in variants:
   out=O/f"{variant}_seed{seed}";out.mkdir(exist_ok=True)
   if (out/"complete.json").exists():records.append(json.loads((out/"complete.json").read_text()));continue
   xx=[];yy=[]
   for sid in train:
    ff=data[sid]["logits"];features=transform(ff,variant);assert np.array_equal(features[:65],transform(ff[:65],variant));mask=data[sid]["y"]>=0;xx.append(features[mask]);yy.append(data[sid]["y"][mask])
   x=torch.tensor(np.concatenate(xx));y=torch.tensor(np.concatenate(yy),dtype=torch.long);seed_all(seed);m=ResidualLogitHead(x.shape[1]);initial_hash=hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in m.parameters())).hexdigest();m.mean.copy_(x.mean(0));m.scale.copy_(x.std(0).clamp(min=.1));m.cuda();x=x.cuda();y=y.cuda();validation={sid:(torch.from_numpy(transform(data[sid]["logits"],variant)).cuda(),data[sid]["y"]) for sid in val}
   def evaluate():return {sid:100*metrics(vy,hp(m,vx))["macro_f1"] for sid,(vx,vy) in validation.items()}
   base=evaluate();bn=float(np.mean(list(base.values())))
   for sid,(vx,vy) in validation.items():
    assert abs(hp(m,vx)-torch.from_numpy(data[sid]["logits"]).softmax(1).numpy()).max()<2e-5
   history=[dict(epoch=0,mean_f1=bn,people=base,eligible=True)];best=bn;bestepoch=0;beststate=snapshot(m);counts=torch.bincount(y,minlength=5).float();weights=(counts.sum()/counts.clamp(min=1)).sqrt();weights/=weights.mean();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001);gen=torch.Generator().manual_seed(seed);torch.manual_seed(100000+seed);torch.cuda.manual_seed_all(100000+seed);orders=[]
   for epoch in range(1,31):
    m.train();order=torch.randperm(len(y),generator=gen);orders.append(hashlib.sha256(order.numpy().tobytes()).hexdigest());losses=[];start=time.perf_counter()
    for idx in order.split(128):
     assert len(idx)>1;idx=idx.cuda();opt.zero_grad(set_to_none=True);loss=nn.functional.cross_entropy(m(x[idx]),y[idx],weight=weights);loss.backward();assert torch.isfinite(loss);opt.step();losses.append(float(loss.detach()))
    scores=evaluate();mean=float(np.mean(list(scores.values())));eligible=mean>=bn-1 and all(scores[s]>=base[s]-2 for s in val);history.append(dict(epoch=epoch,mean_f1=mean,people=scores,eligible=eligible,loss=float(np.mean(losses)),seconds=time.perf_counter()-start))
    if eligible and mean>best:best=mean;bestepoch=epoch;beststate=snapshot(m)
   parent=next(sp for sp in specs if sp["seed"]==seed);save(dict(state=snapshot(m),dim=x.shape[1],variant=variant,seed=seed,epoch=30,parent_sha256=parent["sha256"]),out/"final30.pt");save(dict(state=beststate,dim=x.shape[1],variant=variant,seed=seed,epoch=bestepoch,parent_sha256=parent["sha256"]),out/"model.pt");m.load_state_dict(beststate);replay=evaluate();assert abs(np.mean(list(replay.values()))-best)<1e-10
   for sid,(vx,vy) in validation.items():np.savez_compressed(out/f"s{sid:02d}.npz",p=hp(m,vx),y=vy)
   record=dict(seed=seed,variant=variant,dim=x.shape[1],parameters=sum(p.numel() for p in m.parameters()),initial_parameters_sha256=initial_hash,selected_epoch=bestepoch,selected_f1=best,parent_f1=bn,people=replay,history=history,order_hashes=orders,window_presentations=len(y)*30,checkpoint_sha256=sha(out/"model.pt"));(out/"complete.json").write_text(json.dumps(record,indent=2));records.append(record);del m,opt;torch.cuda.empty_cache();print("HEAD",seed,variant,bestepoch,best,flush=True)
  rr=[r for r in records if r["seed"]==seed];assert all(r["order_hashes"]==rr[0]["order_hashes"] for r in rr);repeat=next(r for r in rr if r["variant"]=="repeat11");hist=next(r for r in rr if r["variant"]=="history11");assert repeat["initial_parameters_sha256"]==hist["initial_parameters_sha256"] and repeat["parameters"]==hist["parameters"]
 (O/"selection.json").write_text(json.dumps(records,indent=2));(O/"audit.json").write_text(json.dumps(dict(status="PASS",models=9,parent_frozen_exact=True,causal_prefix_exact=True,initial_zero_residual_compatibility=True,same_training_orders=True,same_capacity_history_repeat=True),indent=2));print("COMPLETE",flush=True)
if __name__=="__main__":
 with sdpa_kernel(SDPBackend.MATH):run()
