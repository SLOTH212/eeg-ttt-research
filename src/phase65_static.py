from phase30_masked_graph import *
from phase37_screen import load as load_carrier
STATIC_SEED=int(os.environ["STATIC_SEED"])
OUT25=ROOT/"phase65"/f"seed{STATIC_SEED}"
def run(source):
 which=os.environ["META_LABEL_TARGET"];out=OUT25/("static_"+which);out.mkdir(exist_ok=True)
 if (out/"complete.json").exists():return
 checkpoint=ROOT/"phase28/mixed_joint_0/model.pt";model=load_carrier(source,"unfitted")[0];model.requires_grad_(False)
 for module in model.encoder.modules():
  if isinstance(module,nn.LayerNorm):
   for p in module.parameters():p.requires_grad_(True)
 params=[p for p in model.parameters() if p.requires_grad];initial=[p.detach().clone() for p in params];radii=[.1*max(float(v.square().mean().sqrt()),1)*v.numel()**.5 for v in initial];writable={n for n,p in model.named_parameters() if p.requires_grad};frozen={n:v.detach().cpu().clone() for n,v in model.state_dict().items() if n not in writable};xs=[];ys=[];ss=[];labels={}
 for sid in [3,4,12,13,14]:
  indices=np.array(json.loads((ROOT/"phase52"/which/"complete.json").read_text())["labels_used"][str(sid)],dtype=int);x,y=prepared(sid);assert np.all(y[indices]>=0);xs.append(x[indices]);ys.append(y[indices]);ss.extend([sid]*len(indices));labels[str(sid)]=indices.tolist()
 x=torch.cat(xs);y=torch.tensor(np.concatenate(ys),dtype=torch.long);ss=np.array(ss);del xs,ys;cw=np.sqrt(len(y)/np.maximum(np.bincount(y.numpy(),minlength=5),1));counts={s:int((ss==s).sum()) for s in np.unique(ss)};weights=np.array([1/counts[s] for s in ss])*cw[y.numpy()];weights=torch.tensor(weights/weights.mean(),dtype=torch.float32);val={}
 for sid in [5,15,25,35]:
  xx=torch.from_numpy(np.array(np.load(ROOT/"phase17/cache"/f"s{sid:02d}_x.npy",mmap_mode="r")));root=ROOT/("data/processed" if sid<20 else "phase2/processed");val[sid]=(xx,np.load(root/f"s{sid:02d}_meta.npz")["y"])
 def evaluate():return {sid:metrics(vy,probabilities(model,vx))["macro_f1"] for sid,(vx,vy) in val.items()}
 base=evaluate();best=float(np.mean(list(base.values())));initial_score=best;best_epoch=0;best_state=snapshot(model);history=[];seed_all(STATIC_SEED);order_rng=torch.Generator(device="cpu").manual_seed(STATIC_SEED);orders=[];opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=.001);start=time.perf_counter()
 for epoch in range(20):
  model.eval()
  permutation=torch.randperm(len(y),generator=order_rng);orders.append(permutation.tolist());(out/"training_orders.json").write_text(json.dumps(dict(seed=STATIC_SEED,orders=orders)))
  for idx in permutation.split(64):
   opt.zero_grad(set_to_none=True);loss=(nn.functional.cross_entropy(model(x[idx].cuda()),y[idx].cuda(),reduction="none")*weights[idx].cuda()).mean();loss.backward();norm=nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(loss) and torch.isfinite(norm);opt.step()
   with torch.no_grad():
    for p,v,radius in zip(params,initial,radii):
     delta=p-v;p.copy_(v+delta*torch.clamp(radius/torch.linalg.vector_norm(delta).clamp_min(1e-12),max=1.))
  scores=evaluate();score=float(np.mean(list(scores.values())));history.append(dict(epoch=epoch+1,val_f1=score,participants=scores))
  if score>best:best=score;best_epoch=epoch+1;best_state=snapshot(model)
  print("MATCHED STATIC",source,epoch+1,score,flush=True)
 model.load_state_dict(best_state);assert all(torch.equal(model.state_dict()[n].cpu(),v) for n,v in frozen.items());assert all(float(torch.linalg.vector_norm(p-v))<=radius+1e-5 for p,v,radius in zip(params,initial,radii));atomic_save(dict(state=best_state,calibration_training_seed=STATIC_SEED,source=source,source_checkpoint=str(checkpoint),labels_used=labels,scope="LayerNorm affine",best_epoch=best_epoch,best_val_f1=best,initial_val_f1=initial_score,history=history),out/"model.pt");results=[]
 for sid,(vx,vy) in val.items():
  pr=probabilities(model,vx);np.savez_compressed(out/f"s{sid:02d}.npz",y=vy,predictions=pr);results.append(dict(subject=sid,metrics=metrics(vy,pr),frozen_f1=base[sid]))
 (out/"complete.json").write_text(json.dumps(dict(source=source,calibration_training_seed=STATIC_SEED,policy="static_matched",results=results,best_epoch=best_epoch,best_val_f1=best,initial_val_f1=initial_score,labels_used=labels,label_count=sum(map(len,labels.values())),nonwritable_state_unchanged=True,radius_checked=True,history=history,seconds=time.perf_counter()-start,source_validation_only=True),indent=2));print("MATCHED STATIC COMPLETE",source,best,flush=True)
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--source",required=True);a=parser.parse_args();run(a.source)
