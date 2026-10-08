from phase16_masked_pulses import *
from torch.utils.data import DataLoader,TensorDataset
OUT=ROOT/"phase18";OUT.mkdir(exist_ok=True);CACHE=OUT/"cache";CACHE.mkdir(exist_ok=True)
def prefix(model,x):
 batch,n,t,_=x.shape;z=model.patch_embed(x);z=torch.cat([model.cls_token.expand(batch,-1,-1),z],1);pos=model.pos_embed[:,[0,127,128]];spatial=pos[:,1:,:].unsqueeze(2).expand(batch,-1,t,-1).flatten(1,2);z=z+torch.cat([pos[:,:1,:].expand(batch,-1,-1),spatial],1);z[:,1:,:]+=model.time_embed[:,:t,:].unsqueeze(1).expand(batch,n,-1,-1).flatten(1,2);z=model.pos_drop(z)
 for block in model.blocks[:10]:z=block(z,rel_pos_bias=None)
 return z
class Tail(nn.Module):
 def __init__(self,encoder,carrier):
  super().__init__();self.blocks=copy.deepcopy(encoder.blocks[10:]);self.norm=copy.deepcopy(encoder.norm);self.head=nn.Linear(400,5);self.head.weight.data.copy_(torch.tensor(carrier["model"].coef_,dtype=torch.float32));self.head.bias.data.copy_(torch.tensor(carrier["model"].intercept_,dtype=torch.float32));self.register_buffer("mean",torch.tensor(carrier["scaler"].mean_,dtype=torch.float32));self.register_buffer("scale",torch.tensor(carrier["scaler"].scale_,dtype=torch.float32));self.requires_grad_(False)
  for module in self.modules():
   if isinstance(module,nn.LayerNorm):
    for p in module.parameters():p.requires_grad_(True)
  self.head.requires_grad_(True)
 def forward(self,x):
  z=x.flatten(0,1)
  for block in self.blocks:z=block(z,rel_pos_bias=None)
  z=self.norm(z)[:,1:,:].mean(1).reshape(-1,400);return self.head((z-self.mean)/self.scale)
def raw_subject(sid):
 root=ROOT/("data/processed" if sid<20 else ("phase2/processed" if sid<40 else "phase6/processed"));return np.load(root/f"s{sid:02d}_x.npy",mmap_mode="r"),np.load(root/f"s{sid:02d}_meta.npz")["y"]
def cache_subject(encoder,kind,sid):
 path=CACHE/f"{kind}_s{sid:02d}.npy";done=path.with_suffix(".json")
 if done.exists():return np.load(path,mmap_mode="r")
 raw,y=raw_subject(sid);dest=np.lib.format.open_memmap(path,mode="w+",dtype=np.float32,shape=(len(raw),2,31,200));start=time.perf_counter()
 with torch.inference_mode():
  for i in range(0,len(raw),32):
   x=preprocess(np.array(raw[i:i+32]));z=prefix(encoder,x);dest[i:i+32]=z.numpy().reshape(-1,2,31,200)
 dest.flush();done.write_text(json.dumps(dict(subject=sid,kind=kind,windows=len(raw),seconds=time.perf_counter()-start),indent=2));print("PREFIX CACHE",kind,sid,time.perf_counter()-start,flush=True);return np.load(path,mmap_mode="r")
def probabilities_tail(model,x):
 model.eval();out=[]
 with torch.inference_mode():
  for i in range(0,len(x),32):out.append(model(torch.tensor(np.array(x[i:i+32]))).softmax(1).numpy())
 return np.concatenate(out)
def run(source,kind):
 seed_all(17);out=OUT/f"{source}_{kind}";out.mkdir(exist_ok=True)
 if (out/"evaluation.json").exists():return
 encoder=make_encoder(kind=="pretrained");split=json.loads((ROOT/"phase3/protocol.json").read_text())["folds"][source]
 with (ROOT/"phase14/probes"/f"{source}_{kind}400/model.pkl").open("rb") as f:carrier=pickle.load(f)
 tail=Tail(encoder,carrier).eval();baseline=copy.deepcopy(tail);model_path=out/"model.pt"
 if not model_path.exists():
  xs=[];ys=[];ss=[]
  for sid in split["base_train"]:
   cx=cache_subject(encoder,kind,sid);_,y=raw_subject(sid);valid=y>=0;xs.append(np.array(cx[valid]));ys.append(y[valid]);ss.extend([sid]*int(valid.sum()))
  x=torch.from_numpy(np.concatenate(xs));y=torch.from_numpy(np.concatenate(ys));del xs,ys;ss=np.array(ss);counts=np.bincount(y.numpy(),minlength=5);classw=np.sqrt(len(y)/np.maximum(counts,1));ns={s:int(np.sum(ss==s)) for s in np.unique(ss)};weights=np.array([1/ns[s] for s in ss])*classw[y.numpy()];weights/=weights.mean();weights=torch.tensor(weights,dtype=torch.float32);val={sid:(cache_subject(encoder,kind,sid),raw_subject(sid)[1]) for sid in split["base_val"]}
  initial_score=float(np.mean([metrics(vy,probabilities_tail(tail,vx))["macro_f1"] for vx,vy in val.values()]));best=initial_score;bestepoch=0;beststate=copy.deepcopy(tail.state_dict());opt=torch.optim.AdamW([dict(params=[p for n,p in tail.named_parameters() if p.requires_grad and not n.startswith("head.")],lr=1e-5),dict(params=tail.head.parameters(),lr=1e-4)],weight_decay=.001);history=[];start=time.perf_counter();seed_all(17);start_epoch=0;resume_path=out/"training_resume.pt"
  if resume_path.exists():
   resume=torch.load(resume_path,map_location="cpu",weights_only=False);tail.load_state_dict(resume["state"]);opt.load_state_dict(resume["optimizer"]);best=resume["best"];bestepoch=resume["best_epoch"];beststate=resume["best_state"];history=resume["history"];start_epoch=resume["epoch"];torch.set_rng_state(resume["rng_state"]);start=time.perf_counter()-resume["elapsed_seconds"]
  for epoch in range(start_epoch,10):
   tail.eval()
   for idx in torch.randperm(len(y)).split(32):
    opt.zero_grad(set_to_none=True);loss=(nn.functional.cross_entropy(tail(x[idx]),y[idx],reduction="none")*weights[idx]).mean();loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in tail.parameters() if p.requires_grad],1.);assert torch.isfinite(loss) and torch.isfinite(norm);opt.step()
   score=float(np.mean([metrics(vy,probabilities_tail(tail,vx))["macro_f1"] for vx,vy in val.values()]));history.append(dict(epoch=epoch+1,val_f1=score,seconds=time.perf_counter()-start))
   if score>best:best=score;bestepoch=epoch+1;beststate=copy.deepcopy(tail.state_dict())
   tmp=out/"training_resume.tmp";torch.save(dict(state=tail.state_dict(),optimizer=opt.state_dict(),best=best,best_epoch=bestepoch,best_state=beststate,history=history,epoch=epoch+1,rng_state=torch.get_rng_state(),elapsed_seconds=time.perf_counter()-start),tmp);tmp.replace(resume_path)
   print("SOURCE TUNE",source,kind,epoch+1,score,flush=True)
  tail.load_state_dict(beststate);writable={n for n,p in tail.named_parameters() if p.requires_grad};assert all(torch.equal(v,tail.state_dict()[n]) for n,v in baseline.state_dict().items() if n not in writable);torch.save(dict(state=beststate,source=source,kind=kind,seed=17,history=history,initial_val_f1=initial_score,best_val_f1=best,best_epoch=bestepoch,writable_parameters=sum(p.numel() for p in tail.parameters() if p.requires_grad)),model_path);del x,y,val
 else:tail.load_state_dict(torch.load(model_path,map_location="cpu",weights_only=False)["state"])
 results=[]
 for sid in split["evaluation"]+list(range(40,59)):
  raw,y=raw_subject(sid);p0=[];p1=[];prediction_path=out/f"s{sid:02d}.npz";preds=None
  if prediction_path.exists() and prediction_path.stat().st_mtime>=model_path.stat().st_mtime:
   import zipfile
   try:
    with np.load(prediction_path) as saved:
     candidate={k:saved[k] for k in ["frozen","source_tuned"]}
     if np.array_equal(saved["y"],y) and all(v.shape==(len(y),5) and np.isfinite(v).all() for v in candidate.values()):preds=candidate
   except (OSError,ValueError,KeyError,EOFError,zipfile.BadZipFile):pass
  if preds is None:
   with torch.inference_mode():
    for i in range(0,len(raw),32):
     z=prefix(encoder,preprocess(np.array(raw[i:i+32]))).reshape(-1,2,31,200);p0.append(baseline(z).softmax(1).numpy());p1.append(tail(z).softmax(1).numpy())
   preds=dict(frozen=np.concatenate(p0),source_tuned=np.concatenate(p1));temporary=prediction_path.with_suffix(".tmp.npz");np.savez_compressed(temporary,y=y,**preds);temporary.replace(prediction_path)
  for method,pr in preds.items():results.append(dict(subject=sid,cohort="older" if sid>=40 else ("middle_dev" if sid>=20 else "young_dev"),method=method,**metrics(y,pr)))
 (out/"evaluation.json").write_text(json.dumps(dict(source=source,kind=kind,results=results,exploratory=True,scope="source-supervised last2-block LN andhead fine-tuning, not TTA"),indent=2));print("SOURCE TUNE COMPLETE",source,kind,flush=True)
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--source",required=True);parser.add_argument("--kind",required=True);a=parser.parse_args();run(a.source,a.kind)
