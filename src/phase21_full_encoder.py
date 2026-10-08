from phase16_masked_pulses import *
# common remains CPU-only for historical experiments; opt in before CUDA initialization.
os.environ["CUDA_VISIBLE_DEVICES"]="0"
OUT=ROOT/"phase21";OUT.mkdir(exist_ok=True)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False

def subject(sid):
 d=ROOT/("data/processed" if sid<20 else "phase2/processed" if sid<40 else "phase6/processed")
 return np.load(d/f"s{sid:02d}_x.npy",mmap_mode="r"),np.load(d/f"s{sid:02d}_meta.npz")["y"]

class FullCarrier(nn.Module):
 def __init__(self,source,kind):
  super().__init__();self.encoder=make_encoder(kind=="pretrained")
  with (ROOT/"phase14/probes"/f"{source}_{kind}400/model.pkl").open("rb") as f:c=pickle.load(f)
  self.head=nn.Linear(400,5)
  with torch.no_grad():
   self.head.weight.copy_(torch.tensor(c["model"].coef_,dtype=torch.float32));self.head.bias.copy_(torch.tensor(c["model"].intercept_,dtype=torch.float32))
  self.register_buffer("mean",torch.tensor(c["scaler"].mean_,dtype=torch.float32));self.register_buffer("scale",torch.tensor(c["scaler"].scale_,dtype=torch.float32))
  self.requires_grad_(True)
 def forward(self,x):
  z=self.encoder.forward_features(x.flatten(0,1),input_chans=[0,127,128],return_patch_tokens=True).mean(1).reshape(-1,400)
  return self.head((z-self.mean)/self.scale)

def prepared(sid):
 raw,y=subject(sid);xs=[]
 for i in range(0,len(raw),128):xs.append(preprocess(np.array(raw[i:i+128])).reshape(-1,2,2,15,200))
 return torch.cat(xs),y

def probabilities(model,x):
 model.eval();out=[]
 with torch.inference_mode():
  for b in x.split(128):out.append(model(b.to("cuda")).softmax(1).cpu().numpy())
 return np.concatenate(out)

def snapshot(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def atomic_save(d,path):
 tmp=path.with_suffix(".tmp");torch.save(d,tmp);tmp.replace(path)

def run(source,kind,lr,epochs=10,benchmark=False):
 assert torch.cuda.is_available();assert os.sched_getaffinity(0)<=set(range(8,24))
 out=OUT/f"{source}_{kind}_{lr:g}";out.mkdir(exist_ok=True)
 if (out/"complete.json").exists() and not benchmark:return
 model=FullCarrier(source,kind).eval();x,y=prepared(5)
 with torch.inference_mode():cpu=model(x[:4]).softmax(1).numpy()
 model.to("cuda")
 with torch.inference_mode():gpu=model(x[:4].cuda()).softmax(1).cpu().numpy()
 err=float(np.max(np.abs(cpu-gpu)));assert err<.002;assert np.array_equal(cpu.argmax(1),gpu.argmax(1))
 if benchmark:
  opt=torch.optim.AdamW(model.parameters(),lr=lr);b=x[:64].cuda();target=torch.tensor(y[:64].clip(0),device="cuda");start=time.perf_counter()
  for _ in range(12):
   opt.zero_grad(set_to_none=True);loss=nn.functional.cross_entropy(model(b),target);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
  torch.cuda.synchronize();result=dict(seconds_12_steps=time.perf_counter()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),cpu_gpu_max_probability_error=err,parameters=sum(p.numel() for p in model.parameters()),device=torch.cuda.get_device_name(0));(OUT/"benchmark.json").write_text(json.dumps(result,indent=2));print(result,flush=True);return
 del x,y
 split=json.loads((ROOT/"phase3/protocol.json").read_text())["folds"][source]
 val={sid:prepared(sid) for sid in split["base_val"]};xs=[];ys=[];ss=[]
 for sid in split["base_train"]:
  x,y=prepared(sid);valid=y>=0;xs.append(x[valid]);ys.append(y[valid]);ss.extend([sid]*int(valid.sum()))
 x=torch.cat(xs);y=torch.tensor(np.concatenate(ys));del xs,ys;ss=np.array(ss)
 cw=np.sqrt(len(y)/np.maximum(np.bincount(y.numpy(),minlength=5),1));counts={s:int((ss==s).sum()) for s in np.unique(ss)};w=np.array([1/counts[s] for s in ss])*cw[y.numpy()];w=torch.tensor(w/w.mean(),dtype=torch.float32)
 opt=torch.optim.AdamW([dict(params=model.encoder.parameters(),lr=lr),dict(params=model.head.parameters(),lr=1e-4)],weight_decay=.01)
 initial=float(np.mean([metrics(vy,probabilities(model,vx))["macro_f1"] for vx,vy in val.values()]));best=initial;best_epoch=0;best_state=snapshot(model);history=[];start_epoch=0;seed_all(17)
 resume=out/"resume.pt"
 if resume.exists():
  ck=torch.load(resume,map_location="cpu",weights_only=False);model.load_state_dict(ck["state"]);opt.load_state_dict(ck["optimizer"]);best=ck["best"];best_epoch=ck["best_epoch"];best_state=ck["best_state"];history=ck["history"];start_epoch=ck["epoch"];torch.set_rng_state(ck["rng"]);torch.cuda.set_rng_state(ck["cuda_rng"])
 for epoch in range(start_epoch,epochs):
  model.train();start=time.perf_counter();losses=[]
  for idx in torch.randperm(len(y)).split(64):
   opt.zero_grad(set_to_none=True);loss=(nn.functional.cross_entropy(model(x[idx].cuda()),y[idx].cuda(),reduction="none")*w[idx].cuda()).mean();loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(loss) and torch.isfinite(norm);opt.step();losses.append(float(loss.detach()))
  per={sid:metrics(vy,probabilities(model,vx))["macro_f1"] for sid,(vx,vy) in val.items()};score=float(np.mean(list(per.values())))
  history.append(dict(epoch=epoch+1,val_f1=score,participants=per,loss=float(np.mean(losses)),seconds=time.perf_counter()-start))
  if score>best:best=score;best_epoch=epoch+1;best_state=snapshot(model)
  atomic_save(dict(state=snapshot(model),optimizer=opt.state_dict(),best=best,best_epoch=best_epoch,best_state=best_state,history=history,epoch=epoch+1,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),resume)
  print("FULL ENCODER",source,kind,lr,history[-1],flush=True)
 atomic_save(dict(state=best_state,source=source,kind=kind,lr=lr,best_epoch=best_epoch,best_val_f1=best,initial_val_f1=initial,history=history),out/"model.pt")
 (out/"complete.json").write_text(json.dumps(dict(source=source,kind=kind,lr=lr,best_epoch=best_epoch,best_val_f1=best,initial_val_f1=initial,history=history,cpu_gpu_max_probability_error=err,source_validation_only=True),indent=2))
 print("FULL ENCODER COMPLETE",source,kind,lr,best,flush=True)

if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--source",default="young");parser.add_argument("--kind",default="pretrained");parser.add_argument("--lr",type=float,default=2e-5);parser.add_argument("--benchmark",action="store_true");a=parser.parse_args();run(a.source,a.kind,a.lr,benchmark=a.benchmark)
