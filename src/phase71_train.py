from common import *
from phase62_reconstruction import CBraMod,prep
from phase2_temporal import ema
import pickle,argparse,hashlib,time
CLASSIFIER_SEED=int(os.environ["CBRAMOD_CLASSIFIER_SEED"])
OUT66=ROOT/"phase71"/f"seed{CLASSIFIER_SEED}"
class Carrier(nn.Module):
 def __init__(self,kind):
  super().__init__();self.encoder=CBraMod();self.encoder.load_state_dict(torch.load(ROOT/"phase63"/(kind+"_encoder_state.pt"),map_location="cpu",weights_only=True))
  with (ROOT/"phase63"/("probe_"+kind)/"model.pkl").open("rb") as f:probe=pickle.load(f)
  self.head=nn.Linear(400,5)
  with torch.no_grad():
   self.head.weight.copy_(torch.tensor(probe["model"].coef_,dtype=torch.float32));self.head.bias.copy_(torch.tensor(probe["model"].intercept_,dtype=torch.float32))
  self.register_buffer("mean",torch.tensor(probe["scaler"].mean_,dtype=torch.float32));self.register_buffer("scale",torch.tensor(probe["scaler"].scale_,dtype=torch.float32))
  self.encoder.proj_out.requires_grad_(False)
 def forward(self,x):
  z=self.encoder.encoder(self.encoder.patch_embedding(x)).mean(2).flatten(1)
  return self.head((z-self.mean)/self.scale)
def prepared(sid):
 folder=ROOT/("data/processed" if sid<20 else "phase2/processed");raw=np.load(folder/f"s{sid:02d}_x.npy",mmap_mode="r");parts=[]
 for i in range(0,len(raw),128):parts.append(prep(np.array(raw[i:i+128]),[.1,45]))
 return torch.from_numpy(np.concatenate(parts)),np.load(folder/f"s{sid:02d}_meta.npz")["y"]
def probabilities(model,x):
 model.eval();result=[]
 with torch.inference_mode():
  for batch in x.split(128):result.append(model(batch.cuda()).softmax(1).cpu().numpy())
 return np.concatenate(result)
def snapshot(m):return {n:v.detach().cpu().clone() for n,v in m.state_dict().items()}
def save(d,path):
 temp=path.with_suffix(".tmp");torch.save(d,temp);temp.replace(path)
def run(kind,lr):
 assert os.sched_getaffinity(0)<=set(range(8,24));pr=json.loads((OUT66/"protocol.json").read_text());assert lr in pr["encoder_lrs"][kind]
 out=OUT66/f"{kind}_{lr:g}";out.mkdir(exist_ok=True)
 if (out/"complete.json").exists():return
 model=Carrier(kind).eval();val={s:prepared(s) for s in pr["validation_subjects"]};first=val[5][0][:8]
 with torch.no_grad():cpu=model(first).softmax(1).numpy()
 model.cuda()
 with torch.no_grad():gpu=model(first.cuda()).softmax(1).cpu().numpy()
 cpuerror=float(abs(cpu-gpu).max());assert cpuerror<2e-5 and np.array_equal(cpu.argmax(1),gpu.argmax(1))
 projection={n:v.detach().cpu().clone() for n,v in model.encoder.proj_out.state_dict().items()}
 def evaluate():return {s:metrics(y,probabilities(model,x))["macro_f1"] for s,(x,y) in val.items()}
 initial_predictions={s:probabilities(model,x) for s,(x,y) in val.items()};initial_scores={}
 for sid,p in initial_predictions.items():
  ref=np.load(ROOT/"phase63"/("probe_"+kind)/f"s{sid:02d}.npz");assert abs(p-ref["p"]).max()<2e-5 and np.array_equal(p.argmax(1),ref["p"].argmax(1));initial_scores[sid]=metrics(val[sid][1],p)["macro_f1"]
 initial=float(np.mean(list(initial_scores.values())));best=initial;best_epoch=0;best_state=snapshot(model);history=[];start_epoch=0
 xx=[];yy=[];ss=[]
 for sid in pr["training_subjects"]:
  x,y=prepared(sid);valid=y>=0;xx.append(x[valid]);yy.append(y[valid]);ss.extend([sid]*int(valid.sum()))
 x=torch.cat(xx);y=torch.tensor(np.concatenate(yy));ss=np.array(ss);del xx,yy
 cw=np.sqrt(len(y)/np.maximum(np.bincount(y.numpy(),minlength=5),1));counts={s:int((ss==s).sum()) for s in np.unique(ss)};w=np.array([1/counts[s] for s in ss])*cw[y.numpy()];w=torch.tensor(w/w.mean(),dtype=torch.float32)
 opt=torch.optim.AdamW([dict(params=[p for p in model.encoder.parameters() if p.requires_grad],lr=lr),dict(params=model.head.parameters(),lr=1e-4)],weight_decay=.01)
 seed_all(CLASSIFIER_SEED);order_rng=torch.Generator(device="cpu").manual_seed(CLASSIFIER_SEED);orders=[];resume=out/"resume.pt"
 if resume.exists():
  ck=torch.load(resume,map_location="cpu",weights_only=False);model.load_state_dict(ck["state"]);opt.load_state_dict(ck["optimizer"]);best=ck["best"];best_epoch=ck["best_epoch"];best_state=ck["best_state"];history=ck["history"];start_epoch=ck["epoch"];torch.set_rng_state(ck["rng"]);torch.cuda.set_rng_state(ck["cuda_rng"]);order_rng.set_state(ck["order_rng"]);orders=ck["orders"]
 for epoch in range(start_epoch,10):
  model.train();start=time.perf_counter();losses=[];permutation=torch.randperm(len(y),generator=order_rng);orders.append(permutation.tolist())
  for idx in permutation.split(64):
   opt.zero_grad(set_to_none=True);loss=(nn.functional.cross_entropy(model(x[idx].cuda()),y[idx].cuda(),reduction="none")*w[idx].cuda()).mean();loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(loss) and torch.isfinite(norm);opt.step();losses.append(float(loss.detach()))
  scores=evaluate();score=float(np.mean(list(scores.values())));history.append(dict(epoch=epoch+1,val_f1=score,participants=scores,loss=float(np.mean(losses)),seconds=time.perf_counter()-start))
  if score>best:best=score;best_epoch=epoch+1;best_state=snapshot(model)
  assert all(torch.equal(v.cpu(),projection[n]) for n,v in model.encoder.proj_out.state_dict().items())
  save(dict(state=snapshot(model),optimizer=opt.state_dict(),best=best,best_epoch=best_epoch,best_state=best_state,history=history,epoch=epoch+1,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),order_rng=order_rng.get_state(),orders=orders),resume)
  print("CBRAMOD FULL SOURCE",kind,lr,history[-1],flush=True)
 model.load_state_dict(best_state);save(dict(state=best_state,classifier_seed=CLASSIFIER_SEED,kind=kind,lr=lr,best_epoch=best_epoch,best_val_f1=best,initial_val_f1=initial,history=history),out/"model.pt");results=[]
 for sid,(vx,vy) in val.items():
  p=probabilities(model,vx);np.savez_compressed(out/f"s{sid:02d}.npz",y=vy,p=p);results.append(dict(subject=sid,metrics=metrics(vy,p)))
 (out/"training_orders.json").write_text(json.dumps(dict(seed=CLASSIFIER_SEED,orders=orders)))
 (out/"complete.json").write_text(json.dumps(dict(kind=kind,classifier_seed=CLASSIFIER_SEED,lr=lr,best_epoch=best_epoch,best_val_f1=best,initial_val_f1=initial,history=history,results=results,training_valid_windows=len(y),training_subjects=pr["training_subjects"],source_validation_only=True,cpu_gpu_probability_error=cpuerror,initial_probe_argmax_exact=True,reconstruction_projection_unchanged=True),indent=2))
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--kind",required=True);p.add_argument("--lr",type=float,required=True);a=p.parse_args();run(a.kind,a.lr)
