from phase30_masked_graph import *
from phase2_temporal import ema
import hashlib
OUT=ROOT/"phase37"
def load(source,variant):
 folder=ROOT/("phase35/"+source if variant=="fitted" else "phase28/"+source+"_joint_0")
 model=JointCarrier(source).cuda().eval();model.load_state_dict(torch.load(folder/"model.pt",map_location="cpu",weights_only=False)["state"]);model.requires_grad_(False)
 for module in model.encoder.modules():
  if isinstance(module,nn.LayerNorm):
   for p in module.parameters():p.requires_grad_(True)
 return model,hashlib.sha256((folder/"model.pt").read_bytes()).hexdigest()
def audit():
 x,_=prepared(5);x=x[1280:1344].cuda();tokens=torch.tensor(np.load(ROOT/"phase28/tokens/s05.npz")["tokens"][1280:1344].astype(np.int64),device="cuda");rows=[]
 for source in ["young","mixed"]:
  for variant in ["unfitted","fitted"]:
   model,sha=load(source,variant);params=[p for p in model.parameters() if p.requires_grad];initial=[p.detach().clone() for p in params];radii=[.1*max(float(v.square().mean().sqrt()),1)*v.numel()**.5 for v in initial];opt=torch.optim.SGD(params,lr=.01);pred=[]
   for t in range(len(x)):
    with torch.no_grad():pred.append(model(x[t:t+1]).softmax(1).cpu().numpy()[0])
    opt.zero_grad(set_to_none=True);loss,_=model.auxiliary(x[t:t+1],tokens[t:t+1],False);loss.backward();nn.utils.clip_grad_norm_(params,1.);opt.step()
    with torch.no_grad():
     for p,v,r in zip(params,initial,radii):
      d=p-v;p.copy_(v+d*torch.clamp(r/torch.linalg.vector_norm(d).clamp_min(1e-12),max=1.))
   reference=np.array(pred);end=[p.detach().clone() for p in params];del model,opt
   model,_=load(source,variant);graph=MaskedGraphAdapter(model,.01,x[:1]);pred=[]
   for t in range(len(x)):pred.append(graph.step(x[t:t+1],tokens[t:t+1]).detach().cpu().numpy()[0].copy())
   err=float(abs(np.array(pred)-reference).max());perr=max(float(abs(p-v).max()) for p,v in zip(graph.params,end));dis=int((np.array(pred).argmax(1)!=reference.argmax(1)).sum());assert err<2e-5 and perr<2e-6 and dis==0
   rows.append(dict(source=source,variant=variant,windows=64,probability_error=err,parameter_error=perr,argmax_disagreements=dis,checkpoint_sha256=sha));del graph,model
 (OUT/"graph_audit.json").write_text(json.dumps(dict(status="PASS",cases=rows),indent=2));print("GRAPH AUDIT PASS",flush=True)
def run(source,variant,lr):
 assert json.loads((OUT/"graph_audit.json").read_text())["status"]=="PASS"
 out=OUT/f"{source}_{variant}_{lr:g}";out.mkdir(exist_ok=True)
 if (out/"complete.json").exists():return
 model,sha=load(source,variant);x0=np.load(ROOT/"phase17/cache/s05_x.npy",mmap_mode="r");graph=MaskedGraphAdapter(model,lr,torch.tensor(np.array(x0[:1]),device="cuda"));writable={n for n,p in model.named_parameters() if p.requires_grad};frozen={n:v.detach().clone() for n,v in model.state_dict().items() if n not in writable};results=[]
 (out/"selection.json").write_text(json.dumps(dict(source=source,variant=variant,lr=lr,checkpoint_sha256=sha,source_validation_only=True),indent=2))
 for sid in [5,15,25,35]:
  file=out/f"s{sid:02d}.npz";done=out/f"s{sid:02d}.json"
  if done.exists():results.append(json.loads(done.read_text()));continue
  cache=np.load(ROOT/"phase28"/f"{source}_joint_0"/f"s{sid:02d}.npz");y=cache["y"];base=cache["p"];x=np.load(ROOT/"phase17/cache"/f"s{sid:02d}_x.npy",mmap_mode="r");tokens=np.load(ROOT/"phase28/tokens"/f"s{sid:02d}.npz")["tokens"];assert len(x)==len(y)==len(tokens);graph.reset();pred=[];losses=[];norms=[];start=time.perf_counter()
  for t in range(len(y)):
   q=graph.step(torch.tensor(np.array(x[t:t+1]),device="cuda"),torch.tensor(tokens[t:t+1].astype(np.int64),device="cuda"));pred.append(q.detach().cpu().numpy()[0].copy());assert bool(graph.finite);losses.append(float(graph.loss.detach()));norms.append(float(graph.norm.detach()))
  pred=np.array(pred);firsterr=float(abs(pred[0]-base[0]).max());assert firsterr<2e-5;assert all(torch.equal(model.state_dict()[n],v) for n,v in frozen.items());assert all(float((p-v).norm())<=radius+1e-5 for p,v,radius in zip(graph.params,graph.initial,graph.radii));np.savez_compressed(file,y=y,p=pred,loss=np.array(losses),gradient_norm=np.array(norms));rec=dict(source=source,variant=variant,lr=lr,subject=sid,metrics=metrics(y,pred),frozen=metrics(y,base),weak_ema=metrics(y,ema(base,.8)),strong_ema=metrics(y,ema(base,.5)),adapted_ema=metrics(y,ema(pred,.8)),first_probability_error=firsterr,nonwritable_unchanged=True,radius_verified=True,seconds=time.perf_counter()-start);done.write_text(json.dumps(rec,indent=2));results.append(rec);print("HEAD ONLY ONLINE",source,variant,lr,sid,rec["metrics"]["macro_f1"],flush=True)
 (out/"complete.json").write_text(json.dumps(dict(source=source,variant=variant,lr=lr,mean_f1=float(np.mean([r["metrics"]["macro_f1"] for r in results])),source_validation_only=True,people=results),indent=2))
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--audit",action="store_true");parser.add_argument("--source");parser.add_argument("--variant");parser.add_argument("--lr",type=float);a=parser.parse_args()
 if a.audit:audit()
 else:run(a.source,a.variant,a.lr)
