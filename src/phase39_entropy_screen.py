from phase30_masked_graph import *
from phase2_temporal import ema
import hashlib
OUT=ROOT/"phase39"
def load(source,variant):
 folder=ROOT/("phase35/"+source if variant=="fitted" else "phase28/"+source+"_joint_0")
 model=JointCarrier(source).cuda().eval();model.load_state_dict(torch.load(folder/"model.pt",map_location="cpu",weights_only=False)["state"]);model.requires_grad_(False)
 for module in model.encoder.modules():
  if isinstance(module,nn.LayerNorm):
   for p in module.parameters():p.requires_grad_(True)
 return model,hashlib.sha256((folder/"model.pt").read_bytes()).hexdigest()
def run(source,variant,lr):
 assert json.loads((ROOT/"phase37/graph_audit.json").read_text())["status"]=="PASS"
 out=OUT/f"{source}_{variant}_{lr:g}";out.mkdir(exist_ok=True)
 if (out/"complete.json").exists():return
 model,sha=load(source,variant);x0=np.load(ROOT/"phase17/cache/s05_x.npy",mmap_mode="r");graph=GraphAdapter(model,"entropy",lr,torch.tensor(np.array(x0[:1]),device="cuda"));writable={n for n,p in model.named_parameters() if p.requires_grad};frozen={n:v.detach().clone() for n,v in model.state_dict().items() if n not in writable};results=[]
 (out/"selection.json").write_text(json.dumps(dict(source=source,variant=variant,lr=lr,checkpoint_sha256=sha,source_validation_only=True),indent=2))
 for sid in [5,15,25,35]:
  file=out/f"s{sid:02d}.npz";done=out/f"s{sid:02d}.json"
  if done.exists():results.append(json.loads(done.read_text()));continue
  cache=np.load(ROOT/"phase28"/f"{source}_joint_0"/f"s{sid:02d}.npz");y=cache["y"];base=cache["p"];x=np.load(ROOT/"phase17/cache"/f"s{sid:02d}_x.npy",mmap_mode="r");tokens=np.load(ROOT/"phase28/tokens"/f"s{sid:02d}.npz")["tokens"];assert len(x)==len(y)==len(tokens);graph.reset();pred=[];losses=[];norms=[];start=time.perf_counter()
  for t in range(len(y)):
   q=graph.step(torch.tensor(np.array(x[t:t+1]),device="cuda"),torch.ones(5,device="cuda")/5);pred.append(q.detach().cpu().numpy()[0].copy());assert bool(graph.finite);losses.append(float(graph.loss.detach()));norms.append(float(graph.norm.detach()))
  pred=np.array(pred);firsterr=float(abs(pred[0]-base[0]).max());assert firsterr<2e-5;assert all(torch.equal(model.state_dict()[n],v) for n,v in frozen.items());assert all(float((p-v).norm())<=radius+1e-5 for p,v,radius in zip(graph.params,graph.initial,graph.radii));np.savez_compressed(file,y=y,p=pred,loss=np.array(losses),gradient_norm=np.array(norms));rec=dict(source=source,variant=variant,lr=lr,subject=sid,metrics=metrics(y,pred),frozen=metrics(y,base),weak_ema=metrics(y,ema(base,.8)),strong_ema=metrics(y,ema(base,.5)),adapted_ema=metrics(y,ema(pred,.8)),first_probability_error=firsterr,nonwritable_unchanged=True,radius_verified=True,seconds=time.perf_counter()-start);done.write_text(json.dumps(rec,indent=2));results.append(rec);print("ENTROPY SOURCE",source,variant,lr,sid,rec["metrics"]["macro_f1"],flush=True)
 (out/"complete.json").write_text(json.dumps(dict(source=source,variant=variant,lr=lr,mean_f1=float(np.mean([r["metrics"]["macro_f1"] for r in results])),source_validation_only=True,people=results),indent=2))
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--lr",type=float,required=True);a=parser.parse_args();run("mixed","unfitted",a.lr)
