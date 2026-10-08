from phase30_masked_graph import *
from phase2_temporal import ema
def run(weight,objective,lr):
 assert json.loads((OUT30/"graph_audit.json").read_text())["status"]=="PASS";out=OUT30/f"young_w{weight:g}_{objective}_{lr:g}";out.mkdir(exist_ok=True)
 if (out/"complete.json").exists():return
 model=load_model(weight);x0=np.load(ROOT/"phase17/cache/s05_x.npy",mmap_mode="r");example=torch.tensor(np.array(x0[:1]),device="cuda");graph=MaskedGraphAdapter(model,lr,example) if objective=="masked_token" else GraphAdapter(model,"entropy",lr,example);frozen={n:v.detach().clone() for n,v in model.state_dict().items() if not dict(model.named_parameters()).get(n,torch.tensor(0.)).requires_grad};results=[]
 for sid in [5,15,25,35]:
  file=out/f"s{sid:02d}.npz";done=out/f"s{sid:02d}.json"
  if done.exists():results.append(json.loads(done.read_text()));continue
  cache=np.load(ROOT/"phase28"/f"young_joint_{weight:g}"/f"s{sid:02d}.npz");y=cache["y"];base=cache["p"];x=np.load(ROOT/"phase17/cache"/f"s{sid:02d}_x.npy",mmap_mode="r");tokens=np.load(ROOT/"phase28/tokens"/f"s{sid:02d}.npz")["tokens"];assert x.shape==(len(y),2,2,15,200) and tokens.shape==(len(y),2,30);graph.reset();pred=[];losses=[];norms=[];start=time.perf_counter()
  for t in range(len(y)):
   xx=torch.tensor(np.array(x[t:t+1]),device="cuda")
   if objective=="masked_token":q=graph.step(xx,torch.tensor(tokens[t:t+1].astype(np.int64),device="cuda"))
   else:q=graph.step(xx,torch.ones(5,device="cuda")/5)
   pred.append(q.detach().cpu().numpy()[0].copy());assert bool(graph.finite);losses.append(float(graph.loss.detach()));norms.append(float(graph.norm.detach()))
  pred=np.array(pred);firsterr=float(abs(pred[0]-base[0]).max());assert firsterr<2e-5;assert all(torch.equal(model.state_dict()[n],v) for n,v in frozen.items());distances=[float((p-v).norm()) for p,v in zip(graph.params,graph.initial)];assert all(d<=radius+1e-5 for d,radius in zip(distances,graph.radii));np.savez_compressed(file,y=y,p=pred,loss=np.array(losses),gradient_norm=np.array(norms));record=dict(source="young",weight=weight,objective=objective,lr=lr,subject=sid,metrics=metrics(y,pred),frozen=metrics(y,base),weak_ema=metrics(y,ema(base,.8)),strong_ema=metrics(y,ema(base,.5)),first_probability_error=firsterr,nonwritable_unchanged=True,radius_verified=True,seconds=time.perf_counter()-start,source_validation_only=True);done.write_text(json.dumps(record,indent=2));results.append(record);print("JOINT ONLINE SOURCE",weight,objective,lr,sid,record["metrics"]["macro_f1"],flush=True)
 (out/"complete.json").write_text(json.dumps(dict(weight=weight,objective=objective,lr=lr,mean_f1=float(np.mean([x["metrics"]["macro_f1"] for x in results])),source_validation_only=True,people=results),indent=2))
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--weight",type=float,required=True);parser.add_argument("--objective",required=True);parser.add_argument("--lr",type=float,required=True);a=parser.parse_args();run(a.weight,a.objective,a.lr)
