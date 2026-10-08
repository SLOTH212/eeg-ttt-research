from phase21_full_encoder import *
from phase22_cuda_graph import GraphAdapter
OUT22=ROOT/"phase22"

def source_checkpoint(source):
 protocol=json.loads((ROOT/"phase21/protocol.json").read_text());dirs=[ROOT/"phase21"/f"{source}_pretrained_{lr:g}" for lr in protocol["encoder_lrs"]["pretrained"]]
 assert all((d/"complete.json").exists() for d in dirs),"Source training incomplete"
 selected=max(dirs,key=lambda d:json.loads((d/"complete.json").read_text())["best_val_f1"])
 return selected/"model.pt"

def run_screen(source,objective,lr,limit=0):
 checkpoint=source_checkpoint(source);out=OUT22/(f"benchmark_{source}_{objective}_{lr:g}" if limit else f"{source}_{objective}_{lr:g}");out.mkdir(exist_ok=True)
 if (out/"complete.json").exists():return
 model=FullCarrier(source,"pretrained").eval().cuda();model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=False)["state"]);model.requires_grad_(False)
 for module in model.encoder.modules():
  if isinstance(module,nn.LayerNorm):
   for p in module.parameters():p.requires_grad_(True)
 writable={n for n,p in model.named_parameters() if p.requires_grad};params=[p for p in model.parameters() if p.requires_grad];initial=[p.detach().clone() for p in params];radii=[.1*max(float(v.square().mean().sqrt()),1)*v.numel()**.5 for v in initial];frozen={n:v.detach().clone() for n,v in model.state_dict().items() if n not in writable};results=[]
 for sid in ([5] if limit else [5,15,25,35]):
  done=out/f"s{sid:02d}.json"
  if done.exists():results.append(json.loads(done.read_text()));continue
  with torch.no_grad():
   for p,v in zip(params,initial):p.copy_(v)
  x,y=prepared(sid)
  if limit:x=x[:limit];y=y[:limit]
  base=probabilities(model,x);pred=[];ema=[];teacher=None;trace=[];start=time.perf_counter();graph=GraphAdapter(model,objective,lr,x[:1].cuda())
  for t in range(len(y)):
   teacher=base[t].copy() if teacher is None else .8*base[t]+.2*teacher;ema.append(teacher.copy());q=graph.step(x[t:t+1].cuda(),torch.as_tensor(teacher,device="cuda"));pred.append(q.detach().cpu().numpy()[0].copy());assert bool(graph.finite);loss=graph.loss
   if t%100==0:trace.append(dict(t=t,loss=float(loss.detach()),drift=float(torch.sqrt(sum((p-v).square().sum() for p,v in zip(params,initial))).detach())))
  del graph
  pred=np.array(pred);ema=np.array(ema);assert np.max(np.abs(pred[0]-base[0]))<.002;assert all(torch.equal(model.state_dict()[n],v) for n,v in frozen.items())
  result=dict(subject=sid,source=source,objective=objective,lr=lr,checkpoint=str(checkpoint),frozen=metrics(y,base),adapted=metrics(y,pred),weak_ema=metrics(y,ema),seconds=time.perf_counter()-start,trace=trace,predicted_counts=np.bincount(pred.argmax(1),minlength=5).tolist(),nonwritable_state_unchanged=True)
  temporary=out/f"s{sid:02d}.tmp.npz";np.savez_compressed(temporary,y=y,frozen=base,adapted=pred,weak_ema=ema);temporary.replace(out/f"s{sid:02d}.npz");done.write_text(json.dumps(result,indent=2));results.append(result);print("FULL CARRIER ADAPT",source,objective,lr,sid,result["adapted"]["macro_f1"],result["seconds"],flush=True)
 (out/"complete.json").write_text(json.dumps(dict(source=source,objective=objective,lr=lr,results=results,benchmark_limit=limit,source_validation_only=True,backend="cuda_graph"),indent=2))

if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--source",default="young");parser.add_argument("--objective",default="entropy",choices=["entropy","temporal"]);parser.add_argument("--lr",type=float,default=1e-5);parser.add_argument("--limit",type=int,default=0);a=parser.parse_args();run_screen(a.source,a.objective,a.lr,a.limit)
