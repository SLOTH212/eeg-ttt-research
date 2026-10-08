from phase22_cuda_graph import GraphAdapter
from phase22_source_screen import *
from phase2_temporal import ema
# phase2_temporal imports common already cached; GPU opt-in remains active.
OUT23=ROOT/"phase23"

def evaluate(source,objective,sid):
 lock=json.loads((OUT23/"selection_lock.json").read_text());spec=next(s for s in lock["selections"] if s["source"]==source and s["objective"]==objective);lr=spec["lr"];out=OUT23/f"{source}_{objective}";out.mkdir(exist_ok=True);done=out/f"s{sid:02d}.json"
 if done.exists():return
 frozen_path=ROOT/"phase21/evaluation"/f"{source}_pretrained"/f"s{sid:02d}.npz"
 with np.load(frozen_path) as saved:y=saved["y"];base=saved["source_tuned"]
 teacher=ema(base,.8);strong=ema(base,.5);trace=[];start=time.perf_counter();unchanged=True
 if lr==0:pred=base.copy()
 else:
  model=FullCarrier(source,"pretrained").eval().cuda();checkpoint=ROOT/lock["checkpoints"][source]["path"];assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==lock["checkpoints"][source]["sha256"];model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=False)["state"]);model.requires_grad_(False)
  for module in model.encoder.modules():
   if isinstance(module,nn.LayerNorm):
    for p in module.parameters():p.requires_grad_(True)
  writable={n for n,p in model.named_parameters() if p.requires_grad};frozen={n:v.detach().clone() for n,v in model.state_dict().items() if n not in writable};x,yraw=prepared(sid);assert np.array_equal(y,yraw);graph=GraphAdapter(model,objective,lr,x[:1].cuda());pred=[]
  for t in range(len(y)):
   q=graph.step(x[t:t+1].cuda(),torch.as_tensor(teacher[t],device="cuda"));pred.append(q.detach().cpu().numpy()[0].copy());assert bool(graph.finite)
   if t%100==0:trace.append(dict(t=t,loss=float(graph.loss.detach()),drift=float(torch.sqrt(sum((p-v).square().sum() for p,v in zip(graph.params,graph.initial))).detach())))
  pred=np.array(pred);assert np.max(np.abs(pred[0]-base[0]))<.002;unchanged=all(torch.equal(model.state_dict()[n],v) for n,v in frozen.items());assert unchanged
 mask=np.zeros(len(y),bool)
 for b in np.flatnonzero((y[1:]!=y[:-1])&(y[1:]>=0)&(y[:-1]>=0))+1:mask[max(0,b-2):min(len(y),b+3)]=True
 preds=dict(frozen=base,weak_ema=teacher,strong_ema=strong,source_selected=pred);rows=[]
 for method,pr in preds.items():
  assert np.isfinite(pr).all() and pr.shape==(len(y),5)
  rows.append(dict(method=method,**metrics(y,pr),transition_f1=metrics(y[mask],pr[mask])["macro_f1"]))
 tmp=out/f"s{sid:02d}.tmp.npz";np.savez_compressed(tmp,y=y,**preds);tmp.replace(out/f"s{sid:02d}.npz");done.write_text(json.dumps(dict(source=source,objective=objective,subject=sid,lr=lr,cohort="older" if sid>=40 else "middle_dev" if sid>=20 else "young_dev",results=rows,trace=trace,nonwritable_state_unchanged=unchanged,seconds=time.perf_counter()-start,exploratory=True),indent=2));print("TARGET ADAPT",source,objective,sid,lr,rows[-1]["macro_f1"],flush=True)

if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--source",required=True);parser.add_argument("--objective",required=True);parser.add_argument("--subject",type=int,required=True);a=parser.parse_args();evaluate(a.source,a.objective,a.subject)
