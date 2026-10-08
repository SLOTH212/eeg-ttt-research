from phase22_cuda_graph import GraphAdapter
from phase22_source_screen import *
from phase24_observability import values
from phase2_temporal import ema
OUT25=ROOT/"phase25"
def entropy(p):return float(-(p*np.log(np.clip(p,1e-12,1))).sum())
class ProposalForward:
 def __init__(self,model,example):
  self.model=model;self.x=example.clone().cuda();self.channels=torch.tensor([0,127,128],device="cuda");s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(s),torch.inference_mode():
   for _ in range(3):self.forward()
  torch.cuda.current_stream().wait_stream(s);self.graph=torch.cuda.CUDAGraph()
  with torch.inference_mode(),torch.cuda.graph(self.graph):self.forward()
 def forward(self):
  z=self.model.encoder.forward_features(self.x.flatten(0,1),input_chans=self.channels,return_patch_tokens=True).mean(1).reshape(-1,400);self.q=self.model.head((z-self.model.mean)/self.model.scale).softmax(1)
 def run(self,x):self.x.copy_(x);self.graph.replay();return self.q

def run(source,policy,sid):
 out=OUT25/f"{source}_{policy}";out.mkdir(exist_ok=True);done=out/f"s{sid:02d}.json"
 if done.exists():return
 cache=np.load(OUT25/"cache"/f"{source}_s{sid:02d}.npz");base=cache["probabilities"];embedding=cache["embeddings"];x=np.load(ROOT/"phase17/cache"/f"s{sid:02d}_x.npy",mmap_mode="r");root=ROOT/("data/processed" if sid<20 else "phase2/processed");meta=np.load(root/f"s{sid:02d}_meta.npz");y=meta["y"];spectral=meta["features"];histories={}
 for alpha in [.2,.8]:
  rows=[];h=None
  for q in base:h=q.copy() if h is None else alpha*q+(1-alpha)*h;rows.append(h.copy())
  histories[alpha]=np.array(rows)
 gate=None
 if not policy.startswith("cadence"):
  with (ROOT/"phase24/gates"/f"{source}_{policy}.pkl").open("rb") as f:gate=pickle.load(f)
 model=FullCarrier(source,"pretrained").eval().cuda();model.load_state_dict(torch.load(source_checkpoint(source),map_location="cpu",weights_only=False)["state"]);model.requires_grad_(False)
 for module in model.encoder.modules():
  if isinstance(module,nn.LayerNorm):
   for p in module.parameters():p.requires_grad_(True)
 frozen={n:v.detach().clone() for n,v in model.state_dict().items() if not dict(model.named_parameters()).get(n,torch.tensor(0.)).requires_grad};example=torch.tensor(np.array(x[:1]),device="cuda");graph=GraphAdapter(model,"entropy",.001,example);forward=ProposalForward(model,example);pred=[];actions=[];start=time.perf_counter()
 for t in range(len(y)):
  before=[p.detach().clone() for p in graph.params];xt=torch.tensor(np.array(x[t:t+1]),device="cuda");q=graph.step(xt,torch.as_tensor(base[t],device="cuda")).detach().cpu().numpy()[0].copy();pred.append(q);assert bool(graph.finite)
  if gate is None:commit=t%int(policy.removeprefix("cadence"))==0
  else:
   after=forward.run(xt).cpu().numpy()[0].copy();drift=float(torch.sqrt(sum((v-s).square().sum() for v,s in zip(before,graph.initial))))
   features=[entropy(q),float(graph.norm.detach()),float(np.abs(after-q).sum()),drift,float(q.max()),entropy(after),*q.tolist(),float(np.mean(np.abs(spectral[t]-spectral[t-1]))) if t else 0.,float(t==0),*base[t].tolist(),entropy(base[t]),float(np.abs(q-base[t]).sum()),*after.tolist(),*histories[.2][t].tolist(),*histories[.8][t].tolist(),*embedding[t].tolist()];assert len(features)==435;commit=bool(values(gate,np.array(features)[None])[0,1]>0)
  actions.append(int(commit))
  if not commit:
   with torch.no_grad():
    for p,v in zip(graph.params,before):p.copy_(v)
 pred=np.array(pred);assert np.max(np.abs(pred[0]-base[0]))<.002;assert all(torch.equal(model.state_dict()[n],v) for n,v in frozen.items());np.savez_compressed(out/f"s{sid:02d}.npz",y=y,predictions=pred,actions=np.array(actions,dtype=np.int8));result=dict(source=source,policy=policy,subject=sid,metrics=metrics(y,pred),frozen=metrics(y,base),weak_ema=metrics(y,ema(base,.8)),commit_fraction=float(np.mean(actions)),nonwritable_state_unchanged=True,seconds=time.perf_counter()-start,source_validation_only=True);done.write_text(json.dumps(result,indent=2));print("SOURCE GATE",source,policy,sid,result["metrics"]["macro_f1"],result["commit_fraction"],result["seconds"],flush=True)
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--source",required=True);parser.add_argument("--policy",required=True);parser.add_argument("--subject",type=int,required=True);a=parser.parse_args();run(a.source,a.policy,a.subject)
