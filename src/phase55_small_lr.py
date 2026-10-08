from phase53_online import *
from phase30_masked_graph import MaskedGraphAdapter
OUT55=ROOT/"phase55"
class SpectralLRGraph(SpectralGraph):
 def __init__(self,model,lr,example):
  self.scope="allLN";self.target=torch.zeros(1,24,device="cuda");self.selected=[]
  MaskedGraphAdapter.__init__(self,model,lr,example)
def audit():
 x=torch.tensor(np.array(np.load(ROOT/"phase17/cache/s05_x.npy",mmap_mode="r")[1280:1312]),device="cuda");target=torch.tensor(np.load(ROOT/"data/processed/s05_meta.npz")["features"][1280:1312,24:48],device="cuda",dtype=torch.float32);rows=[]
 for variant in ["initial","future","current","reconstruction"]:
  for lr in [0.,1e-5,1e-4,.001]:
   m=load(variant);params=[p for p in m.parameters() if p.requires_grad];initial=[p.detach().clone() for p in params];state={n:v.clone() for n,v in m.state_dict().items()};radii=[.1*max(float(v.square().mean().sqrt()),1)*v.numel()**.5 for v in initial];opt=torch.optim.SGD(params,lr=lr);pred=[]
   for t in range(len(x)):
    with torch.no_grad():pred.append(m(x[t:t+1]).softmax(1).cpu().numpy()[0].copy())
    opt.zero_grad(set_to_none=True);loss=m.reconstruction_loss(x[t:t+1],target[t:t+1]);loss.backward();nn.utils.clip_grad_norm_(params,1.);opt.step()
    with torch.no_grad():
     for p,v,r in zip(params,initial,radii):
      d=p-v;p.copy_(v+d*torch.clamp(r/d.norm().clamp_min(1e-12),max=1.))
   reference=np.array(pred);end=[p.detach().clone() for p in params];del m,opt
   m=load(variant);graph=SpectralLRGraph(m,lr,x[:1]);pred=[]
   for t in range(len(x)):pred.append(graph.step(x[t:t+1],target[t:t+1]).detach().cpu().numpy()[0].copy())
   error=float(abs(np.array(pred)-reference).max());pe=max(float(abs(p-v).max()) for p,v in zip(graph.params,end));dis=int((np.array(pred).argmax(1)!=reference.argmax(1)).sum());assert error<2e-5 and pe<2e-6 and dis==0
   if lr==0:assert all(torch.equal(v,m.state_dict()[n]) for n,v in state.items())
   rows.append(dict(variant=variant,lr=lr,probability_error=error,parameter_error=pe,argmax_disagreements=dis,zero_lr_state_unchanged=True if lr==0 else None));print("LR GRAPH AUDIT",rows[-1],flush=True);del graph,m,state
 (OUT55/"graph_audit.json").write_text(json.dumps(dict(status="PASS",cases=rows),indent=2))
def run(variant,lr):
 assert json.loads((OUT55/"graph_audit.json").read_text())["status"]=="PASS";out=OUT55/f"{variant}_{lr:g}";out.mkdir(exist_ok=True)
 if (out/"complete.json").exists():return
 protocol=json.loads((OUT55/"protocol.json").read_text())
 if variant!="initial":assert hashlib.sha256((ROOT/"phase52"/variant/"model.pt").read_bytes()).hexdigest()==protocol["model_sha256"][variant]
 m=load(variant);x0=np.load(ROOT/"phase17/cache/s05_x.npy",mmap_mode="r");graph=SpectralLRGraph(m,lr,torch.tensor(np.array(x0[:1]),device="cuda"));writable={n for n,p in m.named_parameters() if p.requires_grad};fixed={n:v.clone() for n,v in m.state_dict().items() if n not in writable};results=[]
 for sid in [5,15,25,35]:
  done=out/f"s{sid:02d}.json"
  if done.exists():results.append(json.loads(done.read_text()));continue
  x=np.load(ROOT/"phase17/cache"/f"s{sid:02d}_x.npy",mmap_mode="r");root=ROOT/("data/processed" if sid<20 else "phase2/processed");meta=np.load(root/f"s{sid:02d}_meta.npz");y=meta["y"];target=meta["features"][:,24:48];base=np.load(ROOT/"phase28/mixed_joint_0"/f"s{sid:02d}.npz")["p"];graph.reset();pred=[];loss=[];norm=[];start=time.perf_counter()
  for t in range(len(y)):
   q=graph.step(torch.tensor(np.array(x[t:t+1]),device="cuda"),torch.tensor(target[t:t+1],device="cuda",dtype=torch.float32)).detach().cpu().numpy()[0].copy();pred.append(q);assert bool(graph.finite);loss.append(float(graph.loss.detach()));norm.append(float(graph.norm.detach()))
  pred=np.array(pred);assert abs(pred[0]-base[0]).max()<2e-5;assert all(torch.equal(v,m.state_dict()[n]) for n,v in fixed.items());ratios=[float((p-v).norm())/r for p,v,r in zip(graph.params,graph.initial,graph.radii)];assert max(ratios)<1.00001;np.savez_compressed(out/f"s{sid:02d}.npz",y=y,p=pred,loss=loss,gradient_norm=norm);rec=dict(variant=variant,lr=lr,subject=sid,metrics=metrics(y,pred),adapted_ema=metrics(y,ema(pred,.8)),nonwritable_unchanged=True,radius_verified=True,max_radius_fraction=max(ratios),seconds=time.perf_counter()-start);done.write_text(json.dumps(rec,indent=2));results.append(rec);print("SMALL LR ONLINE",variant,lr,sid,rec["metrics"]["macro_f1"],flush=True)
 (out/"complete.json").write_text(json.dumps(dict(variant=variant,lr=lr,source_validation_only=True,people=results),indent=2))
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--audit",action="store_true");parser.add_argument("--variant");parser.add_argument("--lr",type=float);a=parser.parse_args()
 if a.audit:audit()
 else:run(a.variant,a.lr)
