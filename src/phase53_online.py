from phase52_evaluate import load
from phase50_scope_online import SpectralGraph
from phase46_spectral_head import *
from phase2_temporal import ema
OUT53=ROOT/"phase53"
def audit():
 x=torch.tensor(np.array(np.load(ROOT/"phase17/cache/s05_x.npy",mmap_mode="r")[1280:1408]),device="cuda");target=torch.tensor(np.load(ROOT/"data/processed/s05_meta.npz")["features"][1280:1408,24:48],device="cuda",dtype=torch.float32);rows=[]
 for variant in ["future","current","reconstruction"]:
  for trajectory in ["continuous","reset64"]:
   m=load(variant);params=[p for p in m.parameters() if p.requires_grad];initial=[p.detach().clone() for p in params];radii=[.1*max(float(v.square().mean().sqrt()),1)*v.numel()**.5 for v in initial];opt=torch.optim.SGD(params,lr=.01);pred=[]
   for t in range(len(x)):
    if trajectory=="reset64" and t%64==0:
     with torch.no_grad():
      for p,v in zip(params,initial):p.copy_(v)
    with torch.no_grad():pred.append(m(x[t:t+1]).softmax(1).cpu().numpy()[0].copy())
    opt.zero_grad(set_to_none=True);loss=m.reconstruction_loss(x[t:t+1],target[t:t+1]);loss.backward();nn.utils.clip_grad_norm_(params,1.);opt.step()
    with torch.no_grad():
     for p,v,r in zip(params,initial,radii):
      d=p-v;p.copy_(v+d*torch.clamp(r/d.norm().clamp_min(1e-12),max=1.))
   reference=np.array(pred);end=[p.detach().clone() for p in params];del m,opt
   m=load(variant);graph=SpectralGraph(m,"allLN",x[:1]);pred=[]
   for t in range(len(x)):
    if trajectory=="reset64" and t%64==0:graph.reset()
    pred.append(graph.step(x[t:t+1],target[t:t+1]).detach().cpu().numpy()[0].copy())
   error=float(abs(np.array(pred)-reference).max());pe=max(float(abs(p-v).max()) for p,v in zip(graph.params,end));dis=int((np.array(pred).argmax(1)!=reference.argmax(1)).sum());assert error<2e-5 and pe<2e-6 and dis==0
   graph.reset();a=[]
   for t in range(32):a.append(graph.step(x[t:t+1],target[t:t+1]).detach().cpu().numpy()[0].copy())
   state=[p.detach().clone() for p in graph.params];xx=x.clone();tt=target.clone();xx[32:]*=-1.5;tt[32:]+=100;graph.reset();b=[]
   for t in range(32):b.append(graph.step(xx[t:t+1],tt[t:t+1]).detach().cpu().numpy()[0].copy())
   assert np.array_equal(a,b) and all(torch.equal(p,v) for p,v in zip(graph.params,state));rec=dict(variant=variant,trajectory=trajectory,windows=128,probability_error=error,parameter_error=pe,argmax_disagreements=dis,future_suffix_independence_exact=True);rows.append(rec);print("META ONLINE AUDIT",rec,flush=True);del graph,m
 (OUT53/"graph_audit.json").write_text(json.dumps(dict(status="PASS",cases=rows),indent=2))
def run(variant,trajectory):
 assert json.loads((OUT53/"graph_audit.json").read_text())["status"]=="PASS";out=OUT53/(variant+"_"+trajectory);out.mkdir(exist_ok=True)
 if (out/"complete.json").exists():return
 protocol=json.loads((OUT53/"protocol.json").read_text())
 if variant!="initial":assert hashlib.sha256((ROOT/"phase52"/variant/"model.pt").read_bytes()).hexdigest()==protocol["model_sha256"][variant]
 m=load(variant);x0=np.load(ROOT/"phase17/cache/s05_x.npy",mmap_mode="r");graph=SpectralGraph(m,"allLN",torch.tensor(np.array(x0[:1]),device="cuda"));writable={n for n,p in m.named_parameters() if p.requires_grad};fixed={n:v.clone() for n,v in m.state_dict().items() if n not in writable};results=[]
 for sid in [5,15,25,35]:
  done=out/f"s{sid:02d}.json"
  if done.exists():results.append(json.loads(done.read_text()));continue
  x=np.load(ROOT/"phase17/cache"/f"s{sid:02d}_x.npy",mmap_mode="r");root=ROOT/("data/processed" if sid<20 else "phase2/processed");meta=np.load(root/f"s{sid:02d}_meta.npz");y=meta["y"];target=meta["features"][:,24:48];base=np.load(ROOT/"phase28/mixed_joint_0"/f"s{sid:02d}.npz")["p"];graph.reset();pred=[];loss=[];norm=[];reseterr=0.;start=time.perf_counter()
  for t in range(len(y)):
   reset=trajectory=="reset64" and t%64==0
   if reset:graph.reset()
   q=graph.step(torch.tensor(np.array(x[t:t+1]),device="cuda"),torch.tensor(target[t:t+1],device="cuda",dtype=torch.float32)).detach().cpu().numpy()[0].copy();pred.append(q);assert bool(graph.finite);loss.append(float(graph.loss.detach()));norm.append(float(graph.norm.detach()))
   if reset or t==0:reseterr=max(reseterr,float(abs(q-base[t]).max()))
  pred=np.array(pred);assert reseterr<2e-5;assert all(torch.equal(v,m.state_dict()[n]) for n,v in fixed.items());ratios=[float((p-v).norm())/r for p,v,r in zip(graph.params,graph.initial,graph.radii)];assert max(ratios)<1.00001;np.savez_compressed(out/f"s{sid:02d}.npz",y=y,p=pred,loss=loss,gradient_norm=norm);rec=dict(variant=variant,trajectory=trajectory,subject=sid,metrics=metrics(y,pred),adapted_ema=metrics(y,ema(pred,.8)),max_reset_probability_error=reseterr,nonwritable_unchanged=True,radius_verified=True,max_radius_fraction=max(ratios),seconds=time.perf_counter()-start);done.write_text(json.dumps(rec,indent=2));results.append(rec);print("META ONLINE",variant,trajectory,sid,rec["metrics"]["macro_f1"],flush=True)
 (out/"complete.json").write_text(json.dumps(dict(variant=variant,trajectory=trajectory,source_validation_only=True,people=results),indent=2))
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--audit",action="store_true");parser.add_argument("--variant");parser.add_argument("--trajectory");a=parser.parse_args()
 if a.audit:audit()
 else:run(a.variant,a.trajectory)
