from phase55_small_lr import *
from phase29_forward_graph import LogitForward
OUT60=ROOT/"phase60"
class DecayGraph(SpectralLRGraph):
 def __init__(self,model,lr,example,beta):
  self.beta=beta;super().__init__(model,lr,example)
 def _step(self):
  super()._step()
  if self.beta!=1:
   with torch.no_grad():
    for p,v in zip(self.params,self.initial):p.copy_(v+self.beta*(p-v))
def setting(mode):
 if mode.startswith("decay"):return float(mode[5:]),None
 if mode.startswith("warmup"):return 1.,int(mode[6:])
 return 1.,None
def audit():
 rows=[]
 for variant,lr in [("current",.0001),("future",.01)]:
  for mode in ["decay0.99","decay0.999","warmup64","warmup256"]:
   beta,warm=setting(mode);n=(warm+16 if warm else 64);x=torch.tensor(np.array(np.load(ROOT/"phase17/cache/s05_x.npy",mmap_mode="r")[1280:1280+n]),device="cuda");target=torch.tensor(np.load(ROOT/"data/processed/s05_meta.npz")["features"][1280:1280+n,24:48],device="cuda",dtype=torch.float32);m=load(variant);params=[p for p in m.parameters() if p.requires_grad];initial=[p.detach().clone() for p in params];radii=[.1*max(float(v.square().mean().sqrt()),1)*v.numel()**.5 for v in initial];opt=torch.optim.SGD(params,lr=lr);pred=[]
   for t in range(n):
    with torch.no_grad():pred.append(m(x[t:t+1]).softmax(1).cpu().numpy()[0].copy())
    if warm is not None and t>=warm:continue
    opt.zero_grad(set_to_none=True);loss=m.reconstruction_loss(x[t:t+1],target[t:t+1]);loss.backward();nn.utils.clip_grad_norm_(params,1.);opt.step()
    with torch.no_grad():
     for p,v,r in zip(params,initial,radii):
      d=p-v;p.copy_(v+d*torch.clamp(r/d.norm().clamp_min(1e-12),max=1.))
      if beta!=1:p.copy_(v+beta*(p-v))
   ref=np.array(pred);end=[p.detach().clone() for p in params];del m,opt
   m=load(variant);graph=DecayGraph(m,lr,x[:1],beta);one=LogitForward(m,x[:1]);pred=[]
   for t in range(n):
    if warm is None or t<warm:q=graph.step(x[t:t+1],target[t:t+1])
    else:
     with torch.no_grad():q=one.run(x[t:t+1]).softmax(1)
    pred.append(q.detach().cpu().numpy()[0].copy())
   err=float(abs(np.array(pred)-ref).max());pe=max(float(abs(p-v).max()) for p,v in zip(graph.params,end));dis=int((np.array(pred).argmax(1)!=ref.argmax(1)).sum());assert err<2e-5 and pe<2e-6 and dis==0;rows.append(dict(variant=variant,mode=mode,lr=lr,windows=n,probability_error=err,parameter_error=pe,argmax_disagreements=dis));print("MEMORY CONTROL AUDIT",rows[-1],flush=True);del graph,one,m
 (OUT60/"graph_audit.json").write_text(json.dumps(dict(status="PASS",cases=rows),indent=2))
def run(variant,mode):
 assert json.loads((OUT60/"graph_audit.json").read_text())["status"]=="PASS";out=OUT60/(variant+"_"+mode);out.mkdir(exist_ok=True)
 if (out/"complete.json").exists():return
 protocol=json.loads((OUT60/"protocol.json").read_text());lr=protocol["source_selected_lr"][variant];assert hashlib.sha256((ROOT/"phase52"/variant/"model.pt").read_bytes()).hexdigest()==protocol["model_sha256"][variant];beta,warm=setting(mode);m=load(variant);x0=np.load(ROOT/"phase17/cache/s05_x.npy",mmap_mode="r");graph=DecayGraph(m,lr,torch.tensor(np.array(x0[:1]),device="cuda"),beta);one=LogitForward(m,torch.tensor(np.array(x0[:1]),device="cuda"));writable={n for n,p in m.named_parameters() if p.requires_grad};fixed={n:v.clone() for n,v in m.state_dict().items() if n not in writable};results=[]
 for sid in [5,15,25,35]:
  done=out/f"s{sid:02d}.json"
  if done.exists():results.append(json.loads(done.read_text()));continue
  x=np.load(ROOT/"phase17/cache"/f"s{sid:02d}_x.npy",mmap_mode="r");root=ROOT/("data/processed" if sid<20 else "phase2/processed");meta=np.load(root/f"s{sid:02d}_meta.npz");y=meta["y"];target=meta["features"][:,24:48];base=np.load(ROOT/"phase28/mixed_joint_0"/f"s{sid:02d}.npz")["p"];graph.reset();pred=[];loss=[];norm=[];updates=[];held=None;start=time.perf_counter()
  for t in range(len(x)):
   if mode=="reset64" and t%64==0:graph.reset()
   update=warm is None or t<warm;xt=torch.tensor(np.array(x[t:t+1]),device="cuda")
   if update:
    q=graph.step(xt,torch.tensor(target[t:t+1],device="cuda",dtype=torch.float32));assert bool(graph.finite);loss.append(float(graph.loss.detach()));norm.append(float(graph.norm.detach()))
   else:
    with torch.no_grad():q=one.run(xt).softmax(1)
    loss.append(0.);norm.append(0.)
   pred.append(q.detach().cpu().numpy()[0].copy());updates.append(update)
   if warm is not None and t==warm-1:held=[p.detach().clone() for p in graph.params]
  pred=np.array(pred);assert abs(pred[0]-base[0]).max()<2e-5;assert all(torch.equal(v,m.state_dict()[n]) for n,v in fixed.items());assert all(float((p-v).norm())<=r+1e-5 for p,v,r in zip(graph.params,graph.initial,graph.radii))
  if held is not None:assert all(torch.equal(p,v) for p,v in zip(graph.params,held))
  if mode=="reset64":assert abs(pred[::64]-base[::64]).max()<2e-5
  final=torch.cat([p.detach().flatten() for p in graph.params]).cpu().numpy();np.savez_compressed(out/f"s{sid:02d}.npz",y=y,p=pred,loss=loss,gradient_norm=norm,gradient_step=updates,final_params=final);rec=dict(variant=variant,mode=mode,lr=lr,beta=beta,warmup=warm,subject=sid,metrics=metrics(y,pred),gradient_steps=int(sum(updates)),windows=len(y),zero_loss_entries_after_warmup_are_uncomputed=True,parameter_drift=float(torch.sqrt(sum((p-v).square().sum() for p,v in zip(graph.params,graph.initial)))),nonwritable_unchanged=True,radius_verified=True,warmup_state_held_exact=True if warm else None,seconds=time.perf_counter()-start);done.write_text(json.dumps(rec,indent=2));results.append(rec);print("MEMORY CONTROL",variant,mode,sid,rec["metrics"]["macro_f1"],flush=True)
 (out/"complete.json").write_text(json.dumps(dict(variant=variant,mode=mode,people=results,source_validation_only=True),indent=2))
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--audit",action="store_true");parser.add_argument("--variant");parser.add_argument("--mode");a=parser.parse_args()
 if a.audit:audit()
 else:run(a.variant,a.mode)
