from phase47_pulses import model_for
from phase30_masked_graph import MaskedGraphAdapter
from phase46_spectral_head import *
from phase2_temporal import ema
OUT50=ROOT/"phase50"
def selection():
 per=pd.read_csv(ROOT/"phase48/group_people.csv");g=per[(per["mode"]=="masked")&(per.objective=="spectral")]
 means=g.groupby("group")[["clipped_future_dot","clipped_current_dot"]].mean().reindex(range(13))
 return dict(training_subjects=sorted(g.subject.unique().astype(int).tolist()),future_groups=np.flatnonzero(means.clipped_future_dot.to_numpy()>0).tolist(),current_groups=np.flatnonzero(means.clipped_current_dot.to_numpy()>0).tolist(),future_dots=means.clipped_future_dot.tolist(),current_dots=means.clipped_current_dot.tolist())
class SpectralGraph(MaskedGraphAdapter):
 def __init__(self,model,scope,example):
  sel=json.loads((OUT50/"selection.json").read_text());layout=json.loads((ROOT/"phase48/parameter_layout.json").read_text());self.scope=scope;self.target=torch.zeros(1,24,device="cuda");kind="current_groups" if scope.startswith("current") else "future_groups";self.selected=[p["group"] in sel[kind] for p in layout["parameters"]]
  super().__init__(model,.01,example)
 def _step(self):
  self.opt.zero_grad(set_to_none=True)
  with torch.no_grad():
   z=self.model.encoder.forward_features(self.x.flatten(0,1),input_chans=self.channels,return_patch_tokens=True).mean(1).reshape(-1,400);self.q=self.model.head((z-self.model.mean)/self.model.scale).softmax(1)
  e=self.model.encoder;raw=self.x.flatten(0,1);z=e.patch_embed(raw);z=torch.where(self.mask[:,:,None],self.model.mask_token.expand(2,30,-1),z);z=torch.cat([e.cls_token.expand(2,-1,-1),z],1);pos=e.pos_embed[:,self.channels];spatial=pos[:,1:].unsqueeze(2).expand(2,-1,15,-1).flatten(1,2);spatial=torch.cat([pos[:,:1].expand(2,-1,-1),spatial],1);z=z+spatial;z[:,1:]+=e.time_embed[:,:15].unsqueeze(1).expand(2,2,-1,-1).flatten(1,2);z=e.pos_drop(z)
  for block in e.blocks:z=block(z,rel_pos_bias=None)
  z=e.norm(z)[:,1:].mean(1).reshape(-1,400);pred=self.model.recon_head((z-self.model.recon_input_mean)/self.model.recon_input_scale);truth=(self.target-self.model.recon_target_mean)/self.model.recon_target_scale;self.loss=nn.functional.mse_loss(pred,truth);self.loss.backward();self.norm=nn.utils.clip_grad_norm_(self.params,1.)
  if self.scope.endswith("normmatched"):
   allsq=torch.stack([p.grad.square().sum() for p in self.params]).sum();selsq=torch.stack([p.grad.square().sum()*int(s) for p,s in zip(self.params,self.selected)]).sum();factor=(selsq/allsq.clamp_min(1e-30)).sqrt()
   for p in self.params:p.grad.mul_(factor)
  elif self.scope!="allLN":
   for p,s in zip(self.params,self.selected):
    if not s:p.grad.zero_()
  self.opt.step()
  with torch.no_grad():
   for p,v,r in zip(self.params,self.initial,self.radii):
    d=p-v;p.copy_(v+d*torch.clamp(r/torch.linalg.vector_norm(d).clamp_min(1e-12),max=1.))
  self.finite=torch.isfinite(self.loss)&torch.isfinite(self.norm)
 def step(self,x,target):
  self.x.copy_(x);self.target.copy_(target);self.graph.replay();return self.q
def audit():
 x=torch.tensor(np.array(np.load(ROOT/"phase17/cache/s05_x.npy",mmap_mode="r")[1280:1344]),device="cuda");target=torch.tensor(np.load(ROOT/"data/processed/s05_meta.npz")["features"][1280:1344,24:48],device="cuda",dtype=torch.float32);sel=json.loads((OUT50/"selection.json").read_text());layout=json.loads((ROOT/"phase48/parameter_layout.json").read_text());rows=[]
 for scope in ["allLN","future_groups","current_groups","future_normmatched","current_normmatched"]:
  m=model_for("masked");params=[p for p in m.parameters() if p.requires_grad];initial=[p.detach().clone() for p in params];radii=[.1*max(float(v.square().mean().sqrt()),1)*v.numel()**.5 for v in initial];opt=torch.optim.SGD(params,lr=.01);chosen=sel["current_groups" if scope.startswith("current") else "future_groups"];selected=[a["group"] in chosen for a in layout["parameters"]];pred=[]
  for t in range(len(x)):
   with torch.no_grad():pred.append(m(x[t:t+1]).softmax(1).cpu().numpy()[0].copy())
   opt.zero_grad(set_to_none=True);loss=m.reconstruction_loss(x[t:t+1],target[t:t+1]);loss.backward();nn.utils.clip_grad_norm_(params,1.)
   if scope.endswith("normmatched"):
    a=torch.stack([p.grad.square().sum() for p in params]).sum();b=torch.stack([p.grad.square().sum()*int(s) for p,s in zip(params,selected)]).sum();f=(b/a.clamp_min(1e-30)).sqrt()
    for p in params:p.grad.mul_(f)
   elif scope!="allLN":
    for p,s in zip(params,selected):
     if not s:p.grad.zero_()
   opt.step()
   with torch.no_grad():
    for p,v,r in zip(params,initial,radii):
     d=p-v;p.copy_(v+d*torch.clamp(r/d.norm().clamp_min(1e-12),max=1.))
  end=[p.detach().clone() for p in params];reference=np.array(pred);del m,opt
  m=model_for("masked");graph=SpectralGraph(m,scope,x[:1]);observed=[]
  for t in range(len(x)):observed.append(graph.step(x[t:t+1],target[t:t+1]).detach().cpu().numpy()[0].copy())
  error=float(abs(np.array(observed)-reference).max());pe=max(float(abs(p-v).max()) for p,v in zip(graph.params,end));dis=int((np.array(observed).argmax(1)!=reference.argmax(1)).sum());assert error<2e-5 and pe<2e-6 and dis==0
  # Change the unseen suffix: the first32 online probabilities and states must be identical.
  graph.reset();prefix=[]
  for t in range(32):prefix.append(graph.step(x[t:t+1],target[t:t+1]).detach().cpu().numpy()[0].copy())
  state=[p.detach().clone() for p in graph.params];xx=x.clone();tt=target.clone();xx[32:]=torch.flip(xx[32:],[0]);tt[32:]+=100;graph.reset();prefix2=[]
  for t in range(32):prefix2.append(graph.step(xx[t:t+1],tt[t:t+1]).detach().cpu().numpy()[0].copy())
  assert np.array_equal(prefix,prefix2) and all(torch.equal(p,v) for p,v in zip(graph.params,state));rows.append(dict(scope=scope,probability_error=error,parameter_error=pe,argmax_disagreements=dis,future_suffix_independence_exact=True));print("SPECTRAL GRAPH AUDIT",rows[-1],flush=True);del graph,m
 (OUT50/"graph_audit.json").write_text(json.dumps(dict(status="PASS",cases=rows),indent=2))
def run(scope):
 assert json.loads((OUT50/"graph_audit.json").read_text())["status"]=="PASS";out=OUT50/scope;out.mkdir(exist_ok=True)
 if (out/"complete.json").exists():return
 m=model_for("masked");x0=np.load(ROOT/"phase17/cache/s05_x.npy",mmap_mode="r");graph=SpectralGraph(m,scope,torch.tensor(np.array(x0[:1]),device="cuda"));writable={n for n,p in m.named_parameters() if p.requires_grad};fixed={n:v.clone() for n,v in m.state_dict().items() if n not in writable};results=[]
 for sid in [5,15,25,35]:
  done=out/f"s{sid:02d}.json"
  if done.exists():results.append(json.loads(done.read_text()));continue
  x=np.load(ROOT/"phase17/cache"/f"s{sid:02d}_x.npy",mmap_mode="r");root=ROOT/("data/processed" if sid<20 else "phase2/processed");meta=np.load(root/f"s{sid:02d}_meta.npz");y=meta["y"];target=meta["features"][:,24:48];base=np.load(ROOT/"phase28/mixed_joint_0"/f"s{sid:02d}.npz")["p"];graph.reset();pred=[];loss=[];norm=[];start=time.perf_counter()
  for t in range(len(y)):
   p=graph.step(torch.tensor(np.array(x[t:t+1]),device="cuda"),torch.tensor(target[t:t+1],device="cuda",dtype=torch.float32));pred.append(p.detach().cpu().numpy()[0].copy());assert bool(graph.finite);loss.append(float(graph.loss));norm.append(float(graph.norm))
  pred=np.array(pred);error=float(abs(pred[0]-base[0]).max());assert error<2e-5;assert all(torch.equal(m.state_dict()[n],v) for n,v in fixed.items());assert all(float((p-v).norm())<=r+1e-5 for p,v,r in zip(graph.params,graph.initial,graph.radii));np.savez_compressed(out/f"s{sid:02d}.npz",y=y,p=pred,loss=loss,gradient_norm=norm);rec=dict(subject=sid,scope=scope,metrics=metrics(y,pred),adapted_ema=metrics(y,ema(pred,.8)),frozen=metrics(y,base),weak_ema=metrics(y,ema(base,.8)),strong_ema=metrics(y,ema(base,.5)),first_probability_error=error,nonwritable_unchanged=True,radius_verified=True,seconds=time.perf_counter()-start);done.write_text(json.dumps(rec,indent=2));results.append(rec);print("SCOPE ONLINE",scope,sid,rec["metrics"]["macro_f1"],flush=True)
 (out/"complete.json").write_text(json.dumps(dict(scope=scope,source_validation_only=True,people=results),indent=2))
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--audit",action="store_true");parser.add_argument("--scope");a=parser.parse_args()
 if a.audit:audit()
 else:run(a.scope)
