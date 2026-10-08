from phase66_train import *
OUT69=ROOT/"phase69"
def make(kind):
 choice=json.loads((OUT66/"selection.json").read_text())["selected"][kind];path=Path(choice["folder"])/"model.pt";assert hashlib.sha256(path.read_bytes()).hexdigest()==choice["checkpoint_sha256"]
 m=Carrier(kind).cuda().eval();m.load_state_dict(torch.load(path,map_location="cpu",weights_only=False)["state"]);m.requires_grad_(False)
 for module in m.encoder.modules():
  if isinstance(module,nn.LayerNorm):
   for p in module.parameters():p.requires_grad_(True)
 return m
def sparse(sid,t):
 root=ROOT/("data/processed" if sid<20 else "phase2/processed");raw=np.load(root/f"s{sid:02d}_x.npy",mmap_mode="r");y=np.load(root/f"s{sid:02d}_meta.npz")["y"][t:t+5]
 return torch.tensor(prep(np.array(raw[t:t+5]),[.1,45]),device="cuda"),torch.tensor(y,device="cuda")
def run(kind):
 out=OUT69/kind;out.mkdir(exist_ok=True);m=make(kind);named=[(n,p) for n,p in m.named_parameters() if p.requires_grad];params=[p for n,p in named];initial=[p.detach().clone() for p in params];radii=[.1*max(float(v.square().mean().sqrt()),1)*v.numel()**.5 for v in initial];state=snapshot(m);cw=torch.tensor(json.loads((ROOT/"phase36/mixed_class_weights.json").read_text())["balanced_weights"],device="cuda",dtype=torch.float64);allrows=0
 (out/"writable.json").write_text(json.dumps(dict(names=[n for n,p in named],parameters=sum(p.numel() for p in params),layernorm_only=True),indent=2))
 def restore():
  with torch.no_grad():
   for p,v in zip(params,initial):p.copy_(v)
 for sid in [7,17,27,37]:
  dest=out/f"s{sid:02d}.json"
  if dest.exists():allrows+=len(json.loads(dest.read_text())["rows"]);continue
  times=np.load(ROOT/"phase48/masked"/f"s{sid:02d}_gradients.npz")["times"];rows=[]
  for t0 in times:
   t=int(t0);restore();x,y=sparse(sid,t);assert bool((y>=0).all())
   with torch.no_grad():base=m(x);blp=base.double().log_softmax(1);bp=blp.exp().cpu().numpy();before=-blp[torch.arange(5,device="cuda"),y]
   lp=m(x[:1]).log_softmax(1);loss=-(lp.exp()*lp).sum(1).mean();grads=torch.autograd.grad(loss,params);norm=torch.sqrt(sum(g.square().sum() for g in grads));assert torch.isfinite(norm) and torch.isfinite(loss);clip=torch.clamp(1/(norm+1e-6),max=1.)
   for lr in [.0001,.001,.01]:
    restore()
    with torch.no_grad():
     for p,v,g,radius in zip(params,initial,grads,radii):
      d=-lr*clip*g;p.copy_(v+d*torch.clamp(radius/d.norm().clamp_min(1e-12),max=1.))
     post=m(x);plp=post.double().log_softmax(1);wp=plp.exp().cpu().numpy();after=-plp[torch.arange(5,device="cuda"),y];actual=float(torch.sqrt(sum((p-v).square().sum() for p,v in zip(params,initial))));assert actual<=lr+1e-5
    b=bp[1:].argmax(1);w=wp[1:].argmax(1);truth=y[1:].cpu().numpy();rows.append(dict(kind=kind,subject=sid,t=t,lr=lr,entropy=float(loss.detach()),gradient_norm=float(norm.detach()),actual_update_norm=actual,future_ce_gain=float((before[1:]-after[1:]).mean()),balanced_ce_gain=float(((before[1:]-after[1:])*cw[y[1:]]).mean()),current_ce_gain=float(before[0]-after[0]),future_labels=truth.tolist(),future_base_probabilities=bp[1:].tolist(),future_write_probabilities=wp[1:].tolist(),corrected=int(((b!=truth)&(w==truth)).sum()),harmed=int(((b==truth)&(w!=truth)).sum()),argmax_changes=int((b!=w).sum())))
  restore();assert all(torch.equal(v.cpu(),state[n]) for n,v in m.state_dict().items());dest.write_text(json.dumps(dict(kind=kind,subject=sid,state_restored_exact=True,rows=rows),indent=2));allrows+=len(rows);print("CBRAMOD ENTROPY PULSES",kind,sid,len(rows),flush=True)
 (out/"complete.json").write_text(json.dumps(dict(kind=kind,rows=allrows),indent=2))
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--kind",required=True);a=p.parse_args();run(a.kind)
