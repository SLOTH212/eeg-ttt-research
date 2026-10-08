from phase35_head_train import *
from phase29_forward_graph import LogitForward
OUT36=ROOT/"phase36"
def run(source,variant):
 folder=ROOT/"phase35"/source if variant=="fitted" else ROOT/"phase28"/f"{source}_joint_0";checkpoint=folder/"model.pt";out=OUT36/f"{source}_{variant}";out.mkdir(exist_ok=True)
 if (out/"complete.json").exists():return
 model=JointCarrier(source).cuda().eval();model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=False)["state"]);model.requires_grad_(False)
 for module in model.encoder.modules():
  if isinstance(module,nn.LayerNorm):
   for p in module.parameters():p.requires_grad_(True)
 params=[p for p in model.parameters() if p.requires_grad];initial=[p.detach().clone() for p in params];radii=[.1*max(float(v.square().mean().sqrt()),1)*v.numel()**.5 for v in initial];full_state={k:v.clone() for k,v in model.state_dict().items()};forward=LogitForward(model,torch.zeros(5,2,2,15,200,device="cuda"));split=json.loads((ROOT/"phase3/protocol.json").read_text())["folds"][source];counts=np.zeros(5)
 for sid in split["base_train"]:
  root=ROOT/("data/processed" if sid<20 else "phase2/processed");yy=np.load(root/f"s{sid:02d}_meta.npz")["y"];counts+=np.bincount(yy[yy>=0],minlength=5)
 weights=counts.sum()/(5*np.maximum(counts,1));class_weights=torch.tensor(weights,dtype=torch.float32,device="cuda");(OUT36/f"{source}_class_weights.json").write_text(json.dumps(dict(source_counts=counts.tolist(),balanced_weights=weights.tolist(),source_train_only=True),indent=2));allrows=[];start=time.perf_counter()
 def restore():
  with torch.no_grad():
   for p,v in zip(params,initial):p.copy_(v)
 for sid in [7,17,27,37]:
  dest=out/f"s{sid:02d}.json"
  if dest.exists():allrows.extend(json.loads(dest.read_text())["rows"]);continue
  x,y=prepared(sid);tokens=torch.tensor(np.load(ROOT/"phase28/tokens"/f"s{sid:02d}.npz")["tokens"].astype(np.int64));rows=[]
  for tt in np.linspace(0,len(y)-5,64,dtype=int):
   t=int(tt)
   if np.any(y[t:t+5]<0):continue
   restore();xx=x[t:t+5].cuda();yy=torch.tensor(y[t:t+5],device="cuda")
   with torch.no_grad():base_logits=forward.run(xx).clone();base_losses=-base_logits.log_softmax(1)[torch.arange(5,device="cuda"),yy];base_probs=base_logits.softmax(1).cpu().numpy();q=base_probs[0]
   loss_vec=nn.functional.cross_entropy(model(xx[1:]),yy[1:],reduction="none");task_grads=torch.autograd.grad(loss_vec.mean(),params,retain_graph=True);balanced_grads=torch.autograd.grad((loss_vec*class_weights[yy[1:]]).mean(),params);task_norm=torch.sqrt(sum(g.square().sum() for g in task_grads));balanced_norm=torch.sqrt(sum(g.square().sum() for g in balanced_grads))
   objectives=["masked_token","entropy"] if variant=="fitted" else ["masked_token"]
   for objective in objectives:
    restore()
    if objective=="entropy":
     lp=model(xx[:1]).log_softmax(1);loss=-(lp.exp()*lp).sum(1).mean()
    else:loss,_=model.auxiliary(xx[:1],tokens[t:t+1].cuda(),False)
    grads=torch.autograd.grad(loss,params);norm=torch.sqrt(sum(g.square().sum() for g in grads));assert torch.isfinite(loss) and torch.isfinite(norm);clip=min(1.,1/max(float(norm),1e-12));dot=float(sum((g*h).sum() for g,h in zip(grads,task_grads)));balanced_dot=float(sum((g*h).sum() for g,h in zip(grads,balanced_grads)))
    for lr in [.001,.01,.03,.1]:
     restore()
     with torch.no_grad():
      for p,v,g,radius in zip(params,initial,grads,radii):
       p.copy_(v-lr*clip*g);delta=p-v;distance=float(delta.norm())
       if distance>radius:p.copy_(v+delta*(radius/distance))
      updated=forward.run(xx);prob=updated.softmax(1).cpu().numpy();losses=-updated.log_softmax(1)[torch.arange(5,device="cuda"),yy]
     rows.append(dict(source=source,variant=variant,subject=sid,t=t,objective=objective,lr=lr,aux_loss=float(loss.detach()),gradient_norm=float(norm),gradient_cosine=dot/max(float(norm*task_norm),1e-12),balanced_gradient_cosine=balanced_dot/max(float(norm*balanced_norm),1e-12),first_order_gain=lr*clip*dot,future_ce_gain=float(base_losses[1:].mean()-losses[1:].mean()),balanced_future_ce_gain=float(((base_losses[1:]-losses[1:])*class_weights[yy[1:]]).mean()),current_ce_gain=float(base_losses[0]-losses[0]),base_ce=float(base_losses[1:].mean()),base_correct=int((base_probs[1:].argmax(1)==y[t+1:t+5]).sum()),write_correct=int((prob[1:].argmax(1)==y[t+1:t+5]).sum()),future_argmax_changes=int((prob[1:].argmax(1)!=base_probs[1:].argmax(1)).sum()),entropy=float(-(q*np.log(np.clip(q,1e-12,1))).sum()),confidence=float(q.max()),proposal_l1=float(abs(prob[0]-q).sum()),future_labels=y[t+1:t+5].tolist(),future_base_probabilities=base_probs[1:].tolist(),future_write_probabilities=prob[1:].tolist()))
   restore()
  assert all(torch.equal(v,model.state_dict()[k]) for k,v in full_state.items());dest.write_text(json.dumps(dict(source=source,variant=variant,subject=sid,rows=rows,model_restored_exact=True,source_controller_only=True,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()),indent=2));allrows.extend(rows);print("AUX HEAD WRITE",source,variant,sid,len(rows),flush=True)
 (out/"complete.json").write_text(json.dumps(dict(source=source,variant=variant,pulse_rows=len(allrows),writable_parameters=sum(p.numel() for p in params),source_only=True,model_restored_exact=True,seconds=time.perf_counter()-start),indent=2))
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--source",required=True);parser.add_argument("--variant",required=True);a=parser.parse_args();run(a.source,a.variant)
