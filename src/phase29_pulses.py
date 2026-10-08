from phase28_joint_train import *
import hashlib
from phase29_forward_graph import LogitForward
OUT29=ROOT/"phase29"
def run(source,weight):
 folder=ROOT/"phase28"/f"{source}_joint_{weight:g}";out=OUT29/f"{source}_joint_{weight:g}";out.mkdir(parents=True,exist_ok=True)
 if (out/"complete.json").exists():return
 model=JointCarrier(source).cuda().eval();checkpoint=folder/"model.pt";model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=False)["state"]);model.requires_grad_(False)
 for module in model.encoder.modules():
  if isinstance(module,nn.LayerNorm):
   for p in module.parameters():p.requires_grad_(True)
 params=[p for p in model.parameters() if p.requires_grad];initial=[p.detach().clone() for p in params];full_state={k:v.clone() for k,v in model.state_dict().items()};lrs=[1e-5,1e-4,.001,.01];start=time.perf_counter();allrows=[];radii=[.1*max(float(v.square().mean().sqrt()),1.)*np.sqrt(v.numel()) for v in initial];forward=LogitForward(model,torch.zeros(5,2,2,15,200,device="cuda"))
 def restore():
  with torch.no_grad():
   for p,v in zip(params,initial):p.copy_(v)
 for sid in [7,17,27,37]:
  dest=out/f"s{sid:02d}.json"
  if dest.exists():allrows.extend(json.loads(dest.read_text())["rows"]);continue
  x,y=prepared(sid);tokens=torch.tensor(np.load(ROOT/"phase28/tokens"/f"s{sid:02d}.npz")["tokens"].astype(np.int64));times=np.linspace(0,len(y)-5,64,dtype=int);rows=[]
  for tt in times:
   t=int(tt)
   if np.any(y[t:t+5]<0):continue
   restore();xx=x[t:t+5].cuda();yy=torch.tensor(y[t:t+5],device="cuda");target=tokens[t:t+1].cuda()
   with torch.no_grad():base_logits=forward.run(xx).clone();base_lp=base_logits.log_softmax(1);base_losses=-base_lp[torch.arange(5,device="cuda"),yy];q=base_logits[0].softmax(0).cpu().numpy()
   future_loss=nn.functional.cross_entropy(model(xx[1:]),yy[1:]);future_grads=torch.autograd.grad(future_loss,params);future_norm=torch.sqrt(sum(g.square().sum() for g in future_grads))
   for objective in ["entropy","masked_token"]:
    restore()
    if objective=="entropy":
     lp=model(xx[:1]).log_softmax(1);loss=-(lp.exp()*lp).sum(1).mean()
    else:loss,_=model.auxiliary(xx[:1],target,False)
    grads=torch.autograd.grad(loss,params);norm=torch.sqrt(sum(g.square().sum() for g in grads));clip=min(1.,1/max(float(norm),1e-12));dot=float(sum((g*h).sum() for g,h in zip(grads,future_grads)));cos=dot/max(float(norm*future_norm),1e-12)
    for lr in lrs:
     restore()
     with torch.no_grad():
      for p,v,g,radius in zip(params,initial,grads,radii):
       p.copy_(v-lr*clip*g);delta=p-v;distance=float(delta.norm())
       if distance>radius:p.copy_(v+delta*(radius/distance))
      updated=forward.run(xx);ulp=updated.log_softmax(1);losses=-ulp[torch.arange(5,device="cuda"),yy];after=updated[0].softmax(0).cpu().numpy()
     rows.append(dict(source=source,weight=weight,subject=sid,t=t,objective=objective,lr=lr,aux_loss=float(loss.detach()),gradient_norm=float(norm),gradient_cosine=cos,first_order_gain=lr*clip*dot,future_ce_gain=float(base_losses[1:].mean()-losses[1:].mean()),current_ce_gain=float(base_losses[0]-losses[0]),base_ce=float(base_losses[1:].mean()),current_correct=bool(q.argmax()==y[t]),current_class=int(y[t]),stable=bool(np.all(y[t:t+5]==y[t])),entropy=float(-(q*np.log(np.clip(q,1e-12,1))).sum()),confidence=float(q.max()),proposal_l1=float(np.abs(after-q).sum()),base_correct=int((base_logits[1:].argmax(1)==yy[1:]).sum()),write_correct=int((updated[1:].argmax(1)==yy[1:]).sum())))
   restore()
  assert all(torch.equal(v,model.state_dict()[k]) for k,v in full_state.items());dest.write_text(json.dumps(dict(source=source,weight=weight,subject=sid,rows=rows,model_restored_exact=True,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),source_controller_only=True,execution="audited_logit_cuda_graph_v2"),indent=2));allrows.extend(rows);print("JOINT PULSES",source,weight,sid,len(rows),flush=True)
 (out/"complete.json").write_text(json.dumps(dict(source=source,weight=weight,pulse_rows=len(allrows),writable_parameters=sum(p.numel() for p in params),model_restored_exact=True,source_controller_only=True,seconds=time.perf_counter()-start,execution="audited_logit_cuda_graph_v2"),indent=2))
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--source",required=True);parser.add_argument("--weight",type=float,required=True);a=parser.parse_args();run(a.source,a.weight)
