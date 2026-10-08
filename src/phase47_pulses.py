from phase46_spectral_head import *
from phase29_forward_graph import LogitForward
OUT47=ROOT/"phase47"
def model_for(mode):
 model=SpectralCarrier(mode).cuda().eval();model.requires_grad_(False)
 for module in model.encoder.modules():
  if isinstance(module,nn.LayerNorm):
   for p in module.parameters():p.requires_grad_(True)
 return model
def audit():
 rows=[]
 for mode in ["masked","unmasked"]:
  model=model_for(mode);state={k:v.clone() for k,v in model.state_dict().items()};params=[p for p in model.parameters() if p.requires_grad]
  for sid in [5,15,25,35]:
   x,_=prepared(sid);cache=np.load(ROOT/"phase46/cache"/f"s{sid:02d}.npz");idx=np.linspace(0,len(x)-1,32,dtype=int);target=torch.tensor(cache["target"][idx],device="cuda");xx=x[idx].cuda()
   with torch.no_grad():
    actual=float(model.reconstruction_loss(xx,target));z=torch.tensor(cache[mode][idx],device="cuda");pred=model.recon_head((z-model.recon_input_mean)/model.recon_input_scale);expected=float(nn.functional.mse_loss(pred,(target-model.recon_target_mean)/model.recon_target_scale));p=model(xx).softmax(1).cpu().numpy();perr=float(abs(p-cache["classification"][idx]).max());dis=int((p.argmax(1)!=cache["classification"][idx].argmax(1)).sum())
   assert abs(actual-expected)<1e-4 and perr<2e-5 and dis==0;rows.append(dict(mode=mode,subject=sid,loss_error=abs(actual-expected),classifier_error=perr,argmax_disagreements=dis))
  x,_=prepared(5);xx=x[1280:1282].cuda();target=torch.tensor(np.load(ROOT/"phase46/cache/s05.npz")["target"][1280:1282],device="cuda");loss=model.reconstruction_loss(xx,target);grads=torch.autograd.grad(loss,params);norm=torch.sqrt(sum(g.square().sum() for g in grads));assert norm>1e-8;initial=[p.detach().clone() for p in params];epsilon=.001;losses=[]
  for sign in [1,-1]:
   with torch.no_grad():
    for p,v,g in zip(params,initial,grads):p.copy_(v+sign*epsilon*g/norm)
    losses.append(float(model.reconstruction_loss(xx,target)))
  with torch.no_grad():
   for p,v in zip(params,initial):p.copy_(v)
  derivative=(losses[0]-losses[1])/(2*epsilon);relative=abs(derivative-float(norm))/float(norm);assert relative<.01;assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items());rows.append(dict(mode=mode,directional_gradient=float(norm),central_difference=derivative,relative_error=relative,state_restored_exact=True));del model,state
 (OUT47/"implementation_audit.json").write_text(json.dumps(dict(status="PASS",cases=rows),indent=2));print("SPECTRAL LOSS AUDIT PASS",rows,flush=True)
def run(mode):
 assert json.loads((OUT47/"implementation_audit.json").read_text())["status"]=="PASS";out=OUT47/mode;out.mkdir(exist_ok=True)
 if (out/"complete.json").exists():return
 model=model_for(mode);params=[p for p in model.parameters() if p.requires_grad];initial=[p.detach().clone() for p in params];radii=[.1*max(float(v.square().mean().sqrt()),1)*v.numel()**.5 for v in initial];state={k:v.clone() for k,v in model.state_dict().items()};forward=LogitForward(model,torch.zeros(5,2,2,15,200,device="cuda"));cw=torch.tensor(json.loads((ROOT/"phase36/mixed_class_weights.json").read_text())["balanced_weights"],dtype=torch.float32,device="cuda");count=0;start=time.perf_counter();sha=hashlib.sha256((ROOT/"phase46"/f"{mode}_readout.npz").read_bytes()).hexdigest()
 def restore():
  with torch.no_grad():
   for p,v in zip(params,initial):p.copy_(v)
 for sid in [3,4,7,12,13,14,17,27,37]:
  file=out/f"s{sid:02d}.json"
  if file.exists():count+=len(json.loads(file.read_text())["rows"]);continue
  x,y=prepared(sid);root=ROOT/("data/processed" if sid<20 else "phase2/processed");target=torch.tensor(np.load(root/f"s{sid:02d}_meta.npz")["features"][:,24:48]);rows=[]
  for tt in np.linspace(0,len(y)-5,64,dtype=int):
   t=int(tt)
   if np.any(y[t:t+5]<0):continue
   restore();xx=x[t:t+5].cuda();yy=torch.tensor(y[t:t+5],device="cuda")
   with torch.no_grad():logits=forward.run(xx).clone();base=-logits.log_softmax(1)[torch.arange(5,device="cuda"),yy];bp=logits.softmax(1).cpu().numpy();q=bp[0]
   vec=nn.functional.cross_entropy(model(xx[1:]),yy[1:],reduction="none");task=torch.autograd.grad(vec.mean(),params,retain_graph=True);balanced=torch.autograd.grad((vec*cw[yy[1:]]).mean(),params);tn=torch.sqrt(sum(g.square().sum() for g in task));bn=torch.sqrt(sum(g.square().sum() for g in balanced))
   for objective in (["spectral","entropy"] if mode=="masked" else ["spectral"]):
    restore()
    if objective=="spectral":loss=model.reconstruction_loss(xx[:1],target[t:t+1].cuda())
    else:
     lp=model(xx[:1]).log_softmax(1);loss=-(lp.exp()*lp).sum(1).mean()
    grads=torch.autograd.grad(loss,params);norm=torch.sqrt(sum(g.square().sum() for g in grads));assert torch.isfinite(loss) and torch.isfinite(norm);clip=min(1.,1/max(float(norm),1e-12));dot=float(sum((g*h).sum() for g,h in zip(grads,task)));bdot=float(sum((g*h).sum() for g,h in zip(grads,balanced)))
    for lr in [.001,.01,.1]:
     restore()
     with torch.no_grad():
      for p,v,g,radius in zip(params,initial,grads,radii):
       delta=-lr*clip*g;distance=float(delta.norm());p.copy_(v+delta*min(1.,radius/max(distance,1e-12)))
      logits=forward.run(xx);pr=logits.softmax(1).cpu().numpy();post=-logits.log_softmax(1)[torch.arange(5,device="cuda"),yy]
     rows.append(dict(mode=mode,subject=sid,t=t,objective=objective,lr=lr,aux_loss=float(loss.detach()),gradient_norm=float(norm),gradient_cosine=dot/max(float(norm*tn),1e-12),balanced_gradient_cosine=bdot/max(float(norm*bn),1e-12),first_order_gain=lr*clip*dot,future_ce_gain=float((base[1:]-post[1:]).mean()),balanced_future_ce_gain=float(((base[1:]-post[1:])*cw[yy[1:]]).mean()),current_ce_gain=float(base[0]-post[0]),base_correct=int((bp[1:].argmax(1)==y[t+1:t+5]).sum()),write_correct=int((pr[1:].argmax(1)==y[t+1:t+5]).sum()),future_argmax_changes=int((pr[1:].argmax(1)!=bp[1:].argmax(1)).sum()),entropy=float(-(q*np.log(np.clip(q,1e-12,1))).sum()),confidence=float(q.max()),proposal_l1=float(abs(pr[0]-q).sum()),future_labels=y[t+1:t+5].tolist(),future_base_probabilities=bp[1:].tolist(),future_write_probabilities=pr[1:].tolist()))
   restore()
  assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items());file.write_text(json.dumps(dict(mode=mode,subject=sid,source_controller_only=True,readout_sha256=sha,model_restored_exact=True,rows=rows),indent=2));count+=len(rows);print("SPECTRAL FRESH WRITE",mode,sid,len(rows),flush=True)
 (out/"complete.json").write_text(json.dumps(dict(mode=mode,pulse_rows=count,writable_parameters=sum(p.numel() for p in params),seconds=time.perf_counter()-start,model_restored_exact=True,source_only=True),indent=2))
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--audit",action="store_true");parser.add_argument("--mode");a=parser.parse_args()
 if a.audit:audit()
 else:run(a.mode)
