from phase69_pulses import *
OUT73=ROOT/"phase73"
def run():
 OUT73.mkdir(exist_ok=True)
 assert json.loads((ROOT/"phase72/audit.json").read_text())["status"]=="PASS"
 assert json.loads((ROOT/"phase72/decision.json").read_text())["validity_pass"]
 protocol=dict(classifier_seed=17,checkpoint=json.loads((OUT66/"selection.json").read_text())["selected"]["pretrained"],controllers=[7,17,27,37],times="phase48/masked/sXX_gradients.npz",horizon=4,learning_rates=[.001,.01,.1],primary="refitted_lr0.01",variants=["refitted","original","original_normmatched"],mask="Bernoulli 0.5, dedicated seed 73000+subject*100000+t",writable="9600 LayerNorm affine only",prediction_before_update=True,advance_rule="primary mean CE positive, >=3/4 people positive, balanced CE nonnegative, corrected>=harmed; no automatic target rollout",scope="source-only cold single-write diagnostic; source people already observed")
 (OUT73/"protocol.json").write_text(json.dumps(protocol,indent=2))
 m=make("pretrained");original=[p.detach().clone() for p in m.encoder.proj_out[0].parameters()];r=np.load(ROOT/"phase72/readout.npz");fitted=[torch.tensor(r["coef"],device="cuda"),torch.tensor(r["bias"],device="cuda")]
 params=[p for p in m.parameters() if p.requires_grad];assert sum(p.numel() for p in params)==9600;initial=[p.detach().clone() for p in params];state=snapshot(m);radii=[.1*max(float(v.square().mean().sqrt()),1)*v.numel()**.5 for v in initial]
 cw=torch.tensor(json.loads((ROOT/"phase36/mixed_class_weights.json").read_text())["balanced_weights"],device="cuda",dtype=torch.float64)
 def restore():
  with torch.no_grad():
   for p,v in zip(params,initial):p.copy_(v)
 def head(vals):
  with torch.no_grad():
   for p,v in zip(m.encoder.proj_out[0].parameters(),vals):p.copy_(v)
 def lossfn(x,mask):
  pred=m.encoder(x,mask)
  return ((pred-x).square()*mask[...,None]).sum()/(mask.sum()*200)
 for sid in protocol["controllers"]:
  dest=OUT73/f"s{sid:02d}.json"
  if dest.exists():continue
  times=np.load(ROOT/"phase48/masked"/f"s{sid:02d}_gradients.npz")["times"];rows=[]
  for t0 in times:
   t=int(t0);restore();head(original);x,y=sparse(sid,t);assert bool((y>=0).all())
   with torch.no_grad():base=m(x).double().log_softmax(1);bp=base.exp().cpu().numpy();before=-base[torch.arange(5,device="cuda"),y]
   gen=torch.Generator(device="cpu").manual_seed(73000+sid*100000+t);mask=(torch.rand((1,2,30),generator=gen)<.5).cuda();assert 0<int(mask.sum())<60
   gradients={};losses={};norms={};clips={}
   for variant,values in [("refitted",fitted),("original",original)]:
    head(values);loss=lossfn(x[:1],mask);g=torch.autograd.grad(loss,params);norm=torch.sqrt(sum(v.square().sum() for v in g));assert torch.isfinite(loss) and torch.isfinite(norm);gradients[variant]=g;losses[variant]=float(loss.detach());norms[variant]=norm.detach();clips[variant]=torch.clamp(1/(norm+1e-6),max=1.)
   for lr in protocol["learning_rates"]:
    for variant in protocol["variants"]:
     restore();key="original" if variant.startswith("original") else "refitted";head(original if key=="original" else fitted);mult=clips[key]
     if variant=="original_normmatched":mult=clips["refitted"]*norms["refitted"]/norms["original"].clamp_min(1e-20)
     with torch.no_grad():
      for p,v,g,radius in zip(params,initial,gradients[key],radii):
       d=-lr*mult*g;p.copy_(v+d*torch.clamp(radius/d.norm().clamp_min(1e-12),max=1.))
      actual=float(torch.sqrt(sum((p-v).square().sum() for p,v in zip(params,initial))));assert actual<=lr+1e-5
      post=m(x).double().log_softmax(1);wp=post.exp().cpu().numpy();after=-post[torch.arange(5,device="cuda"),y];postloss=float(lossfn(x[:1],mask))
     b=bp[1:].argmax(1);w=wp[1:].argmax(1);truth=y[1:].cpu().numpy()
     rows.append(dict(subject=sid,t=t,lr=lr,variant=variant,masked_patches=int(mask.sum()),aux_loss_before=losses[key],aux_loss_after=postloss,gradient_norm=float(norms[key]),actual_update_norm=actual,future_ce_gain=float((before[1:]-after[1:]).mean()),balanced_ce_gain=float(((before[1:]-after[1:])*cw[y[1:]]).mean()),current_ce_gain=float(before[0]-after[0]),future_labels=truth.tolist(),future_base_probabilities=bp[1:].tolist(),future_write_probabilities=wp[1:].tolist(),corrected=int(((b!=truth)&(w==truth)).sum()),harmed=int(((b==truth)&(w!=truth)).sum()),argmax_changes=int((b!=w).sum())))
  restore();head(original);assert all(torch.equal(v.cpu(),state[n]) for n,v in m.state_dict().items())
  dest.write_text(json.dumps(dict(subject=sid,state_restored_exact=True,rows=rows),indent=2));print("WAVEFORM WRITE",sid,len(rows),flush=True)
 (OUT73/"complete.json").write_text(json.dumps(dict(complete=True,people=4,rows=2304)))
if __name__=="__main__":run()
