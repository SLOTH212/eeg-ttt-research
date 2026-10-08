from phase86_meta_waveform import *
import pandas as pd
O=ROOT/"phase87";O.mkdir(exist_ok=True)
def run():
 pr=lock()
 for which in ["future","current","reconstruction"]:assert (OUT86/which/"complete.json").exists()
 protocol=dict(parent_protocol_sha256=hashlib.sha256((OUT86/"protocol.json").read_bytes()).hexdigest(),models={w:hashlib.sha256((OUT86/w/"head.pt").read_bytes()).hexdigest() for w in ["future","current","reconstruction"]},controllers=pr["controllers"],validation=pr["held_source_validation"],variants=["initial","future","current","reconstruction","initial_normmatched_future"],lr=.01,primary="future",times="controllers from phase48, validation exact phase75 times",mask=pr["mask"],advance=pr["advance"],extra_target_labels=0)
 (O/"protocol.json").write_text(json.dumps(protocol,indent=2))
 m=meta_model();named=ln_named(m);params=[p for n,p in named];initial=[p.detach().clone() for p in params];state=snapshot(m);heads={"initial":{k:v.detach().clone() for k,v in m.encoder.proj_out[0].state_dict().items()}}
 for which in ["future","current","reconstruction"]:heads[which]=torch.load(OUT86/which/"head.pt",weights_only=False,map_location="cuda")["head"]
 cw=torch.tensor(json.loads((ROOT/"phase36/mixed_class_weights.json").read_text())["balanced_weights"],device="cuda",dtype=torch.float64)
 def restore():
  with torch.no_grad():
   for p,v in zip(params,initial):p.copy_(v)
 def sethead(which):m.encoder.proj_out[0].load_state_dict(heads[which])
 for group,sids in [("controllers",pr["controllers"]),("validation",pr["held_source_validation"])]:
  for sid in sids:
   dest=O/f"s{sid:02d}.json"
   if dest.exists():continue
   times=np.load(ROOT/"phase48/masked"/f"s{sid:02d}_gradients.npz")["times"] if group=="controllers" else np.array(json.loads((ROOT/"phase75"/f"s{sid:02d}.json").read_text())["times"])
   rows=[]
   for t0 in times:
    t=int(t0);restore();x,y=sparse(sid,t);mask=mask_for(sid,t)
    with torch.no_grad():blp=m(x).double().log_softmax(1);bp=blp.exp().cpu().numpy();before=-blp[torch.arange(5,device="cuda"),y]
    gradients={};norms={};clips={};losses={}
    for which in ["initial","future","current","reconstruction"]:
     sethead(which);aux=reconstruction(m,x,mask);g=torch.autograd.grad(aux,params);norm=torch.sqrt(sum(v.square().sum() for v in g));assert torch.isfinite(norm) and norm>0
     gradients[which]=g;norms[which]=norm.detach();clips[which]=torch.clamp(1/(norm+1e-6),max=1.).detach();losses[which]=float(aux.detach())
    for variant in protocol["variants"]:
     restore();which="initial" if variant.startswith("initial") else variant;sethead(which);mult=clips[which]
     if variant=="initial_normmatched_future":mult=clips["future"]*norms["future"]/norms["initial"]
     with torch.no_grad():
      for p,v,g in zip(params,initial,gradients[which]):p.copy_(v-.01*mult*g)
      norm=float(torch.sqrt(sum((p-v).square().sum() for p,v in zip(params,initial))));assert norm<=.01001
      afterlp=m(x).double().log_softmax(1);wp=afterlp.exp().cpu().numpy();after=-afterlp[torch.arange(5,device="cuda"),y];postloss=float(reconstruction(m,x,mask))
     b=bp[1:].argmax(1);w=wp[1:].argmax(1);yy=y[1:].cpu().numpy()
     rows.append(dict(group=group,subject=sid,t=t,variant=variant,aux_loss_before=losses[which],aux_loss_after=postloss,gradient_norm=float(norms[which]),actual_update_norm=norm,current_ce_gain=float(before[0]-after[0]),future_ce_gain=float((before[1:]-after[1:]).mean()),balanced_ce_gain=float(((before[1:]-after[1:])*cw[y[1:]]).mean()),future_labels=yy.tolist(),future_base_probabilities=bp[1:].tolist(),future_write_probabilities=wp[1:].tolist(),corrected=int(((b!=yy)&(w==yy)).sum()),harmed=int(((b==yy)&(w!=yy)).sum()),argmax_changes=int((b!=w).sum())))
   restore();sethead("initial");assert all(torch.equal(v.cpu(),state[n]) for n,v in m.state_dict().items());dest.write_text(json.dumps(dict(subject=sid,group=group,times=times.tolist(),state_restored_exact=True,rows=rows),indent=2));print("PULSES",group,sid,len(rows),flush=True)
 # Reconstruction remains a required auxiliary validity check, separately from task benefit.
 vals=[];checks=[];(O/"validation_cases").mkdir(exist_ok=True)
 @torch.no_grad()
 def recon(x,mask):
  return np.concatenate([m.encoder(torch.tensor(x[i:i+8],device="cuda"),torch.tensor(mask[i:i+8],device="cuda")).cpu().numpy() for i in range(0,len(x),8)])
 for sid in pr["held_source_validation"]:
  x=np.load(ROOT/"phase62/cases"/f"s{sid:02d}_existing_input.npz")["x"];mask=np.load(ROOT/"phase62/cases"/f"s{sid:02d}_existing_bernoulli.npz")["mask"];sethead("initial")
  with torch.no_grad():base_logits=m(torch.tensor(x[:8],device="cuda")).clone()
  for which in heads:
   sethead(which);p=recon(x,mask);sh=recon(np.roll(x,48,axis=0),mask);wm=[]
   for i in range(len(x)):
    target=x[i][mask[i]].astype(float);pred=p[i][mask[i]].astype(float);sp=sh[i][mask[i]].astype(float);wm.append([np.square(pred-target).mean(),np.square(target).mean(),np.square(sp-target).mean()])
   wm=np.array(wm);mean=wm.mean(0)
   with torch.no_grad():
    xt=torch.tensor(x[:2],device="cuda");mt=torch.tensor(mask[:2],device="cuda");p0=m.encoder(xt,mt);changed=xt.clone();changed[mt]+=3.17;leak=float(abs(m.encoder(changed,mt)-p0).max());assert leak==0
    err=float(abs(m(torch.tensor(x[:8],device="cuda"))-base_logits).max());assert err==0
   np.savez_compressed(O/"validation_cases"/f"{which}_s{sid:02d}.npz",target=x[mask],prediction=p[mask],shuffle_prediction=sh[mask],mask=mask,window_metrics=wm)
   vals.append(dict(subject=sid,variant=which,mse=mean[0],zero_mse=mean[1],shuffle_mse=mean[2],ratio_to_zero=mean[0]/mean[1],validity_pass=bool(mean[0]<mean[1] and mean[0]<mean[2])))
   checks.append(dict(subject=sid,variant=which,masked_target_perturbation_error=leak,classifier_logit_error=err));print("VALIDITY",vals[-1],flush=True)
 restore();sethead("initial");assert all(torch.equal(v.cpu(),state[n]) for n,v in m.state_dict().items())
 pd.DataFrame(vals).to_csv(O/"reconstruction_validation.csv",index=False);(O/"execution_audit.json").write_text(json.dumps(dict(status="PASS",model_restored_exact=True,classification_head_independent_of_auxiliary=True,reconstruction_checks=checks),indent=2));(O/"complete.json").write_text(json.dumps(dict(people=8,pulse_rows=2560,reconstruction_cases=16)))
if __name__=="__main__":
 with sdpa_kernel(SDPBackend.MATH):run()
