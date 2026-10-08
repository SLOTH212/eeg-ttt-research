from phase86_meta_waveform import *
import pandas as pd
O=ROOT/"phase88";O.mkdir(exist_ok=True)
def run():
 assert json.loads((ROOT/"phase87/audit.json").read_text())["status"]=="PASS";assert not json.loads((ROOT/"phase87/decision.json").read_text())["advance"]
 pr=lock();protocol=dict(purpose="Posthoc mechanism falsification: input-dependent meta-write versus source-fixed parameter correction with same learned head and label budget.",meta_subjects=pr["meta_subjects"],evaluation_groups={"controllers":pr["controllers"],"validation":pr["held_source_validation"]},heads=["future","current"],variants=["source_mean","source_mean_normmatched_current"],fixed_update="Mean of actual lr.01 clip1 fresh source updates; equal-person weighting over exact317 meta training pulses; no extra labels.",normmatched="Source mean direction scaled to observed current-window update norm already recorded in phase87; causal magnitude, fixed direction.",comparison="Same head dynamic phase87 writes, same people/times/masks/base classifier. Not deployment of rejected phase87 head; no change to prior failed advance.",label_cost={"future":1268,"current":317},selection="No parameter or threshold search; report all source groups and heads.",parent_sha256=hashlib.sha256((ROOT/"phase86/protocol.json").read_bytes()).hexdigest())
 (O/"protocol.json").write_text(json.dumps(protocol,indent=2));m=meta_model();named=ln_named(m);params=[p for n,p in named];initial=[p.detach().clone() for p in params];state=snapshot(m);shapes=[list(p.shape) for p in params];sizes=[p.numel() for p in params]
 (O/"parameters.json").write_text(json.dumps(dict(names=[n for n,p in named],shapes=shapes,sizes=sizes)))
 def restore():
  with torch.no_grad():
   for p,v in zip(params,initial):p.copy_(v)
 def apply(vector):
  at=0
  with torch.no_grad():
   for p,v in zip(params,initial):n=p.numel();p.copy_(v+vector[at:at+n].view_as(p));at+=n
 def actualvector():return torch.cat([(p-v).flatten() for p,v in zip(params,initial)])
 means={}
 for which in protocol["heads"]:
  m.encoder.proj_out[0].load_state_dict(torch.load(OUT86/which/"head.pt",map_location="cuda",weights_only=False)["head"]);rows=[];subjects=[];timesall=[];counts={}
  for sid in pr["meta_subjects"]:
   times=np.load(ROOT/"phase48/masked"/f"s{sid:02d}_gradients.npz")["times"];counts[sid]=len(times)
   for t0 in times:
    t=int(t0);restore();x,y=sparse(sid,t);aux=reconstruction(m,x,mask_for(sid,t));g=torch.autograd.grad(aux,params);norm=torch.sqrt(sum(v.square().sum() for v in g));clip=torch.clamp(1/(norm+1e-6),max=1.)
    with torch.no_grad():
     for p,v,gg in zip(params,initial,g):p.copy_(v-.01*clip*gg)
     vec=actualvector();assert float(vec.norm())<=.01001;rows.append(vec.cpu().numpy());subjects.append(sid);timesall.append(t)
  restore();updates=np.array(rows);subjects=np.array(subjects);weights=np.array([1/(5*counts[int(s)]) for s in subjects]);mean=np.sum(updates.astype(float)*weights[:,None],axis=0);assert np.linalg.norm(mean)>0 and np.linalg.norm(mean)<=.01001
  means[which]=torch.tensor(mean,device="cuda",dtype=torch.float32);np.savez_compressed(O/(which+"_source_directions.npz"),updates=updates,subjects=subjects,times=np.array(timesall),weights=weights,mean=mean);print("SOURCE MEAN",which,len(updates),float(np.linalg.norm(mean)),flush=True)
 cw=torch.tensor(json.loads((ROOT/"phase36/mixed_class_weights.json").read_text())["balanced_weights"],device="cuda",dtype=torch.float64)
 for group,sids in protocol["evaluation_groups"].items():
  for sid in sids:
   original=json.loads((ROOT/"phase87"/f"s{sid:02d}.json").read_text());lookup={(r["t"],r["variant"]):r for r in original["rows"]};rows=[]
   for t0 in original["times"]:
    t=int(t0);restore();x,y=sparse(sid,t)
    with torch.no_grad():blp=m(x).double().log_softmax(1);bp=blp.exp().cpu().numpy();before=-blp[torch.arange(5,device="cuda"),y]
    for which in protocol["heads"]:
     prior=lookup[t,which];assert abs(bp[1:]-np.array(prior["future_base_probabilities"])).max()<1e-10;v=means[which]
     for variant in protocol["variants"]:
      vector=v if variant=="source_mean" else v*(prior["actual_update_norm"]/float(v.norm()));apply(vector)
      with torch.no_grad():
       anorm=float(actualvector().norm());plp=m(x).double().log_softmax(1);wp=plp.exp().cpu().numpy();after=-plp[torch.arange(5,device="cuda"),y]
      if variant!="source_mean":assert abs(anorm-prior["actual_update_norm"])<1e-6
      truth=y[1:].cpu().numpy();b=bp[1:].argmax(1);w=wp[1:].argmax(1)
      rows.append(dict(group=group,subject=sid,t=t,head=which,variant=variant,actual_update_norm=anorm,dynamic_update_norm=prior["actual_update_norm"],future_ce_gain=float((before[1:]-after[1:]).mean()),balanced_ce_gain=float(((before[1:]-after[1:])*cw[y[1:]]).mean()),future_labels=truth.tolist(),future_base_probabilities=bp[1:].tolist(),future_write_probabilities=wp[1:].tolist(),dynamic_future_ce_gain=prior["future_ce_gain"],dynamic_balanced_ce_gain=prior["balanced_ce_gain"],corrected=int(((b!=truth)&(w==truth)).sum()),harmed=int(((b==truth)&(w!=truth)).sum()),argmax_changes=int((b!=w).sum())))
   restore();(O/f"s{sid:02d}.json").write_text(json.dumps(dict(subject=sid,group=group,rows=rows),indent=2));print("FIXED COMPARISON",group,sid,len(rows),flush=True)
 m.encoder.proj_out[0].load_state_dict({k.removeprefix("encoder.proj_out.0."):v for k,v in state.items() if k.startswith("encoder.proj_out.0.")});assert all(torch.equal(v.cpu(),state[n]) for n,v in m.state_dict().items())
 (O/"complete.json").write_text(json.dumps(dict(status="PASS",rows=2048,source_pulses_perhead=317,state_restored_exact=True,no_additional_label_supervision=True)))
if __name__=="__main__":
 with sdpa_kernel(SDPBackend.MATH):run()
