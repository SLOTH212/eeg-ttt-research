from phase86_meta_waveform import *
from phase77_predict import SpectralNet,context
import datetime
O=ROOT/"phase104";O.mkdir(exist_ok=True)
def run():
 old=lock();teacherpath=ROOT/"phase4/results/context11_mixed_17/model.pt"
 protocol=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),purpose="Fixed causal teacher-generated current loss as a source-only fresh-WRITE diagnostic, with direct-teacher and equal-probability static controls.",controllers=old["controllers"],validation=old["held_source_validation"],teacher=str(teacherpath),teacher_sha256=hashlib.sha256(teacherpath.read_bytes()).hexdigest(),teacher_scope="Existing single seed17 context11 spectral classifier, same10source trainpeople but moretrainingbudget and current+10pastwindow context. Extra architecture/history/compute disclosed; no new fitting or unique labels.",student=old["initial_classifier"],student_sha256=old["classifier_sha256"],variants=["teacher_soft_clip","teacher_hard_clip","teacher_soft_matched"],primary="teacher_soft_clip at LR.01 clipnorm1,9600LN, current teacher probabilities only; reset everypulse; exactphase91times.",secondary="Hard teacherargmax and softloss normmatched to constrainedwave phase91; no selection amongvariants.",static_controls="Future-arrival predictions of unchanged student, unchanged teacher, and .5teacher+.5student; no future teacher output enters currentWRITE.",useful_write_gate="Both sourcegroups: meanfutureCE>0,>=3/4positive people, balancedCE>=0, corrected>=harmed and corrected>0.",online_extension_gate="Usefulwrite plus meanfutureCE and futureaccuracy at least as good as directteacher and equal staticblend in bothgroups; otherwise do not add online or target runs.",scope="Allsourcepeople previouslyobserved; no reconstruction-validity claim, no threshold/LRsearch, no new independent validation. Sourcevalidation usedonlyto evaluate fixed methods.")
 if (O/"protocol.json").exists():
  p=json.loads((O/"protocol.json").read_text());protocol["created_utc"]=p["created_utc"];assert p==protocol
 else:
  (O/"protocol.json").write_text(json.dumps(protocol,indent=2));(O/"protocol.sha256").write_text(hashlib.sha256((O/"protocol.json").read_bytes()).hexdigest())
 ck=torch.load(teacherpath,map_location="cpu",weights_only=False);teacher=SpectralNet(ck["dim"]);teacher.load_state_dict(ck["state"]);teacher.cuda().eval().requires_grad_(False);tsnap=snapshot(teacher)
 m=meta_model();named=ln_named(m);params=[p for n,p in named];initial=[p.detach().clone() for p in params];state=snapshot(m);cw=torch.tensor(json.loads((ROOT/"phase36/mixed_class_weights.json").read_text())["balanced_weights"],device="cuda",dtype=torch.float64)
 def restore():
  with torch.no_grad():
   for p,v in zip(params,initial):p.copy_(v)
 audits=[]
 for group,sids in [("controllers",old["controllers"]),("validation",old["held_source_validation"])]:
  for sid in sids:
   dest=O/f"s{sid:02d}.json"
   if dest.exists():continue
   folder=ROOT/("data/processed" if sid<20 else "phase2/processed");features=np.load(folder/f"s{sid:02d}_meta.npz")["features"];ct=context(features);assert np.array_equal(ct[:65],context(features[:65]))
   with torch.no_grad():tp=torch.cat([teacher(torch.from_numpy(ct[a:a+128]).cuda()).double().log_softmax(1).exp() for a in range(0,len(ct),128)])
   legacy=(np.load(ROOT/f"phase79/predictions/source_s{sid:02d}_native_raw.npz")["context11_seed17"] if sid in old["held_source_validation"] else np.load(ROOT/f"phase4/results/context11_mixed_17/predictions_s{sid:02d}.npz")["frozen"]);err=float(abs(tp.cpu().numpy()-legacy).max());assert err<2e-5 and np.array_equal(tp.argmax(1).cpu().numpy(),legacy.argmax(1));assert (tp>0).all()
   previous=json.loads((ROOT/"phase91"/f"s{sid:02d}.json").read_text());refs={r["t"]:r for r in previous["rows"] if r["variant"]=="future"};rows=[]
   for ii,t in enumerate(previous["times"]):
    restore();x,y=sparse(sid,t);lp=m(x).double().log_softmax(1);before=-lp[torch.arange(5,device="cuda"),y];bp=lp.detach().exp().cpu().numpy();assert abs(bp[1:]-np.array(refs[t]["future_base_probabilities"])).max()<1e-10
    q=tp[t].detach();soft=-(q*lp[0]).sum();hard=-lp[0,q.argmax()];gradients={"soft":torch.autograd.grad(soft,params,retain_graph=True),"hard":torch.autograd.grad(hard,params)};norms={k:torch.sqrt(sum(v.square().sum() for v in g)).detach() for k,g in gradients.items()}
    if ii==0:
     pert=x.clone();pert[1:]*=3
     with torch.no_grad():first=m(pert).double().log_softmax(1)[0];assert torch.equal(first,lp[0].detach())
    for variant in protocol["variants"]:
     restore();which=variant.split("_")[1];g=gradients[which];norm=norms[which];assert torch.isfinite(norm) and norm>0;scale=refs[t]["actual_update_norm"]/norm if variant.endswith("matched") else .01*torch.clamp(1/(norm+1e-6),max=1.)
     with torch.no_grad():
      for p,v,d in zip(params,initial,g):p.copy_(v-scale*d)
      actual=float(torch.sqrt(sum((p-v).square().sum() for p,v in zip(params,initial))));assert actual<=.01001;afterlp=m(x).double().log_softmax(1);wp=afterlp.exp().cpu().numpy();after=-afterlp[torch.arange(5,device="cuda"),y]
     yy=y[1:].cpu().numpy();b=bp[1:].argmax(1);w=wp[1:].argmax(1)
     rows.append(dict(group=group,subject=sid,t=t,variant=variant,actual_update_norm=actual,current_teacher_probabilities=q.cpu().numpy().tolist(),current_teacher_correct=bool(int(q.argmax())==int(y[0])),current_ce_gain=float(before.detach()[0]-after[0]),future_ce_gain=float((before.detach()[1:]-after[1:]).mean()),balanced_ce_gain=float(((before.detach()[1:]-after[1:])*cw[y[1:]]).mean()),future_labels=yy.tolist(),future_base_probabilities=bp[1:].tolist(),future_write_probabilities=wp[1:].tolist(),future_teacher_probabilities=tp[t+1:t+5].cpu().numpy().tolist(),corrected=int(((b!=yy)&(w==yy)).sum()),harmed=int(((b==yy)&(w!=yy)).sum()),argmax_changes=int((b!=w).sum())))
   restore();assert all(torch.equal(v.cpu(),state[n]) for n,v in m.state_dict().items()) and all(torch.equal(v.cpu(),tsnap[n]) for n,v in teacher.state_dict().items());dest.write_text(json.dumps(dict(subject=sid,group=group,rows=rows,state_restored_exact=True,teacher_legacy_probability_error=err,future_input_cannot_change_current_student_logit=True),indent=2));audits.append(dict(subject=sid,teacher_legacy_error=err));print("TEACHER WRITE",group,sid,len(rows),flush=True)
 assert all((O/f"s{s:02d}.json").exists() for s in old["controllers"]+old["held_source_validation"])
 (O/"audit.json").write_text(json.dumps(dict(status="PASS",pulse_rows=1536,teacher_causal_context=True,current_student_logit_future_perturbation_exact=True,states_restored_exact=True,teacher_already_source_trained=True),indent=2));print("COMPLETE",flush=True)
if __name__=="__main__":
 with sdpa_kernel(SDPBackend.MATH):run()
