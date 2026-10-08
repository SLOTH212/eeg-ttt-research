from phase86_meta_waveform import *
from phase77_predict import SpectralNet,context
import datetime
O=ROOT/"phase114";O.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run():
 old=lock();models=json.loads((ROOT/"phase79/protocol.json").read_text())["models"];specs=[]
 for seed in [29,43]:
  student=next(s for s in models if s["model"]=="cbramod" and s["seed"]==seed);teacher=ROOT/f"phase4/results/context11_mixed_{seed}/model.pt";specs.append(dict(seed=seed,student=student,teacher=str(teacher),teacher_sha256=sha(teacher)))
 protocol=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),purpose="Two additional fixed classifier/teacher seeds for source teacher-WRITE robustness; do not reopen phase104failedonlinegate.",models=specs,seed17_reference={f"s{s:02d}":sha(ROOT/"phase104"/f"s{s:02d}.json") for s in old["controllers"]+old["held_source_validation"]},controllers=old["controllers"],validation=old["held_source_validation"],method="Exactphase104primary softcurrentteacherCE, LN9600,lr.01,clip1; freshreset each exactphase91pulse. No hard/normmatched secondary replication.",static_controls="Same-seed frozenstudent, directcontext11teacher and .5probabilityblend, with futureteacher onlyatfuture-arrival/evaluation.",aggregation="Mean metricswithinperson across17/29/43; no probabilityensemble or seedpseudoreplication. Correction/harm counts averagedacrossseeds perperson, not newunique events.",scope="Previouslyobserved8sourcepeople, no newtraining, no targetinference or onlineextension regardlessoutcome; keep priorfailed primarycompetitivegate.",cpu_ids=list(range(8,24)),gpu=True)
 if (O/"protocol.json").exists():
  p=json.loads((O/"protocol.json").read_text());protocol["created_utc"]=p["created_utc"];assert p==protocol
 else:(O/"protocol.json").write_text(json.dumps(protocol,indent=2));(O/"protocol.sha256").write_text(sha(O/"protocol.json"))
 cw=torch.tensor(json.loads((ROOT/"phase36/mixed_class_weights.json").read_text())["balanced_weights"],device="cuda",dtype=torch.float64);all_audits=[]
 for sp in specs:
  seed=sp["seed"];out=O/f"seed{seed}";out.mkdir(exist_ok=True);studentpath=Path(sp["student"]["path"]);assert sha(studentpath)==sp["student"]["sha256"] and sha(Path(sp["teacher"]))==sp["teacher_sha256"]
  m=meta_model();ck=torch.load(studentpath,map_location="cpu",weights_only=False);m.load_state_dict(ck["state"]);m.eval();params=[p for n,p in ln_named(m)];initial=[p.detach().clone() for p in params];snapshot0=snapshot(m);assert all(torch.equal(v,snapshot0[k]) for k,v in ck["state"].items())
  tc=torch.load(sp["teacher"],map_location="cpu",weights_only=False);teacher=SpectralNet(tc["dim"]);teacher.load_state_dict(tc["state"]);teacher.cuda().eval().requires_grad_(False);ts=snapshot(teacher)
  def restore():
   with torch.no_grad():
    for p,v in zip(params,initial):p.copy_(v)
  for group,sids in [("controllers",old["controllers"]),("validation",old["held_source_validation"])]:
   for sid in sids:
    dest=out/f"s{sid:02d}.json"
    if dest.exists():continue
    folder=ROOT/("data/processed" if sid<20 else "phase2/processed");ff=np.load(folder/f"s{sid:02d}_meta.npz")["features"];ct=context(ff);assert np.array_equal(ct[:65],context(ff[:65]))
    with torch.no_grad():tp=torch.cat([teacher(torch.from_numpy(ct[a:a+128]).cuda()).double().log_softmax(1).exp() for a in range(0,len(ct),128)])
    legacy=np.load(ROOT/"phase79/predictions"/f"source_s{sid:02d}_native_raw.npz")[f"context11_seed{seed}"] if group=="validation" else np.load(ROOT/f"phase4/results/context11_mixed_{seed}/predictions_s{sid:02d}.npz")["frozen"]
    teacher_error=float(abs(tp.cpu().numpy()-legacy).max());assert teacher_error<2e-5 and np.array_equal(tp.argmax(1).cpu().numpy(),legacy.argmax(1));times=json.loads((ROOT/"phase91"/f"s{sid:02d}.json").read_text())["times"];rows=[]
    for ii,t in enumerate(times):
     restore();x,y=sparse(sid,t);lp=m(x).double().log_softmax(1);before=-lp[torch.arange(5,device="cuda"),y];bp=lp.detach().exp().cpu().numpy()
     if group=="validation":
      gold=np.load(ROOT/"phase79/predictions"/f"source_s{sid:02d}_native_raw.npz")[f"cbramod_seed{seed}"][t:t+5];assert abs(bp-gold).max()<2e-5
     q=tp[t].detach();loss=-(q*lp[0]).sum();g=torch.autograd.grad(loss,params);norm=torch.sqrt(sum(v.square().sum() for v in g));assert norm>0 and torch.isfinite(norm);scale=.01*torch.clamp(1/(norm+1e-6),max=1.)
     if ii==0:
      pert=x.clone();pert[1:]*=3
      with torch.no_grad():assert torch.equal(m(pert).double().log_softmax(1)[0],lp[0].detach())
     with torch.no_grad():
      for p,v,d in zip(params,initial,g):p.copy_(v-scale*d)
      actual=float(torch.sqrt(sum((p-v).square().sum() for p,v in zip(params,initial))));assert actual<=.01001;afterlp=m(x).double().log_softmax(1);after=-afterlp[torch.arange(5,device="cuda"),y];wp=afterlp.exp().cpu().numpy()
     yy=y[1:].cpu().numpy();b=bp[1:].argmax(1);w=wp[1:].argmax(1);rows.append(dict(seed=seed,group=group,subject=sid,t=t,variant="teacher_soft_clip",actual_update_norm=actual,future_ce_gain=float((before.detach()[1:]-after[1:]).mean()),balanced_ce_gain=float(((before.detach()[1:]-after[1:])*cw[y[1:]]).mean()),future_labels=yy.tolist(),future_base_probabilities=bp[1:].tolist(),future_write_probabilities=wp[1:].tolist(),future_teacher_probabilities=tp[t+1:t+5].cpu().numpy().tolist(),corrected=int(((b!=yy)&(w==yy)).sum()),harmed=int(((b==yy)&(w!=yy)).sum())))
    restore();assert all(torch.equal(v.cpu(),snapshot0[k]) for k,v in m.state_dict().items()) and all(torch.equal(v.cpu(),ts[k]) for k,v in teacher.state_dict().items());dest.write_text(json.dumps(dict(seed=seed,subject=sid,group=group,rows=rows,state_restored_exact=True,teacher_legacy_probability_error=teacher_error,actual_student_checkpoint_sha256=sha(studentpath),future_input_cannot_change_current_prediction=True),indent=2));print("REPLICATED WRITE",seed,group,sid,flush=True)
  del m,teacher;torch.cuda.empty_cache()
 assert all((O/f"seed{seed}"/f"s{s:02d}.json").exists() for seed in [29,43] for s in old["controllers"]+old["held_source_validation"])
 (O/"audit.json").write_text(json.dumps(dict(status="PASS",new_pulse_writes=1024,seeds=[29,43],source_people=8,actual_checkpoint_identities_verified=True,teacher_compatibility_verified=True,states_restored_exact=True,no_online_or_target_extension=True),indent=2));print("COMPLETE",flush=True)
if __name__=="__main__":
 with sdpa_kernel(SDPBackend.MATH):run()
