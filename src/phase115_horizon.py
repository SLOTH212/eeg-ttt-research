from phase86_meta_waveform import *
from phase77_predict import SpectralNet,context
import datetime
O=ROOT/"phase115";O.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run():
 old=lock();models=json.loads((ROOT/"phase79/protocol.json").read_text())["models"];specs=[]
 for seed in [17,29,43]:
  sp=next(s for s in models if s["model"]=="cbramod" and s["seed"]==seed); tp=ROOT/f"phase4/results/context11_mixed_{seed}/model.pt"
  specs.append(dict(seed=seed,student=sp,teacher=str(tp),teacher_sha256=sha(tp)))
 people=old["controllers"]+old["held_source_validation"]; eligible={}
 for sid in people:
  folder=ROOT/("data/processed" if sid<20 else "phase2/processed")
  n=len(np.load(folder/f"s{sid:02d}_x.npy",mmap_mode="r"));times=json.loads((ROOT/"phase91"/f"s{sid:02d}.json").read_text())["times"]
  eligible[str(sid)]=dict(n=n,included=[t for t in times if t+60<n],excluded_insufficient_duration=[t for t in times if t+60>=n])
 protocol=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),models=specs,eligibility=eligible,controllers=old["controllers"],validation=old["held_source_validation"],primary="Descriptive paired fixed horizon decay of single softteacherWRITE: cumulative next1,4,10,20,60windows and disjoint1-4,5-10,11-20,21-60. No hypothesis significance gate or method tuning.",update="Same phase104/114 current softteacherCE,9600LN,lr.01,clip1,coldrestore eachpulse; first5batch preserved for legacy gradient compatibility.",controls=["frozen_student","teacher_on_arrival","half_blend_on_arrival","current_teacher_held_constant"],label_use="Only retrospective scored-window evaluation and same-current-stage vs changed-stage strata. Unknown labels excluded from scoring only; not from input or eligibility.",scope="All8sourcepeople previouslyobserved. Fixed 60future windows (30minutes), duration-only eligibility locked before new inference. No continuous online updates, no targets or checkpoint selection; cannot reopen prior failed competitive gate.",aggregation="Seed metric means within person, then equalpeople. Same duration-eligible pulse set across horizons. Scored counts may vary due unknown labels. Overlapping pulse outcomes not unique independent observations.",cpu_ids=list(range(8,24)),gpu=True)
 if (O/"protocol.json").exists():
  p=json.loads((O/"protocol.json").read_text());protocol["created_utc"]=p["created_utc"];assert p==protocol
 else:(O/"protocol.json").write_text(json.dumps(protocol,indent=2));(O/"protocol.sha256").write_text(sha(O/"protocol.json"))
 total=0; audits=[]
 for sp in specs:
  seed=sp["seed"]; out=O/f"seed{seed}";out.mkdir(exist_ok=True)
  path=Path(sp["student"]["path"]); assert sha(path)==sp["student"]["sha256"]
  m=meta_model();ck=torch.load(path,map_location="cpu",weights_only=False);m.load_state_dict(ck["state"]);m.eval();params=[p for n,p in ln_named(m)];initial=[p.detach().clone() for p in params];state=snapshot(m)
  assert sum(p.numel() for p in params)==9600
  tpfile=Path(sp["teacher"]);assert sha(tpfile)==sp["teacher_sha256"];tc=torch.load(tpfile,map_location="cpu",weights_only=False)
  teacher=SpectralNet(tc["dim"]);teacher.load_state_dict(tc["state"]);teacher.cuda().eval().requires_grad_(False);tstate=snapshot(teacher)
  def restore():
   with torch.no_grad():
    for p,v in zip(params,initial):p.copy_(v)
  for sid in people:
   dest=out/f"s{sid:02d}.npz"; aud=out/f"s{sid:02d}_audit.json"
   if dest.exists() and aud.exists():
    a=json.loads(aud.read_text());assert sha(dest)==a["artifact_sha256"];audits.append(a);total+=a["pulses"];continue
   folder=ROOT/("data/processed" if sid<20 else "phase2/processed");raw=np.load(folder/f"s{sid:02d}_x.npy",mmap_mode="r");meta=np.load(folder/f"s{sid:02d}_meta.npz");features=meta["features"];labels=meta["y"];ct=context(features)
   with torch.no_grad():tp=torch.cat([teacher(torch.from_numpy(ct[a:a+128]).cuda()).double().log_softmax(1).exp() for a in range(0,len(ct),128)])
   legacyfile=ROOT/"phase104"/f"s{sid:02d}.json" if seed==17 else ROOT/"phase114"/f"seed{seed}"/f"s{sid:02d}.json"
   legacy={r["t"]:r for r in json.loads(legacyfile.read_text())["rows"] if r["variant"]=="teacher_soft_clip"}
   times=eligible[str(sid)]["included"];bp=[];wp=[];tt=[];ys=[];ns=[];max_legacy=0.;max_batch=0.;argmax_batch=0
   for t in times:
    restore();x=torch.tensor(prep(np.array(raw[t:t+61]),[.1,45]),device="cuda")
    with torch.no_grad():fullbase=m(x).double().log_softmax(1).exp().cpu().numpy()
    lp=m(x[:5]).double().log_softmax(1);base5=lp.detach().exp().cpu().numpy();q=tp[t].detach();loss=-(q*lp[0]).sum();grads=torch.autograd.grad(loss,params);norm=torch.sqrt(sum(g.square().sum() for g in grads));assert torch.isfinite(norm)
    scale=.01*torch.clamp(1/(norm+1e-6),max=1.)
    with torch.no_grad():
     for p,v,g in zip(params,initial,grads):p.copy_(v-scale*g)
     actual=float(torch.sqrt(sum((p-v).square().sum() for p,v in zip(params,initial))));assert actual<=.01001
     fullwrite=m(x).double().log_softmax(1).exp().cpu().numpy();write5=m(x[:5]).double().log_softmax(1).exp().cpu().numpy()
    lr=legacy[t];err=max(abs(base5[1:]-np.array(lr["future_base_probabilities"])).max(),abs(write5[1:]-np.array(lr["future_write_probabilities"])).max(),abs(tp[t+1:t+5].cpu().numpy()-np.array(lr["future_teacher_probabilities"])).max())
    assert err<2e-5 and abs(actual-lr["actual_update_norm"])<1e-6;max_legacy=max(max_legacy,float(err))
    be=max(abs(fullbase[:5]-base5).max(),abs(fullwrite[:5]-write5).max());assert be<2e-5;max_batch=max(max_batch,float(be));argmax_batch+=int((fullbase[:5].argmax(1)!=base5.argmax(1)).sum()+(fullwrite[:5].argmax(1)!=write5.argmax(1)).sum())
    fullbase[:5]=base5;fullwrite[:5]=write5
    bp.append(fullbase);wp.append(fullwrite);tt.append(tp[t:t+61].cpu().numpy());ys.append(labels[t:t+61]);ns.append(actual)
   restore();assert all(torch.equal(v.cpu(),state[k]) for k,v in m.state_dict().items()) and all(torch.equal(v.cpu(),tstate[k]) for k,v in teacher.state_dict().items())
   np.savez_compressed(dest,times=np.array(times),labels=np.array(ys),base=np.array(bp),write=np.array(wp),teacher=np.array(tt),norms=np.array(ns))
   a=dict(seed=seed,subject=sid,pulses=len(times),state_restored_exact=True,max_legacy_probability_error=max_legacy,max_batch_probability_error=max_batch,batch_argmax_differences=argmax_batch,artifact_sha256=sha(dest),legacy_sha256=sha(legacyfile));aud.write_text(json.dumps(a,indent=2));audits.append(a);total+=len(times);print("HORIZON",seed,sid,len(times),"legacy",max_legacy,"batch",max_batch,flush=True)
  del m,teacher;torch.cuda.empty_cache()
 (O/"audit.json").write_text(json.dumps(dict(status="PASS",pulses=total,audits=audits,future_windows_per_pulse=60,no_online_or_target_rollout=True),indent=2));print("COMPLETE",total,flush=True)
if __name__=="__main__":
 with sdpa_kernel(SDPBackend.MATH):run()
