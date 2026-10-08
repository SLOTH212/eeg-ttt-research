from phase86_meta_waveform import *
from phase77_predict import SpectralNet,context
from phase66_train import prepared,probabilities
import datetime
O=ROOT/"phase116";O.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run():
 train=[0,1,2,10,11,20,21,30,31,32];val=[5,15,25,35];specs=[]
 models=json.loads((ROOT/"phase79/protocol.json").read_text())["models"]
 for seed in [17,29,43]:
  sp=next(s for s in models if s["model"]=="cbramod" and s["seed"]==seed);tp=ROOT/f"phase4/results/context11_mixed_{seed}/model.pt";specs.append(dict(seed=seed,student=sp,teacher=str(tp),teacher_sha256=sha(tp)))
 methods=["true_label","teacher_soft","half_label_teacher"]
 protocol=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),train=train,validation=val,models=specs,methods=methods,parameters="Exactly same9600LNaffines as teacherWRITE; allother parameters/buffers immutable.",training=dict(epochs=5,batch=64,lr=.0001,optimizer="AdamW",weight_decay=0.,gradient_clip=1.,sampling="Same eachwindowonceperepoch, same dedicated per-seed permutations acrossmethods; equalperson lossweights. Allmethodsuse identicalscored sourcewindows; no classweights.",targets="Onehot trueclass, same-seed current+past10teacher probabilities, or fixed .5 mixture; temperature1. Teacher is trained on same source10, so training targets are in-sample, with extra historical/model/training budget.",mode="eval mode throughout, gradients enabled onlyforLN; deterministic dropout disabled, same deploymentmode asWRITE."),selection="Earliest maximum mean raw MacroF1 over4sourceval people amongepoch0..5, provided everyperson within2pp of owninitial andmeanwithin1pp. Initialepoch0 fallback. No EMA or teacher/blend selection.",controls="Frozeninitial and currentteacher/static .5blend on sourcevalidation. True_label matchedsameLN/batches/optimizer/epochs. This is offline adaptation, notTTT.",scope="Allsourcepeople previouslyobserved; no targetpredictions, no onlineupdates; no targetrollout before independentcheckpoint replay and three-seed sourceassessment.",storage="Onlyselected/final9600parameterstates plusinitialcheckpointSHA; no fullmodelduplicates or optimizers.",cpu_ids=list(range(8,24)),gpu=True)
 p=O/"protocol.json"
 if p.exists():
  old=json.loads(p.read_text());protocol["created_utc"]=old["created_utc"];assert protocol==old
 else:p.write_text(json.dumps(protocol,indent=2));(O/"protocol.sha256").write_text(sha(p))
 parts=[];ys=[];ss=[];contexts={};validmasks={}
 for sid in train:
  xx,yy=prepared(sid);good=yy>=0;parts.append(xx[good]);ys.append(yy[good]);ss.extend([sid]*int(good.sum()));validmasks[sid]=good
  folder=ROOT/("data/processed" if sid<20 else "phase2/processed");ff=np.load(folder/f"s{sid:02d}_meta.npz")["features"];contexts[sid]=context(ff)
  assert np.array_equal(contexts[sid][:65],context(ff[:65]))
 x=torch.cat(parts);del parts; y=torch.tensor(np.concatenate(ys),dtype=torch.long);ss=np.array(ss);counts={s:int((ss==s).sum()) for s in train};w=np.array([1/counts[int(s)] for s in ss]);w=torch.tensor(w/w.mean(),dtype=torch.float32)
 validation={sid:prepared(sid) for sid in val}
 for sid in val:
  folder=ROOT/("data/processed" if sid<20 else "phase2/processed");contexts[sid]=context(np.load(folder/f"s{sid:02d}_meta.npz")["features"])
 print("PREPARED",len(y),tuple(x.shape),flush=True)
 allresults=[];orders_all={}
 for sp in specs:
  seed=sp["seed"];path=Path(sp["student"]["path"]);assert sha(path)==sp["student"]["sha256"]
  teacherpath=Path(sp["teacher"]);assert sha(teacherpath)==sp["teacher_sha256"];tc=torch.load(teacherpath,map_location="cpu",weights_only=False);teacher=SpectralNet(tc["dim"]);teacher.load_state_dict(tc["state"]);teacher.cuda().eval().requires_grad_(False);teacherstate=snapshot(teacher);pred={}
  with torch.no_grad():
   for sid in train+val:
    ct=contexts[sid];pred[sid]=torch.cat([teacher(torch.from_numpy(ct[a:a+128]).cuda()).softmax(1).cpu() for a in range(0,len(ct),128)])
  assert all(torch.equal(v.cpu(),teacherstate[k]) for k,v in teacher.state_dict().items())
  qt=torch.cat([pred[sid][validmasks[sid]] for sid in train]);assert qt.shape==(len(y),5) and torch.allclose(qt.sum(1),torch.ones(len(y)),atol=1e-6,rtol=0)
  targethash=hashlib.sha256(qt.numpy().tobytes()).hexdigest();del teacher;torch.cuda.empty_cache()
  baseck=torch.load(path,map_location="cpu",weights_only=False)["state"]
  for method in methods:
   out=O/f"seed{seed}_{method}";out.mkdir(exist_ok=True)
   if (out/"complete.json").exists():
    done=json.loads((out/"complete.json").read_text());allresults.append(done);orders_all[(seed,method)]=done["order_hashes"];continue
   m=meta_model();m.load_state_dict(baseck);m.requires_grad_(False)
   for mod in m.encoder.modules():
    if isinstance(mod,nn.LayerNorm):
     for pp in mod.parameters():pp.requires_grad_(True)
   m.eval();named=ln_named(m);params=[p for n,p in named];names=[n for n,p in named];assert sum(p.numel() for p in params)==9600 and sum(p.numel() for p in m.parameters() if p.requires_grad)==9600
   initial=snapshot(m)
   def lnstate():return {n:p.detach().cpu().clone() for n,p in named}
   def evaluate():
    result=[];prob={}
    for sid,(vx,vy) in validation.items():
     pp=probabilities(m,vx);prob[sid]=pp;met=metrics(vy,pp);result.append(dict(subject=sid,macro_f1=100*met["macro_f1"],ce=met["ce"]))
    return result,prob
   baseline,baseprob=evaluate();by={v["subject"]:v["macro_f1"] for v in baseline};bn=float(np.mean(list(by.values())))
   for sid,pp in baseprob.items():
    legacy=np.load(ROOT/"phase79/predictions"/f"source_s{sid:02d}_native_raw.npz")[f"cbramod_seed{seed}"];assert abs(pp-legacy).max()<2e-5 and np.array_equal(pp.argmax(1),legacy.argmax(1))
   target=nn.functional.one_hot(y,5).float() if method=="true_label" else qt if method=="teacher_soft" else .5*nn.functional.one_hot(y,5).float()+.5*qt
   opt=torch.optim.AdamW(params,lr=.0001,weight_decay=0.);gen=torch.Generator().manual_seed(seed);seed_all(seed);best=bn;bestepoch=0;beststate=lnstate();history=[dict(epoch=0,mean_f1=bn,eligible=True,people=baseline)];orderhashes=[]
   for epoch in range(1,6):
    order=torch.randperm(len(y),generator=gen);orderhashes.append(hashlib.sha256(order.numpy().tobytes()).hexdigest());losses=[];start=time.perf_counter()
    for idx in order.split(64):
     m.eval();opt.zero_grad(set_to_none=True);lp=m(x[idx].cuda()).log_softmax(1);loss=(-torch.sum(target[idx].cuda()*lp,dim=1)*w[idx].cuda()).mean();loss.backward();norm=nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(loss) and torch.isfinite(norm);opt.step();losses.append(float(loss.detach()))
    scores,_=evaluate();mean=float(np.mean([r["macro_f1"] for r in scores]));eligible=mean>=bn-1 and all(r["macro_f1"]>=by[r["subject"]]-2 for r in scores)
    row=dict(epoch=epoch,mean_f1=mean,eligible=eligible,people=scores,loss=float(np.mean(losses)),seconds=time.perf_counter()-start);history.append(row)
    if eligible and mean>best:best=mean;bestepoch=epoch;beststate=lnstate()
    assert all(torch.equal(v.detach().cpu(),initial[k]) for k,v in m.state_dict().items() if k not in names)
    (out/"history.json").write_text(json.dumps(dict(history=history,orders=orderhashes),indent=2));print("DISTILL",seed,method,epoch,mean,eligible,flush=True)
   save(dict(ln_state=lnstate(),epoch=5,initial_checkpoint_sha256=sha(path)),out/"final5_ln.pt")
   save(dict(ln_state=beststate,epoch=bestepoch,initial_checkpoint_sha256=sha(path)),out/"selected_ln.pt")
   m.load_state_dict(beststate,strict=False);selected,sprob=evaluate();assert abs(np.mean([r["macro_f1"] for r in selected])-best)<1e-10
   for sid in val:np.savez_compressed(out/f"s{sid:02d}.npz",p=sprob[sid],base=baseprob[sid],teacher=pred[sid].numpy(),y=validation[sid][1])
   done=dict(seed=seed,method=method,selected_epoch=bestepoch,mean_f1=best,initial_f1=bn,people=selected,training_windows=len(y),window_presentations=5*len(y),target_sha256=hashlib.sha256(target.numpy().tobytes()).hexdigest(),teacher_target_sha256=targethash,order_hashes=orderhashes,initial_checkpoint_sha256=sha(path),selected_sha256=sha(out/"selected_ln.pt"),frozen_nonLN_state_exact=True,history=history)
   (out/"complete.json").write_text(json.dumps(done,indent=2));allresults.append(done);orders_all[(seed,method)]=orderhashes;del m,opt;torch.cuda.empty_cache();print("SELECTED",seed,method,bestepoch,best,flush=True)
  assert all(orders_all[(seed,m)]==orders_all[(seed,methods[0])] for m in methods)
 (O/"selection.json").write_text(json.dumps(allresults,indent=2));(O/"audit.json").write_text(json.dumps(dict(status="PASS",models=len(allresults),same_order_within_seed=True,same_window_presentations=True,only9600LNtrained=True,teacher_frozen_exact=True,no_targets=True),indent=2));print("COMPLETE",flush=True)
if __name__=="__main__":
 with sdpa_kernel(SDPBackend.MATH):run()
