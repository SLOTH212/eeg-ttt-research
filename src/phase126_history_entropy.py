from phase86_meta_waveform import *
from phase66_train import prepared
from phase119_models import ResidualLogitHead
from phase77_predict import SpectralNet,context
import datetime
O=ROOT/"phase126";O.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def centered(z):return z-z.mean(1,keepdim=True)
def current_head(head,past,t,current):
 ix=np.maximum(np.arange(t-10,t+1),0)
 return head(torch.cat([current if i==t else past[int(i)] for i in ix])[None])[0]
def future_head(head,past,t,current_before,future_logits):
 rows=[]
 for j in range(1,5):
  ix=np.maximum(np.arange(t+j-10,t+j+1),0);parts=[]
  for i in ix:parts.append(past[int(i)] if i<t else current_before if i==t else future_logits[int(i)-t-1])
  rows.append(torch.cat(parts))
 return head(torch.stack(rows))
def run():
 old=lock();models=json.loads((ROOT/"phase79/protocol.json").read_text())["models"];specs=[]
 for seed in [17,29,43]:
  parent=next(s for s in models if s["model"]=="cbramod" and s["seed"]==seed);h=ROOT/"phase119"/f"history11_seed{seed}"/"model.pt";tp=ROOT/f"phase4/results/context11_mixed_{seed}/model.pt";specs.append(dict(seed=seed,parent=parent,history_head=str(h),head_sha256=sha(h),teacher=str(tp),teacher_sha256=sha(tp)))
 methods=["raw_entropy","history_entropy"];lrs=[.001,.01]
 pr=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),models=specs,controllers=old["controllers"],validation=old["held_source_validation"],methods=methods,lrs=lrs,primary=dict(method="history_entropy",lr=.001),method="Cold singleWRITE of9600CBraModLNaffines; current compositeprediction emittedbeforeWRITE. Entropy ofcurrent wholehistoryclassifier or rawparent currentclassifier. Historyheadfrozen. Clipgradientnorm1, no other writes overfuture4windows. Secondary.01 doesnotreplaceprimary.",cache_semantics="Pre-t history cachedunder originalparent. At t emitcurrentusing pre-WRITE logits (atstartup repeatcurrentforpadding; gradients throughduplicatedcurrent). AfterWRITE, retain allcached <=t logits exactly, append updatedparent logitsonlyasfuturewindowsarrive. Never retroactively recomputepastfeatures. Currentpastcacheforward andfuturelabel-freeprefix audited.",numeric_control="Same5windowparentbatch before/after; futureKEEP builtwithbefore logitsin samecacheflow. ZeroWRITE replay mustmatch exactly toexclude batch-size numericalgains.",controls="Frozen composite historyclassifier, same-seed directcontext11teacher, fixed .5composite/teacherblend. Teacherfutureoutputs forstatic scoringonly, neverloss.",gate="Primarythree-seed metricmeanswithinperson: eachsourcegroup futureCE>0,>=3/4positivepeople,balancedCE>=0,corrected>=harmedandcorrected>0. Also compareCE/accuracy withdirectteacher andstaticblend. Noonline/targetupdates unlessseparatelaterprotocol; thisrun sourceonly.",scope="Same64phase91sourcepulses/8knownpeople, no newblindclaims or labels. Sourcehistoryhead has extra30supervisedepochs. This is newentropyobjective on staticcausalclassifier, notextension of failedexternalteacherWRITEbranch.",cpu_ids=list(range(8,24)),gpu=True)
 pp=O/"protocol.json"
 if pp.exists():
  p=json.loads(pp.read_text());pr["created_utc"]=p["created_utc"];assert p==pr
 else:pp.write_text(json.dumps(pr,indent=2));(O/"protocol.sha256").write_text(sha(pp))
 data={sid:prepared(sid) for sid in old["controllers"]+old["held_source_validation"]};audits=[]
 for sp in specs:
  seed=sp["seed"];out=O/f"seed{seed}";out.mkdir(exist_ok=True);ip=Path(sp["parent"]["path"]);assert sha(ip)==sp["parent"]["sha256"];m=meta_model();m.load_state_dict(torch.load(ip,map_location="cpu",weights_only=False)["state"]);m.requires_grad_(False)
  for mod in m.encoder.modules():
   if isinstance(mod,nn.LayerNorm):
    for p in mod.parameters():p.requires_grad_(True)
  m.eval();params=[p for n,p in ln_named(m)];initial=[p.detach().clone() for p in params];state=snapshot(m);assert sum(p.numel() for p in params)==9600
  hc=torch.load(sp["history_head"],map_location="cpu",weights_only=False);assert hc["parent_sha256"]==sha(ip) and sha(Path(sp["history_head"]))==sp["head_sha256"];head=ResidualLogitHead(hc["dim"]);head.load_state_dict(hc["state"]);head.cuda().eval().requires_grad_(False);hs=snapshot(head)
  tc=torch.load(sp["teacher"],map_location="cpu",weights_only=False);assert sha(Path(sp["teacher"]))==sp["teacher_sha256"];teacher=SpectralNet(tc["dim"]);teacher.load_state_dict(tc["state"]);teacher.cuda().eval().requires_grad_(False);ts=snapshot(teacher)
  def restore():
   with torch.no_grad():
    for p,v in zip(params,initial):p.copy_(v)
  for group,sids in [("controllers",old["controllers"]),("validation",old["held_source_validation"])]:
   for sid in sids:
    dest=out/f"s{sid:02d}.npz";ap=out/f"s{sid:02d}_audit.json"
    if dest.exists() and ap.exists():
     a=json.loads(ap.read_text());assert sha(dest)==a["sha256"];audits.append(a);continue
    vx,vy=data[sid];restore()
    with torch.no_grad():past=centered(torch.cat([m(b.cuda()) for b in vx.split(128)]))
    if group=="validation":
     cached=np.load(ROOT/"phase119/logits"/f"seed{seed}_s{sid:02d}.npz")["logits"];assert abs(past.cpu().numpy()-cached).max()<1e-4
    folder=ROOT/("data/processed" if sid<20 else "phase2/processed");ct=context(np.load(folder/f"s{sid:02d}_meta.npz")["features"])
    with torch.no_grad():
     teacherp=torch.cat([teacher(torch.from_numpy(ct[a:a+128]).cuda()).double().log_softmax(1).exp() for a in range(0,len(ct),128)])
     statichead=torch.cat([head(torch.from_numpy(context(past.cpu().numpy())[a:a+512]).cuda()).double().log_softmax(1).exp() for a in range(0,len(past),512)])
    times=json.loads((ROOT/"phase91"/f"s{sid:02d}.json").read_text())["times"];bp=[];wp=np.empty((2,2,len(times),4,5),dtype=np.float64);norms=np.empty((2,2,len(times)));gnorms=np.empty((2,len(times)));lossvalues=np.empty((2,len(times)));labels=[];tp=[];maxcompat=0.
    for ti,t in enumerate(times):
     restore();x=vx[t:t+5].cuda();before=m(x);c0=centered(before);current=current_head(head,past,t,c0[0]);lpraw=before[0].double().log_softmax(0);lph=current.double().log_softmax(0);losses=[-(lpraw.exp()*lpraw).sum(),-(lph.exp()*lph).sum()]
     with torch.no_grad():
      base=future_head(head,past,t,c0[0].detach(),c0[1:].detach()).double().log_softmax(1).exp()
      zero=centered(m(x));zeroq=future_head(head,past,t,zero[0],zero[1:]).double().log_softmax(1).exp();assert torch.equal(base,zeroq)
      err=float(abs(base-statichead[t+1:t+5]).max());assert err<2e-5;maxcompat=max(maxcompat,err)
      if ti==0:
       pert=x.clone();pert[1:]*=3;pc=centered(m(pert));assert torch.equal(current_head(head,past,t,pc[0]),current.detach())
     grads=[torch.autograd.grad(loss,params,retain_graph=i==0) for i,loss in enumerate(losses)]
     for mi,g in enumerate(grads):
      norm=torch.sqrt(sum(v.square().sum() for v in g));assert torch.isfinite(norm);gnorms[mi,ti]=float(norm);lossvalues[mi,ti]=float(losses[mi].detach())
      for li,lr in enumerate(lrs):
       restore();scale=lr*torch.clamp(1/(norm+1e-6),max=1.)
       with torch.no_grad():
        for p,v,d in zip(params,initial,g):p.copy_(v-scale*d)
        actual=float(torch.sqrt(sum((p-v).square().sum() for p,v in zip(params,initial))));assert actual<=lr+1e-5;after=centered(m(x));q=future_head(head,past,t,c0[0].detach(),after[1:]).double().log_softmax(1).exp();wp[mi,li,ti]=q.cpu().numpy();norms[mi,li,ti]=actual
     bp.append(base.cpu().numpy());labels.append(vy[t:t+5]);tp.append(teacherp[t:t+5].cpu().numpy())
    restore();assert all(torch.equal(v.cpu(),state[k]) for k,v in m.state_dict().items()) and all(torch.equal(v.cpu(),hs[k]) for k,v in head.state_dict().items()) and all(torch.equal(v.cpu(),ts[k]) for k,v in teacher.state_dict().items())
    np.savez_compressed(dest,times=np.array(times),labels=np.array(labels),base=np.array(bp),write=wp,teacher=np.array(tp),norms=norms,gradient_norms=gnorms,losses=lossvalues)
    a=dict(seed=seed,subject=sid,group=group,sha256=sha(dest),writes=256,zero_write_exact=True,future_input_cannot_change_current=True,max_static_head_compatibility_error=maxcompat,all_states_restored_exact=True,past_cache_retained_after_write=True);ap.write_text(json.dumps(a,indent=2));audits.append(a);print("HISTORY WRITE",seed,group,sid,flush=True)
  del m,head,teacher;torch.cuda.empty_cache()
 assert len(audits)==24
 (O/"audit.json").write_text(json.dumps(dict(status="PASS",writes=sum(a["writes"] for a in audits),cases=len(audits),audits=audits,source_only=True),indent=2));print("COMPLETE",flush=True)
if __name__=="__main__":
 with sdpa_kernel(SDPBackend.MATH):run()
