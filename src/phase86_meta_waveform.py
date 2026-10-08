from phase69_pulses import *
from torch.func import functional_call
from torch.nn.attention import sdpa_kernel,SDPBackend
import datetime
OUT86=ROOT/"phase86";OUT86.mkdir(exist_ok=True)
torch.backends.mha.set_fastpath_enabled(False)
def lock():
 pr=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),purpose="CBraMod source-supervised meta-learning of original waveform auxiliary projection; learn future-useful writes without changing frozen classifier.",meta_subjects=[3,4,12,13,14],controllers=[7,17,27,37],held_source_validation=[5,15,25,35],times="Existing phase48/masked/sXX_gradients.npz times; no new sampling selection.",initial_classifier="phase66/pretrained_0.0001/model.pt",classifier_sha256=hashlib.sha256((ROOT/"phase66/pretrained_0.0001/model.pt").read_bytes()).hexdigest(),initial_readout="phase72/readout.npz",readout_sha256=hashlib.sha256((ROOT/"phase72/readout.npz").read_bytes()).hexdigest(),meta_trainable="Original encoder.proj_out.0 weight and bias,40200params only",inner="One current masked waveform MSE step, 9600 LN affine, lr.01 clipnorm1; fresh state per pulse; no persistent classifier update.",mask="Bernoulli.5 seed73000+sid*100000+t, same as phase73",outer="Future4 meanCE + .001current MSE. CurrentCE and reconstruction-only equal-steps controls.",optimizer="AdamW lr.001 wd0, batch4 accumulated independent fast models, equal-person weights, headgrad clip1,5epochs, dedicated order generator seed17 after constructor.",primary="Future-meta head one-step future task benefit at lr.01.",advance="Primary must beat KEEP, initial head, current-meta, reconstruction-only and initial-head norm-matched to primary; mean futureCE>0, at least3/4 positive, balancedCE>=0, corrected>=harmed. Must also replicate signs >=3/4 source-validation people and preserve reconstruction validity all4 before any online extension.",selection="Final epoch5 only. No LR, epoch, subject or threshold search.",scope="All evaluation people previously observed; not new blind validation. Extra meta classification labels explicitly counted. If advancing, static classifier calibration with exact meta labels required.",cpu_ids=list(range(8,24)),gpu=0)
 p=OUT86/"protocol.json"
 if p.exists():old=json.loads(p.read_text());pr["created_utc"]=old["created_utc"];assert pr==old
 else:p.write_text(json.dumps(pr,indent=2));(OUT86/"protocol.sha256").write_text(hashlib.sha256(p.read_bytes()).hexdigest())
 return pr
def meta_model(dtype=torch.float32):
 m=make("pretrained").to(dtype=dtype);r=np.load(ROOT/"phase72/readout.npz")
 with torch.no_grad():
  m.encoder.proj_out[0].weight.copy_(torch.tensor(r["coef"],device="cuda",dtype=dtype));m.encoder.proj_out[0].bias.copy_(torch.tensor(r["bias"],device="cuda",dtype=dtype))
 m.encoder.proj_out[0].requires_grad_(True)
 return m
def ln_named(m):return [(n,p) for n,p in m.named_parameters() if p.requires_grad and not n.startswith("encoder.proj_out.")]
def mask_for(sid,t):
 gen=torch.Generator(device="cpu").manual_seed(73000+sid*100000+t)
 return (torch.rand((1,2,30),generator=gen)<.5).cuda()
def reconstruction(m,x,mask):
 p=m.encoder(x[:1],mask);return ((p-x[:1]).square()*mask[...,None]).sum()/(mask.sum()*200)
def objective(m,x,y,mask,which,create_graph=True):
 aux=reconstruction(m,x,mask);named=ln_named(m)
 if which=="reconstruction":return .001*aux,aux*0,aux,aux.detach()*0,{}
 g=torch.autograd.grad(aux,[p for n,p in named],create_graph=create_graph);norm=torch.sqrt(sum(v.square().sum() for v in g));clip=torch.clamp(1/(norm+1e-6),max=1.);fast={n:p-.01*v*clip for (n,p),v in zip(named,g)}
 xx,yy=(x[1:],y[1:]) if which=="future" else (x[:1],y[:1])
 logits=functional_call(m,fast,(xx,),strict=False);task=nn.functional.cross_entropy(logits,yy)
 return task+.001*aux,task,aux,norm,fast
def audit():
 lock();m=meta_model(torch.float64);state=snapshot(m);sid=3;times=np.load(ROOT/"phase48/masked/s03_gradients.npz")["times"];t=int(times[len(times)//2]);x,y=sparse(sid,t);x=x.double();mask=mask_for(sid,t);head=list(m.encoder.proj_out[0].parameters());initial=[p.detach().clone() for p in head];rows=[]
 assert sum(p.numel() for n,p in ln_named(m))==9600 and sum(p.numel() for p in head)==40200
 for which in ["current","future"]:
  for component in ["total","task"]:
   total,task,aux,norm,fast=objective(m,x,y,mask,which);loss=total if component=="total" else task;grads=torch.autograd.grad(loss,head);hn=torch.sqrt(sum(g.square().sum() for g in grads));assert torch.isfinite(hn) and hn>0
   for eps in [1e-4,5e-5]:
    values=[]
    for sign in [1,-1]:
     with torch.no_grad():
      for p,v,g in zip(head,initial,grads):p.copy_(v+sign*eps*g/hn)
     total,task,*_=objective(m,x,y,mask,which,False);values.append(float((total if component=="total" else task).detach()))
    with torch.no_grad():
     for p,v in zip(head,initial):p.copy_(v)
    fd=(values[0]-values[1])/(2*eps);err=abs(fd-float(hn))/float(hn);assert err<.002,(which,component,eps,err)
    rows.append(dict(target=which,component=component,epsilon=eps,gradient_norm=float(hn),finite_difference=fd,relative_error=err));print("HYPERGRAD",rows[-1],flush=True)
  _,_,_,_,fast=objective(m,x,y,mask,which,False)
  with torch.no_grad():
   fp=functional_call(m,fast,(x,),strict=False).clone()
   for n,p in ln_named(m):p.copy_(fast[n])
   physical=m(x).clone()
   for n,p in ln_named(m):p.copy_(state[n].cuda())
  err=float(abs(fp-physical).max());assert err<1e-10;assert all(torch.equal(v.cpu(),state[n]) for n,v in m.state_dict().items())
 (OUT86/"implementation_audit.json").write_text(json.dumps(dict(status="PASS",dtype="float64",finite_differences=rows,functional_physical_exact_to_1e10=True,classifier_and_readout_restored_exact=True,sdpa_math=True),indent=2))
def train(which):
 pr=lock();assert json.loads((OUT86/"implementation_audit.json").read_text())["status"]=="PASS";out=OUT86/which;out.mkdir(exist_ok=True)
 if (out/"complete.json").exists():return
 m=meta_model();head=list(m.encoder.proj_out[0].parameters());frozen={n:v for n,v in snapshot(m).items() if not n.startswith("encoder.proj_out.")};initial=[p.detach().clone() for p in head];records=[];cache={};labels={};counts={}
 for sid in pr["meta_subjects"]:
  times=np.load(ROOT/"phase48/masked"/f"s{sid:02d}_gradients.npz")["times"];counts[sid]=len(times);used=set()
  for t0 in times:
   t=int(t0);x,y=sparse(sid,t);assert bool((y>=0).all());records.append((sid,t));cache[sid,t]=(x.cpu(),y.cpu());used.update([] if which=="reconstruction" else [t] if which=="current" else range(t+1,t+5))
  labels[str(sid)]=sorted(used)
 opt=torch.optim.AdamW(head,lr=.001,weight_decay=0);rng=torch.Generator(device="cpu").manual_seed(17);orders=[];history=[];started=time.perf_counter()
 for epoch in range(5):
  order=torch.randperm(len(records),generator=rng).tolist();orders.append(order);losses=[]
  for i in range(0,len(order),4):
   batch=order[i:i+4];acc=[torch.zeros_like(p) for p in head]
   for idx in batch:
    sid,t=records[idx];x,y=[v.cuda() for v in cache[sid,t]];total,task,aux,norm,_=objective(m,x,y,mask_for(sid,t),which);assert torch.isfinite(total)
    grads=torch.autograd.grad(total,head);weight=len(records)/(len(counts)*counts[sid]*len(batch))
    for a,g in zip(acc,grads):a.add_(g.detach(),alpha=weight)
    losses.append([float(task.detach()),float(aux.detach()),float(norm.detach())])
   opt.zero_grad(set_to_none=True)
   for p,g in zip(head,acc):p.grad=g
   hn=nn.utils.clip_grad_norm_(head,1.);assert torch.isfinite(hn);opt.step()
  assert all(torch.equal(v.cpu(),frozen[n]) for n,v in m.state_dict().items() if n in frozen)
  values=np.array(losses);rec=dict(epoch=epoch+1,task_ce=float(values[:,0].mean()),waveform_mse=float(values[:,1].mean()),inner_gradient_norm=float(values[:,2].mean()),head_change_l2=float(torch.sqrt(sum((p-v).square().sum() for p,v in zip(head,initial)))),seconds=time.perf_counter()-started);history.append(rec);print("META",which,rec,flush=True)
 torch.save(dict(head={k:v.cpu() for k,v in m.encoder.proj_out[0].state_dict().items()},target=which,history=history,labels_used=labels),out/"head.pt")
 (out/"training_orders.json").write_text(json.dumps(dict(seed=17,orders=orders)))
 (out/"complete.json").write_text(json.dumps(dict(target=which,pulses=len(records),epochs=5,labels_used=labels,label_count=sum(map(len,labels.values())),history=history,classifier_unchanged_exact=True,seconds=time.perf_counter()-started),indent=2));print("COMPLETE",which,flush=True)
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--audit",action="store_true");parser.add_argument("--target");a=parser.parse_args()
 with sdpa_kernel(SDPBackend.MATH):
  if a.audit:audit()
  else:train(a.target)
