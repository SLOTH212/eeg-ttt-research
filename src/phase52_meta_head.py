from phase47_pulses import model_for
from phase46_spectral_head import *
from torch.func import functional_call
OUT52=ROOT/"phase52"
def model_meta(dtype=torch.float32):
 m=model_for("masked").to(dtype=dtype)
 for p in m.recon_head.parameters():p.requires_grad_(True)
 return m
def step_objective(m,x,target,y,which,create_graph=True):
 named=[(n,p) for n,p in m.named_parameters() if p.requires_grad and n.startswith("encoder.")];aux=m.reconstruction_loss(x[:1],target[:1]);g=torch.autograd.grad(aux,[p for n,p in named],create_graph=create_graph);norm=torch.sqrt(sum(v.square().sum() for v in g));clip=torch.clamp(1/(norm+1e-6),max=1.);fast={n:p-.01*v*clip for (n,p),v in zip(named,g)};xx=x[1:] if which=="future" else x[:1];yy=y[1:] if which=="future" else y[:1];logits=functional_call(m,fast,(xx,),strict=False);task=nn.functional.cross_entropy(logits,yy);outer=task+.001*aux;return outer,task,aux,norm
def pulse(sid,t,dtype=torch.float32):
 root=ROOT/("data/processed" if sid<20 else "phase2/processed");raw=np.load(root/f"s{sid:02d}_x.npy",mmap_mode="r");meta=np.load(root/f"s{sid:02d}_meta.npz");x=preprocess(np.array(raw[t:t+5])).reshape(5,2,2,15,200).cuda().to(dtype);target=torch.tensor(meta["features"][t:t+5,24:48],device="cuda",dtype=dtype);y=torch.tensor(meta["y"][t:t+5],device="cuda");return x,target,y
def audit():
 sid=3;times=np.load(ROOT/"phase48/masked/s03_gradients.npz")["times"];t=int(times[len(times)//2]);rows=[]
 for which in ["future","current"]:
  m=model_meta(torch.float64);state={n:v.clone() for n,v in m.state_dict().items()};x,target,y=pulse(sid,t,torch.float64);head=list(m.recon_head.parameters());initial=[p.detach().clone() for p in head];outer,task,aux,norm=step_objective(m,x,target,y,which);hg=torch.autograd.grad(outer,head);hn=torch.sqrt(sum(v.square().sum() for v in hg));assert torch.isfinite(hn) and hn>0;epsilon=1e-4;values=[]
  for sign in [1,-1]:
   with torch.no_grad():
    for p,v,g in zip(head,initial,hg):p.copy_(v+sign*epsilon*g/hn)
   loss,*_=step_objective(m,x,target,y,which,False);values.append(float(loss.detach()))
  with torch.no_grad():
   for p,v in zip(head,initial):p.copy_(v)
  fd=(values[0]-values[1])/(2*epsilon);relative=abs(fd-float(hn))/max(float(hn),1e-12);assert relative<.001;assert all(torch.equal(v,m.state_dict()[n]) for n,v in state.items())
  # Functional fast weights must match a physical one-step update, without leaving persistent changes.
  named=[(n,p) for n,p in m.named_parameters() if p.requires_grad and n.startswith("encoder.")];a=m.reconstruction_loss(x[:1],target[:1]);g=torch.autograd.grad(a,[p for n,p in named]);norm=torch.sqrt(sum(v.square().sum() for v in g));clip=torch.clamp(1/(norm+1e-6),max=1);fast={n:p-.01*v*clip for (n,p),v in zip(named,g)}
  with torch.no_grad():
   fp=functional_call(m,fast,(x,),strict=False).clone()
   for n,p in named:p.copy_(fast[n])
   physical=m(x).clone()
   for n,p in named:p.copy_(state[n])
  err=float(abs(fp-physical).max());assert err<1e-10;assert all(torch.equal(v,m.state_dict()[n]) for n,v in state.items());rows.append(dict(target=which,subject=sid,t=t,hypergradient_norm=float(hn),finite_difference=fd,relative_error=relative,functional_physical_logit_error=err,classifier_and_readout_restored_exact=True));print("META AUDIT",rows[-1],flush=True);del m,state
 (OUT52/"implementation_audit.json").write_text(json.dumps(dict(status="PASS",dtype="float64",epsilon=1e-4,cases=rows),indent=2))
def train(which):
 assert json.loads((OUT52/"implementation_audit.json").read_text())["status"]=="PASS";out=OUT52/which;out.mkdir(exist_ok=True)
 if (out/"complete.json").exists():return
 protocol=json.loads((OUT52/"protocol.json").read_text());seed_all(17);m=model_meta();frozen={n:v.clone() for n,v in m.state_dict().items() if not n.startswith("recon_head.")};head=list(m.recon_head.parameters());initial=[p.detach().clone() for p in head];records=[];labels={};counts={}
 for sid in protocol["meta_training_subjects"]:
  times=np.load(ROOT/"phase48/masked"/f"s{sid:02d}_gradients.npz")["times"];counts[sid]=len(times);used=set()
  for t in times:records.append((sid,int(t)));used.update([] if which=="reconstruction" else [int(t)] if which=="current" else range(int(t)+1,int(t)+5))
  labels[str(sid)]=sorted(used)
 cached={}
 for sid,t in records:
  x,target,y=pulse(sid,t);cached[sid,t]=(x.cpu(),target.cpu(),y.cpu())
 optimizer=torch.optim.AdamW(head,lr=.001,weight_decay=0.);history=[];start=time.perf_counter()
 for epoch in range(5):
  permutation=torch.randperm(len(records)).tolist();losses=[];norms=[]
  for start_index in range(0,len(records),4):
   batch=permutation[start_index:start_index+4];acc=[torch.zeros_like(p) for p in head]
   for index in batch:
    sid,t=records[index];x,target,y=[a.cuda() for a in cached[sid,t]]
    if which=="reconstruction":
     aux=m.reconstruction_loss(x[:1],target[:1]);outer=.001*aux;task=torch.zeros((),device="cuda");norm=torch.zeros((),device="cuda")
    else:outer,task,aux,norm=step_objective(m,x,target,y,which)
    assert torch.isfinite(outer);grads=torch.autograd.grad(outer,head);weight=len(records)/(len(counts)*counts[sid]*len(batch))
    for a,g in zip(acc,grads):a.add_(g.detach(),alpha=weight)
    losses.append([float(task.detach()),float(aux.detach())]);norms.append(float(norm.detach()))
   optimizer.zero_grad(set_to_none=True)
   for p,g in zip(head,acc):p.grad=g
   hn=nn.utils.clip_grad_norm_(head,1.);assert torch.isfinite(hn);optimizer.step()
  assert all(torch.equal(v,m.state_dict()[n]) for n,v in frozen.items());values=np.array(losses);rec=dict(epoch=epoch+1,mean_outer_task_ce=float(values[:,0].mean()),mean_current_reconstruction_mse=float(values[:,1].mean()),mean_inner_gradient_norm=float(np.mean(norms)),head_change_l2=float(torch.sqrt(sum((p-v).square().sum() for p,v in zip(head,initial)))));history.append(rec);print("META TRAIN",which,rec,flush=True)
 state=snapshot(m);atomic_save(dict(state=state,target=which,training_subjects=protocol["meta_training_subjects"],labels_used=labels,label_count=sum(map(len,labels.values())),epochs=5,history=history,classifier_unchanged_exact=True),out/"model.pt");(out/"complete.json").write_text(json.dumps(dict(target=which,meta_training_pulses=len(records),training_subjects=protocol["meta_training_subjects"],labels_used=labels,label_count=sum(map(len,labels.values())),epochs=5,history=history,classifier_unchanged_exact=True,seconds=time.perf_counter()-start),indent=2));print("META COMPLETE",which,flush=True)
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--audit",action="store_true");parser.add_argument("--target");a=parser.parse_args()
 if a.audit:audit()
 else:train(a.target)
