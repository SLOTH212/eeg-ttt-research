from phase52_meta_head import *
OUT57=ROOT/"phase57"
VARIANTS=["fresh_ordinary","fresh_balanced","states_ordinary","states_balanced"]
LAYOUT57=json.loads((ROOT/"phase48/parameter_layout.json").read_text())
def start_model(dtype=torch.float32):
 m=model_meta(dtype);m.load_state_dict(torch.load(ROOT/"phase52/current/model.pt",map_location="cpu",weights_only=False)["state"]);return m
def ln_named(m):return [(n,p) for n,p in m.named_parameters() if p.requires_grad and n.startswith("encoder.")]
def set_flat(m,flat):
 layout=LAYOUT57;named=ln_named(m);assert [n for n,p in named]==[a["name"] for a in layout["parameters"]];params=[p for n,p in named];packed=torch.as_tensor(flat,device=params[0].device,dtype=params[0].dtype);values=[packed[a["offset"]:a["offset"]+a["length"]].reshape(a["shape"]) for a in layout["parameters"]]
 with torch.no_grad():torch._foreach_copy_(params,values)
def source_base(m):
 base={n:p.detach().clone() for n,p in ln_named(m)};m._meta_radii={n:.1*max(float(v.square().mean().sqrt()),1)*v.numel()**.5 for n,v in base.items()};return base
def restore_base(m,base):
 with torch.no_grad():
  for n,p in ln_named(m):p.copy_(base[n])
def weights():
 raw=np.array(json.loads((ROOT/"phase36/mixed_class_weights.json").read_text())["balanced_weights"]);means=[]
 for sid in [3,4,12,13,14]:
  times=np.load(ROOT/"phase48/masked"/f"s{sid:02d}_gradients.npz")["times"];labels=np.load(ROOT/"data/processed"/f"s{sid:02d}_meta.npz")["y"][times];means.append(float(raw[labels].mean()))
 normalizer=float(np.mean(means));return raw/normalizer,normalizer
def objective(m,x,target,y,base,cw,create_graph=True):
 named=ln_named(m);aux=m.reconstruction_loss(x[:1],target[:1]);grads=torch.autograd.grad(aux,[p for n,p in named],create_graph=create_graph);norm=torch.sqrt(sum(g.square().sum() for g in grads));clip=torch.clamp(1/(norm+1e-6),max=1);fast={}
 for (n,p),g in zip(named,grads):
  radius=m._meta_radii[n];d=p-.01*clip*g-base[n];fast[n]=base[n]+d*torch.clamp(radius/torch.linalg.vector_norm(d).clamp_min(1e-12),max=1)
 logits=functional_call(m,fast,(x[:1],),strict=False);task=(nn.functional.cross_entropy(logits,y[:1],reduction="none")*cw[y[:1]]).mean();return task+.001*aux,task,aux,norm,fast
def audit():
 assert (OUT57/"state_pool_complete.json").exists();rows=[];balanced,_=weights();pool=np.load(OUT57/"state_pool/train/s03_continuous.npz");idx=len(pool["times"])//2;t=int(pool["times"][idx])
 for variant in VARIANTS:
  m=start_model(torch.float64);base=source_base(m)
  if variant.startswith("states"):set_flat(m,pool["states"][idx])
  state={n:v.clone() for n,v in m.state_dict().items()};x,target,y=pulse(3,t,torch.float64);cw=torch.tensor(balanced if variant.endswith("balanced") else np.ones(5),device="cuda",dtype=torch.float64);head=list(m.recon_head.parameters());initial=[p.detach().clone() for p in head]
  for component in ["total","task_only"]:
   outer,task,aux,norm,fast=objective(m,x,target,y,base,cw);loss=outer if component=="total" else task;grads=torch.autograd.grad(loss,head);gn=torch.sqrt(sum(v.square().sum() for v in grads));assert torch.isfinite(gn) and gn>1e-12;epsilon=1e-4;vals=[]
   for sign in [1,-1]:
    with torch.no_grad():
     for p,v,g in zip(head,initial,grads):p.copy_(v+sign*epsilon*g/gn)
    result=objective(m,x,target,y,base,cw,False);vals.append(float(result[0 if component=="total" else 1].detach()))
   with torch.no_grad():
    for p,v in zip(head,initial):p.copy_(v)
   fd=(vals[0]-vals[1])/(2*epsilon);relative=abs(fd-float(gn))/float(gn);assert relative<.001;assert all(torch.equal(v,m.state_dict()[n]) for n,v in state.items());rows.append(dict(variant=variant,component=component,gradient_norm=float(gn),finite_difference=fd,relative_error=relative,state_restored_exact=True));print("STATE META AUDIT",rows[-1],flush=True)
  del m,state
 (OUT57/"implementation_audit.json").write_text(json.dumps(dict(status="PASS",cases=rows,projection_relative_to_source=True,weighting_before_reduction=True),indent=2))
def train(variant):
 assert variant in VARIANTS and json.loads((OUT57/"implementation_audit.json").read_text())["status"]=="PASS";out=OUT57/variant;out.mkdir(exist_ok=True)
 if (out/"complete.json").exists():return
 protocol=json.loads((OUT57/"protocol.json").read_text());seed_all(17);m=start_model();base=source_base(m);fixed={n:v.clone() for n,v in m.state_dict().items() if not n.startswith("recon_head.")};head=list(m.recon_head.parameters());records=[];cached={};pools={};counts={};labels={}
 for sid in protocol["training_subjects"]:
  a=np.load(OUT57/"state_pool/train"/f"s{sid:02d}_continuous.npz");b=np.load(OUT57/"state_pool/train"/f"s{sid:02d}_reset64.npz");assert np.array_equal(a["times"],b["times"]);counts[sid]=len(a["times"]);labels[str(sid)]=a["times"].tolist()
  for index,t0 in enumerate(a["times"]):
   t=int(t0);records.append((sid,t));x,target,y=pulse(sid,t);cached[sid,t]=(x[:1].cpu(),target[:1].cpu(),y[:1].cpu());pools[sid,t]=(a["states"][index].copy(),b["states"][index].copy())
 balanced,normalizer=weights();cw=torch.tensor(balanced if variant.endswith("balanced") else np.ones(5),device="cuda",dtype=torch.float32);rng=np.random.default_rng(17);orders=[rng.permutation(len(records)).tolist() for _ in range(5)];codes=np.random.default_rng(5717).integers(0,3,size=(5,len(records)));optimizer=torch.optim.AdamW(head,lr=.001,weight_decay=0.);history=[];state_counts=np.zeros(3,dtype=int);start=time.perf_counter()
 for epoch in range(5):
  losses=[];norms=[]
  for offset in range(0,len(records),4):
   batch=orders[epoch][offset:offset+4];acc=[torch.zeros_like(p) for p in head]
   for index in batch:
    sid,t=records[index];code=int(codes[epoch,index]) if variant.startswith("states") else 0;state_counts[code]+=1
    if code:set_flat(m,pools[sid,t][code-1])
    else:restore_base(m,base)
    x,target,y=[a.cuda() for a in cached[sid,t]];outer,task,aux,norm,fast=objective(m,x,target,y,base,cw);assert torch.isfinite(outer);grads=torch.autograd.grad(outer,head);weight=len(records)/(len(counts)*counts[sid]*len(batch))
    for a,g in zip(acc,grads):a.add_(g.detach(),alpha=weight)
    losses.append([float(task.detach()),float(aux.detach())]);norms.append(float(norm.detach()));restore_base(m,base)
   optimizer.zero_grad(set_to_none=True)
   for p,g in zip(head,acc):p.grad=g
   gn=nn.utils.clip_grad_norm_(head,1.);assert torch.isfinite(gn);optimizer.step()
  assert all(torch.equal(v,m.state_dict()[n]) for n,v in fixed.items());a=np.array(losses);rec=dict(epoch=epoch+1,mean_task_ce=float(a[:,0].mean()),mean_aux_mse=float(a[:,1].mean()),mean_inner_gradient_norm=float(np.mean(norms)));history.append(rec);print("FACTORIAL META",variant,rec,flush=True)
 assert all(torch.equal(v,m.state_dict()[n]) for n,v in fixed.items());meta=dict(variant=variant,labels_used=labels,label_count=sum(map(len,labels.values())),prior_meta_epochs=5,new_meta_epochs=5,state_choice_counts=state_counts.tolist(),state_choices_seed=5717,minibatch_seed=17,source_class_weight_normalizer=normalizer,class_weights=cw.cpu().tolist(),classifier_restored_exact=True,history=history,seconds=time.perf_counter()-start,parent_sha256=protocol["initial_head_sha256"]);atomic_save(dict(state=snapshot(m),**meta),out/"model.pt");(out/"complete.json").write_text(json.dumps(meta,indent=2));print("FACTORIAL COMPLETE",variant,flush=True)
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--audit",action="store_true");parser.add_argument("--variant");a=parser.parse_args()
 if a.audit:audit()
 else:train(a.variant)
