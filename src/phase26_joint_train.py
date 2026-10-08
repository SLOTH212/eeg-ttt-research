from phase21_full_encoder import *
OUT26=ROOT/"phase26"
class JointCarrier(FullCarrier):
 def __init__(self,source):
  super().__init__(source,"pretrained");sd=torch.load(ROOT/"phase14/weights_only.pt",map_location="cpu",weights_only=True);self.mask_token=nn.Parameter(sd["student.mask_token"].clone());self.aux_head=nn.Linear(200,8192)
  with torch.no_grad():self.aux_head.weight.copy_(sd["student.lm_head.weight"]);self.aux_head.bias.copy_(sd["student.lm_head.bias"])
 def auxiliary(self,x,tokens,random_mask=True):
  raw=x.flatten(0,1);target=tokens.flatten(0,1)
  if random_mask:
   order=torch.rand((len(raw),30),device=raw.device).argsort(1)[:,:15];mask=torch.zeros((len(raw),30),dtype=torch.bool,device=raw.device).scatter_(1,order,True)
  else:mask=(torch.arange(30,device=raw.device)[None]%2==0).expand(len(raw),-1)
  z=masked_tokens(self.encoder,raw,mask,self.mask_token);logits=self.aux_head(z[mask]);truth=target[mask];loss=nn.functional.cross_entropy(logits,truth);accuracy=(logits.argmax(1)==truth).float().mean();return loss,accuracy

def val_scores(model,val):return {sid:metrics(y,probabilities(model,x))["macro_f1"] for sid,(x,y,tokens) in val.items()}
def aux_scores(model,val):
 model.eval();result={}
 with torch.inference_mode():
  for sid,(x,y,tokens) in val.items():
   indices=np.linspace(0,len(y)-1,min(256,len(y)),dtype=int);losses=[];accuracy=[]
   for idx in np.array_split(indices,8):
    loss,acc=model.auxiliary(x[idx].cuda(),tokens[idx].cuda(),False);losses.append(float(loss));accuracy.append(float(acc))
   result[sid]=dict(masked_ce=float(np.mean(losses)),masked_accuracy=float(np.mean(accuracy)))
 return result

def run(source,weight,benchmark=False):
 out=OUT26/f"{source}_joint_{weight:g}";out.mkdir(exist_ok=True)
 if (out/"complete.json").exists() and not benchmark:return
 selection=json.loads((ROOT/"phase21/evaluation"/f"{source}_pretrained/selection.json").read_text());encoder_lr=selection["selection"]["lr"];split=json.loads((ROOT/"phase3/protocol.json").read_text())["folds"][source];model=JointCarrier(source).cuda();opt=torch.optim.AdamW([dict(params=model.encoder.parameters(),lr=encoder_lr),dict(params=model.head.parameters(),lr=1e-4),dict(params=list(model.aux_head.parameters())+[model.mask_token],lr=1e-4)],weight_decay=.01)
 if benchmark:
  x,y=prepared(0);valid=np.flatnonzero(y>=0)[:32];x=x[valid].cuda();y=torch.tensor(y[valid],device="cuda");tokens=torch.tensor(np.load(OUT26/"tokens/s00.npz")["tokens"][valid].astype(np.int64),device="cuda");model.train();start=time.perf_counter()
  for _ in range(4):
   opt.zero_grad(set_to_none=True);task=nn.functional.cross_entropy(model(x),y);aux,acc=model.auxiliary(x,tokens);loss=task+weight*aux;loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(loss) and torch.isfinite(norm);opt.step()
  torch.cuda.synchronize();result=dict(seconds4steps=time.perf_counter()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),parameters=sum(p.numel() for p in model.parameters()),task_loss=float(task.detach()),aux_loss=float(aux.detach()),aux_accuracy=float(acc.detach()));(OUT26/"benchmark.json").write_text(json.dumps(result,indent=2));print(result,flush=True);return
 xs=[];ys=[];ts=[];ss=[]
 for sid in split["base_train"]:
  x,y=prepared(sid);valid=y>=0;xs.append(x[valid]);ys.append(y[valid]);ts.append(np.load(OUT26/"tokens"/f"s{sid:02d}.npz")["tokens"][valid]);ss.extend([sid]*int(valid.sum()))
 x=torch.cat(xs);y=torch.tensor(np.concatenate(ys),dtype=torch.long);tokens=torch.tensor(np.concatenate(ts).astype(np.int64));ss=np.array(ss);del xs,ys,ts;counts=np.bincount(y.numpy(),minlength=5);cw=np.sqrt(len(y)/np.maximum(counts,1));ns={s:int((ss==s).sum()) for s in np.unique(ss)};w=np.array([1/ns[s] for s in ss])*cw[y.numpy()];w=torch.tensor(w/w.mean(),dtype=torch.float32);val={}
 for sid in split["base_val"]:
  vx,vy=prepared(sid);vt=torch.tensor(np.load(OUT26/"tokens"/f"s{sid:02d}.npz")["tokens"].astype(np.int64));val[sid]=(vx,vy,vt)
 reference_path=OUT26/f"{source}_classifier_only_auxiliary.json"
 if not reference_path.exists():
  reference=JointCarrier(source).eval().cuda();path=ROOT/"phase21"/selection["selected_directory"]/"model.pt";missing=reference.load_state_dict(torch.load(path,map_location="cpu",weights_only=False)["state"],strict=False);assert set(missing.missing_keys)=={"mask_token","aux_head.weight","aux_head.bias"} and not missing.unexpected_keys;reference_path.write_text(json.dumps(dict(source=source,checkpoint=str(path),classification=val_scores(reference,val),auxiliary=aux_scores(reference,val),aux_head="Original pretrained auxiliary weights paired with classification-only adapted encoder; no refit"),indent=2));del reference
 model.eval();initial_per=val_scores(model,val);initial=float(np.mean(list(initial_per.values())));best=initial;best_epoch=0;best_state=snapshot(model);history=[];start_epoch=0;initial_aux=aux_scores(model,val);seed_all(17);resume_path=out/"resume.pt"
 if resume_path.exists():
  ck=torch.load(resume_path,map_location="cpu",weights_only=False);assert ck["encoder_lr"]==encoder_lr and ck["aux_weight"]==weight;model.load_state_dict(ck["state"]);opt.load_state_dict(ck["optimizer"]);best=ck["best"];best_epoch=ck["best_epoch"];best_state=ck["best_state"];history=ck["history"];start_epoch=ck["epoch"];torch.set_rng_state(ck["rng"]);torch.cuda.set_rng_state(ck["cuda_rng"])
 for epoch in range(start_epoch,10):
  model.train();start=time.perf_counter();task_losses=[];aux_losses=[]
  for idx in torch.randperm(len(y)).split(64):
   opt.zero_grad(set_to_none=True);xb=x[idx].cuda();task=(nn.functional.cross_entropy(model(xb),y[idx].cuda(),reduction="none")*w[idx].cuda()).mean();aux,_=model.auxiliary(xb,tokens[idx].cuda(),True);loss=task+weight*aux;loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(loss) and torch.isfinite(norm);opt.step();task_losses.append(float(task.detach()));aux_losses.append(float(aux.detach()))
  model.eval();per=val_scores(model,val);score=float(np.mean(list(per.values())));diagnostics=aux_scores(model,val);history.append(dict(epoch=epoch+1,val_f1=score,participants=per,auxiliary_validation=diagnostics,task_loss=float(np.mean(task_losses)),aux_loss=float(np.mean(aux_losses)),seconds=time.perf_counter()-start))
  if score>best:best=score;best_epoch=epoch+1;best_state=snapshot(model)
  atomic_save(dict(state=snapshot(model),optimizer=opt.state_dict(),best=best,best_epoch=best_epoch,best_state=best_state,history=history,epoch=epoch+1,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),encoder_lr=encoder_lr,aux_weight=weight),resume_path)
  print("SOURCE JOINT",source,weight,epoch+1,score,history[-1]["task_loss"],history[-1]["aux_loss"],flush=True)
 model.load_state_dict(best_state);selected_aux=aux_scores(model,val);atomic_save(dict(state=best_state,source=source,aux_weight=weight,encoder_lr=encoder_lr,best_epoch=best_epoch,best_val_f1=best,initial_val_f1=initial,history=history),out/"model.pt");(out/"complete.json").write_text(json.dumps(dict(source=source,aux_weight=weight,encoder_lr=encoder_lr,best_epoch=best_epoch,best_val_f1=best,initial_val_f1=initial,initial_auxiliary=initial_aux,selected_auxiliary=selected_aux,history=history,source_validation_only=True,training_windows=len(y),training_class_counts=counts.tolist()),indent=2));print("SOURCE JOINT COMPLETE",source,weight,best,flush=True)
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--source",default="young");parser.add_argument("--weight",type=float,default=.01);parser.add_argument("--benchmark",action="store_true");a=parser.parse_args();run(a.source,a.weight,a.benchmark)
