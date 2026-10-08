from phase21_full_encoder import *
from phase28_joint_train import JointCarrier,aux_scores
import hashlib
OUT35=ROOT/"phase38"
def run(source,classifier_seed):
 out=OUT35/f"{source}_seed{classifier_seed}";out.mkdir(exist_ok=True)
 if (out/"complete.json").exists():return
 parent=ROOT/"phase33"/f"{source}_joint_0_seed{classifier_seed}"/"model.pt";model=JointCarrier(source).cuda().eval();model.load_state_dict(torch.load(parent,map_location="cpu",weights_only=False)["state"]);model.requires_grad_(False);model.aux_head.requires_grad_(True);model.mask_token.requires_grad_(True);params=list(model.aux_head.parameters())+[model.mask_token];frozen={k:v.clone() for k,v in model.state_dict().items() if k!="mask_token" and not k.startswith("aux_head.")};split=json.loads((ROOT/"phase3/protocol.json").read_text())["folds"][source];xs=[];ts=[]
 for sid in split["base_train"]:
  x,y=prepared(sid);valid=y>=0;xs.append(x[valid]);ts.append(np.load(ROOT/"phase28/tokens"/f"s{sid:02d}.npz")["tokens"][valid])
 x=torch.cat(xs);tokens=torch.tensor(np.concatenate(ts).astype(np.int64));del xs,ts;val={}
 for sid in split["base_val"]:
  vx,vy=prepared(sid);vt=torch.tensor(np.load(ROOT/"phase28/tokens"/f"s{sid:02d}.npz")["tokens"].astype(np.int64));val[sid]=(vx,vy,vt)
 opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01);initial=aux_scores(model,val);best=float(np.mean([v["masked_ce"] for v in initial.values()]));best_epoch=0;best_state=snapshot(model);history=[];start_epoch=0;seed_all(17);resume=out/"resume.pt"
 if resume.exists():
  saved=torch.load(resume,map_location="cpu",weights_only=False);model.load_state_dict(saved["state"]);opt.load_state_dict(saved["optimizer"]);best=saved["best_ce"];best_epoch=saved["best_epoch"];best_state=saved["best_state"];history=saved["history"];start_epoch=saved["epoch"];torch.set_rng_state(saved["rng"]);torch.cuda.set_rng_state(saved["cuda_rng"])
 for epoch in range(start_epoch,10):
  model.eval();losses=[];start=time.perf_counter()
  for idx in torch.randperm(len(x)).split(64):
   opt.zero_grad(set_to_none=True);loss,_=model.auxiliary(x[idx].cuda(),tokens[idx].cuda(),True);loss.backward();norm=nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(loss) and torch.isfinite(norm);opt.step();losses.append(float(loss.detach()))
  assert all(torch.equal(v,model.state_dict()[k]) for k,v in frozen.items());scores=aux_scores(model,val);score=float(np.mean([v["masked_ce"] for v in scores.values()]));history.append(dict(epoch=epoch+1,auxiliary_ce=score,training_ce=float(np.mean(losses)),participants=scores,seconds=time.perf_counter()-start))
  if score<best:best=score;best_epoch=epoch+1;best_state=snapshot(model)
  atomic_save(dict(state=snapshot(model),optimizer=opt.state_dict(),best_ce=best,best_epoch=best_epoch,best_state=best_state,history=history,epoch=epoch+1,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),resume);print("AUX HEAD ONLY",source,history[-1],flush=True)
 model.load_state_dict(best_state);validation=[]
 for sid,(vx,vy,vt) in val.items():
  pr=probabilities(model,vx);reference=np.load(ROOT/"phase33"/f"{source}_joint_0_seed{classifier_seed}"/f"s{sid:02d}.npz");assert np.array_equal(vy,reference["y"]);error=float(abs(pr-reference["p"]).max());disagreements=int((pr.argmax(1)!=reference["p"].argmax(1)).sum());assert error<2e-5 and disagreements==0;idx=np.linspace(0,len(vy)-1,min(256,len(vy)),dtype=int);other=np.roll(idx,len(idx)//2);aux_losses=[];shuffled=[];accuracies=[]
  with torch.no_grad():
   for positions in np.array_split(np.arange(len(idx)),8):
    loss,acc=model.auxiliary(vx[idx[positions]].cuda(),vt[idx[positions]].cuda(),False);shuffle_loss,_=model.auxiliary(vx[other[positions]].cuda(),vt[idx[positions]].cuda(),False);aux_losses.append(float(loss));shuffled.append(float(shuffle_loss));accuracies.append(float(acc))
  validation.append(dict(subject=sid,classification=metrics(vy,pr),max_frozen_probability_error=error,argmax_disagreements=disagreements,masked_ce=float(np.mean(aux_losses)),masked_accuracy=float(np.mean(accuracies)),input_shuffled_ce=float(np.mean(shuffled))));np.savez_compressed(out/f"s{sid:02d}.npz",y=vy,p=pr)
 assert all(torch.equal(v,model.state_dict()[k]) for k,v in frozen.items());atomic_save(dict(state=best_state,source=source,best_epoch=best_epoch,best_aux_ce=best,history=history,parent_classifier_checkpoint=str(parent),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest()),out/"model.pt");(out/"complete.json").write_text(json.dumps(dict(source=source,best_epoch=best_epoch,best_aux_ce=best,initial_auxiliary=initial,history=history,validation=validation,classifier_state_unchanged_exact=True,trainable_names=[n for n,p in model.named_parameters() if p.requires_grad],trainable_parameters=sum(p.numel() for p in params),source_only=True,additional_auxiliary_training_epochs=10,parent_classifier_sha256=hashlib.sha256(parent.read_bytes()).hexdigest()),indent=2));print("AUX HEAD COMPLETE",source,best,flush=True)
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--source",required=True);parser.add_argument("--classifier-seed",type=int,required=True);a=parser.parse_args();run(a.source,a.classifier_seed)
