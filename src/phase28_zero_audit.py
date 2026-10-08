from phase28_joint_train import *
if __name__=="__main__":
 model=JointCarrier("young").cuda().train();x,y=prepared(5);idx=np.linspace(0,len(y)-1,12,dtype=int);idx=idx[y[idx]>=0];x=x[idx].cuda();y=torch.tensor(y[idx],device="cuda");tokens=torch.tensor(np.load(ROOT/"phase28/tokens/s05.npz")["tokens"][idx].astype(np.int64),device="cuda");seed_all(17);rng=torch.get_rng_state();crng=torch.cuda.get_rng_state()
 def gradients(with_aux):
  torch.set_rng_state(rng);torch.cuda.set_rng_state(crng);model.zero_grad(set_to_none=True);task=nn.functional.cross_entropy(model(x),y)
  if with_aux:aux,_=model.auxiliary(x,tokens);loss=task+0*aux
  else:loss=task
  loss.backward();grads={k:p.grad.detach().clone() for k,p in model.named_parameters() if p.grad is not None};return grads,float(task.detach()),float(aux.detach()) if with_aux else None,torch.cuda.get_rng_state()
 ref,task,_,_=gradients(False);repeat,_,_,_=gradients(False);actual,_,aux,after=gradients(True)
 def compare(other):
  maxdiff=max(float((other[k]-v).abs().max()) for k,v in ref.items());den=sum(float(v.double().square().sum()) for v in ref.values());num=sum(float((other[k].double()-v.double()).square().sum()) for k,v in ref.items());return dict(max_abs_difference=maxdiff,relative_l2_difference=(num/den)**.5)
 baseline=compare(repeat);difference=compare(actual);assert difference["relative_l2_difference"]<1e-5;assert all(torch.count_nonzero(v)==0 for k,v in actual.items() if k not in ref);assert not torch.equal(crng,after)
 result=dict(status="PASS",source_subject=5,windows=len(idx),zero_auxiliary_vs_task_only=difference,task_only_repeat=baseline,unused_auxiliary_gradients_zero=True,mask_rng_consumed=True,task_loss=task,auxiliary_loss=aux,scope="Matched RNG and batch. FP32 nondeterministic GPU reductions: tolerance relativeL2<1e-5, not bitwise equality. No target labels used.");(OUT26/"zero_control_audit.json").write_text(json.dumps(result,indent=2));print(result,flush=True)
