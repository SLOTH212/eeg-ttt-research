from phase52_meta_head import *
rows=[]
for which in ["future","current"]:
 m=model_meta(torch.float64);head=list(m.recon_head.parameters());state={n:v.clone() for n,v in m.state_dict().items()};x,target,y=pulse(3,1429,torch.float64);outer,task,aux,norm=step_objective(m,x,target,y,which);hg=torch.autograd.grad(task,head);hn=torch.sqrt(sum(v.square().sum() for v in hg));assert hn>1e-12;initial=[p.detach().clone() for p in head];values=[];epsilon=1e-4
 for sign in [1,-1]:
  with torch.no_grad():
   for p,v,g in zip(head,initial,hg):p.copy_(v+sign*epsilon*g/hn)
  _,tt,_,_=step_objective(m,x,target,y,which,False);values.append(float(tt.detach()))
 with torch.no_grad():
  for p,v in zip(head,initial):p.copy_(v)
 fd=(values[0]-values[1])/(2*epsilon);relative=abs(fd-float(hn))/float(hn);assert relative<.001;assert all(torch.equal(v,m.state_dict()[n]) for n,v in state.items());rows.append(dict(target=which,task_only_hypergradient_norm=float(hn),finite_difference=fd,relative_error=relative,readout_restored_exact=True));print(rows[-1],flush=True)
(OUT52/"task_only_gradient_audit.json").write_text(json.dumps(dict(status="PASS",cases=rows,reason="Verify supervised meta-gradient separately so MSE regularizer cannot hide missing task gradients"),indent=2))
