from phase69_pulses import *
OUT74=ROOT/"phase74"
def run():
 OUT74.mkdir(exist_ok=True);protocol=dict(scope="posthoc source-only mechanistic diagnosis, not learned gate or layer selection",subjects=[7,17,27,37],classifier_seed=17,auxiliary="phase72 refitted waveform",future_horizon=4,layer_groups=12,finite_difference_eps=[.01,.005],future_labels="diagnostic only, never online inputs",contrasts=["current_stage","all_future_same_as_current","at_least_one_future_stage_change"])
 (OUT74/"protocol.json").write_text(json.dumps(protocol,indent=2))
 m=make("pretrained");r=np.load(ROOT/"phase72/readout.npz")
 with torch.no_grad():m.encoder.proj_out[0].weight.copy_(torch.tensor(r["coef"],device="cuda"));m.encoder.proj_out[0].bias.copy_(torch.tensor(r["bias"],device="cuda"))
 named=[(n,p) for n,p in m.named_parameters() if p.requires_grad];params=[p for n,p in named];initial=[p.detach().clone() for p in params];state=snapshot(m);layout=[];off=0
 for n,p in named:layout.append(dict(name=n,offset=off,length=p.numel(),layer=int(n.split(".")[3])));off+=p.numel()
 (OUT74/"layout.json").write_text(json.dumps(layout,indent=2));assert off==9600
 cw=torch.tensor(json.loads((ROOT/"phase36/mixed_class_weights.json").read_text())["balanced_weights"],device="cuda",dtype=torch.float64)
 def restore():
  with torch.no_grad():
   for p,v in zip(params,initial):p.copy_(v)
 def grad(loss):return torch.cat([v.flatten() for v in torch.autograd.grad(loss,params)]).detach()
 def task(x,y):
  lp=m(x).double().log_softmax(1);return -lp[torch.arange(len(y),device="cuda"),y]
 audits=[]
 for sid in protocol["subjects"]:
  times=np.load(ROOT/"phase48/masked"/f"s{sid:02d}_gradients.npz")["times"];arrays={k:[] for k in ["waveform","entropy","current_ce","future_ce","future_balanced_ce","labels"]};finite=[]
  for i,t0 in enumerate(times):
   t=int(t0);restore();x,y=sparse(sid,t);gen=torch.Generator(device="cpu").manual_seed(73000+sid*100000+t);mask=(torch.rand((1,2,30),generator=gen)<.5).cuda()
   pred=m.encoder(x[:1],mask);loss=((pred-x[:1]).square()*mask[...,None]).sum()/(mask.sum()*200);wave=grad(loss)
   lp=m(x[:1]).log_softmax(1);entropy=grad(-(lp.exp()*lp).sum())
   current=grad(task(x[:1],y[:1]).mean());future=grad(task(x[1:],y[1:]).mean());balanced=grad((task(x[1:],y[1:])*cw[y[1:]]).mean())
   for k,v in [("waveform",wave),("entropy",entropy),("current_ce",current),("future_ce",future),("future_balanced_ce",balanced)]:arrays[k].append(v.cpu().numpy())
   arrays["labels"].append(y.cpu().numpy())
   if i in [0,32]:
    direction=wave/wave.norm().clamp_min(1e-20);analytical=float((future.double()*direction.double()).sum());dparts=torch.split(direction,[p.numel() for p in params])
    for eps in protocol["finite_difference_eps"]:
     values=[]
     for sign in [-1,1]:
      with torch.no_grad():
       for p,v,d in zip(params,initial,dparts):p.copy_(v+sign*eps*d.reshape(p.shape))
       values.append(float(task(x[1:],y[1:]).mean()))
     fd=(values[1]-values[0])/(2*eps);finite.append(dict(subject=sid,t=t,eps=eps,analytical=analytical,finite_difference=fd,absolute_error=abs(fd-analytical)))
     restore()
  assert all(torch.equal(v.cpu(),state[n]) for n,v in m.state_dict().items())
  np.savez_compressed(OUT74/f"s{sid:02d}_gradients.npz",times=times,**{k:np.array(v) for k,v in arrays.items()});audits.extend(finite);print("LAYER GRADIENT DIAGNOSIS",sid,len(times),flush=True)
 (OUT74/"execution_audit.json").write_text(json.dumps(dict(state_unchanged=True,finite_differences=audits),indent=2));(OUT74/"complete.json").write_text(json.dumps(dict(complete=True,people=4,pulses=256)))
if __name__=="__main__":run()
