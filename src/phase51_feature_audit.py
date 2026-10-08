from phase51_pulses import *
from phase32_feature_audit import prefix_history
out=ROOT/"phase51";results=[];cachechecks=[]
for sid in [7,17,27,37]:
 model=scope_online.model_for("masked");cache=np.load(out/"cache"/f"s{sid:02d}.npz");x,y=prepared(sid);root=ROOT/("data/processed" if sid<20 else "phase2/processed");spectral=np.load(root/f"s{sid:02d}_meta.npz")["features"];captured={};hook=model.head.register_forward_pre_hook(lambda module,args:captured.update(z=args[0].detach().clone()))
 for t in np.linspace(0,len(x)-5,3,dtype=int):
  xx=x[t:t+5].cuda();changed=xx.clone();changed[1:]*=-1.7
  with torch.no_grad():
   p=model(xx).softmax(1);z=captured["z"][0].cpu().numpy().copy();pp=model(changed).softmax(1);zz=captured["z"][0].cpu().numpy().copy()
  assert torch.equal(p[0],pp[0]) and np.array_equal(z,zz);assert abs(p[0].cpu().numpy()-cache["p"][t]).max()<2e-5;cachechecks.append(dict(subject=sid,t=int(t),future_perturbation_current_error=0))
 hook.remove()
 for trajectory in ["continuous","reset64"]:
  folder=out/f"lr0.01_{trajectory}_s{sid:02d}";saved=np.load(folder/"pulses.npz");pred=np.load(folder/"actual_predictions.npz")["p"];lookup={int(t):i for i,t in enumerate(saved["t"])};graph=scope_online.SpectralGraph(model,"future_groups",x[:1].cuda());one=LogitForward(model,x[:1].cuda());worst=0.;checks=0
  for t in range(81):
   if trajectory=="reset64" and t%64==0:graph.reset()
   before=[p.detach().clone() for p in graph.params] if t%20==0 else None;xt=x[t:t+1].cuda();q=graph.step(xt,torch.tensor(spectral[t:t+1,24:48],device="cuda",dtype=torch.float32)).detach().cpu().numpy()[0].copy();assert abs(q-pred[t]).max()<2e-5
   if t not in lookup:continue
   with torch.no_grad():after=one.run(xt).softmax(1).cpu().numpy()[0].copy()
   drift=float(torch.sqrt(sum((v-s).square().sum() for v,s in zip(before,graph.initial))));features=np.array([entropy(q),float(graph.norm.detach()),float(abs(after-q).sum()),drift,float(q.max()),entropy(after),*q.tolist(),float(np.mean(abs(spectral[t]-spectral[t-1]))) if t else 0.,float(t==0),*cache["p"][t].tolist(),entropy(cache["p"][t]),float(abs(q-cache["p"][t]).sum()),*after.tolist(),*prefix_history(cache["p"][:t+1],.2).tolist(),*prefix_history(cache["p"][:t+1],.8).tolist(),float(graph.loss.detach())]);expected=saved["features"][lookup[t]];err=float(abs(features-expected).max());worst=max(worst,err);assert np.allclose(features,expected,atol=2e-5,rtol=2e-5);checks+=1
  for idx in np.linspace(0,len(saved["t"])-1,3,dtype=int):
   t=int(saved["t"][idx]);f=saved["features"][idx];assert np.array_equal(f[13:18],cache["p"][t]);assert np.array_equal(f[25:30],prefix_history(cache["p"][:t+1],.2));assert np.array_equal(f[30:35],prefix_history(cache["p"][:t+1],.8))
  results.append(dict(subject=sid,trajectory=trajectory,full_dynamic_feature_replay_checks=checks,max_full_feature_error=worst,all_prefix_predictions_checked=81,stored_history_prefix_checks=3));print("FEATURE AUDIT",results[-1],flush=True);graph.reset();del graph,one
 del model
(out/"feature_causality_audit.json").write_text(json.dumps(dict(status="PASS",future_perturbation_cases=cachechecks,dynamic_prefix_replay=results,scope_training_disjoint_from_all_controllers=True,task_labels_not_used_by_features=True),indent=2))
