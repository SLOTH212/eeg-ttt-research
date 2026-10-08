from phase22_cuda_graph import GraphAdapter
from phase22_source_screen import *
OUT24=ROOT/"phase24"
def entropy(p):return float(-(p*np.log(np.clip(p,1e-12,1))).sum())
def run(source,sid,trajectory):
 out=OUT24/f"{source}_{trajectory}_s{sid:02d}";out.mkdir(exist_ok=True)
 if (out/"complete.json").exists():return
 checkpoint=source_checkpoint(source);model=FullCarrier(source,"pretrained").eval().cuda();model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=False)["state"]);model.requires_grad_(False)
 for module in model.encoder.modules():
  if isinstance(module,nn.LayerNorm):
   for p in module.parameters():p.requires_grad_(True)
 x,y=prepared(sid);meta=ROOT/("data/processed" if sid<20 else "phase2/processed")/f"s{sid:02d}_meta.npz";spectral=np.load(meta)["features"];anchor=np.load(ROOT/"phase21/evaluation"/f"{source}_pretrained"/f"s{sid:02d}.npz")["source_tuned"];graph=GraphAdapter(model,"entropy",.001,x[:1].cuda());params=graph.params;frozen={n:v.detach().clone() for n,v in model.state_dict().items() if not dict(model.named_parameters()).get(n,torch.tensor(0.)).requires_grad};rows=[];pred=[];max_current_error=0.;used=set();start=time.perf_counter();captured={}
 hook=model.head.register_forward_pre_hook(lambda module,args:captured.update(z=args[0].detach()))
 histories={}
 for alpha in [.2,.8]:
  history=[];state=None
  for q in anchor:state=q.copy() if state is None else alpha*q+(1-alpha)*state;history.append(state.copy())
  histories[alpha]=np.array(history)
 def restore(state):
  with torch.no_grad():
   for p,v in zip(params,state):p.copy_(v)
 for t in range(len(y)):
  if trajectory=="reset64" and t%64==0:graph.reset()
  pulse=t%20==0 and t+4<len(y)
  before=[p.detach().clone() for p in params] if pulse else None
  q=graph.step(x[t:t+1].cuda(),torch.as_tensor(anchor[t],device="cuda")).detach().cpu().numpy()[0].copy();pred.append(q);assert bool(graph.finite)
  if not pulse:continue
  proposal=[p.detach().clone() for p in params]
  with torch.inference_mode():after=model(x[t:t+1].cuda()).softmax(1).cpu().numpy()[0]
  drift=float(torch.sqrt(sum((v-s).square().sum() for v,s in zip(before,graph.initial))))
  features=[entropy(q),float(graph.norm.detach()),float(np.abs(after-q).sum()),drift,float(q.max()),entropy(after),*q.tolist(),float(np.mean(np.abs(spectral[t]-spectral[t-1]))) if t else 0.,float(t==0),*anchor[t].tolist(),entropy(anchor[t]),float(np.abs(q-anchor[t]).sum())]
  features.extend(after.tolist()+histories[.2][t].tolist()+histories[.8][t].tolist())
  # Current/proposal/history observations precede any label inspection; source embedding is read below from the RESET view current sample.
  if np.any(y[t:t+5]<0):continue
  views={}
  for action,state in [("commit",proposal),("keep",before),("reset",graph.initial)]:
   restore(state)
   with torch.inference_mode():views[action]=model(x[t:t+5].cuda()).log_softmax(1).cpu().numpy()
   if action=="reset":features.extend(captured["z"][0].cpu().numpy().tolist())
  restore(proposal);assert all(torch.equal(p,v) for p,v in zip(params,proposal));max_current_error=max(max_current_error,float(np.max(np.abs(np.exp(views["keep"][0])-q))))
  losses={k:-v[np.arange(5),y[t:t+5]] for k,v in views.items()};future=np.array([losses['keep'][1:].mean()-losses['reset'][1:].mean(),losses['keep'][1:].mean()-losses['commit'][1:].mean()]);current=np.array([losses['keep'][0]-losses['reset'][0],losses['keep'][0]-losses['commit'][0]])
  if drift==0:assert future[0]==0 and current[0]==0
  rows.append(dict(t=t,features=features,future_gain=future.tolist(),current_gain=current.tolist(),current_class=int(y[t]),stable=bool(np.all(y[t+1:t+5]==y[t])),current_correct=bool(q.argmax()==y[t])));used.update(range(t+1,t+5))
 assert max_current_error<.002;assert all(torch.equal(model.state_dict()[n],v) for n,v in frozen.items())
 hook.remove();np.savez_compressed(out/"actual_predictions.npz",y=y,predictions=np.array(pred))
 np.savez_compressed(out/"pulses.npz",features=np.array([r['features'] for r in rows]),future_gain=np.array([r['future_gain'] for r in rows]),current_gain=np.array([r['current_gain'] for r in rows]),t=np.array([r['t'] for r in rows]),current_class=np.array([r['current_class'] for r in rows]),stable=np.array([r['stable'] for r in rows]),current_correct=np.array([r['current_correct'] for r in rows]),future_labels_used=np.array(sorted(used)))
 (out/"complete.json").write_text(json.dumps(dict(source=source,subject=sid,trajectory=trajectory,checkpoint=str(checkpoint),lr=.001,horizon=4,pulses=len(rows),labels_used_future=len(used),actual_trajectory_metrics=metrics(y,np.array(pred)),max_counterfactual_current_error=max_current_error,nonwritable_state_unchanged=True,seconds=time.perf_counter()-start,source_controller_only=True,feature_count=435,proposal_state_restored_exact=True),indent=2));print("RETENTION PULSES",source,sid,trajectory,len(rows),time.perf_counter()-start,flush=True)
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--source",required=True);parser.add_argument("--subject",type=int,required=True);parser.add_argument("--trajectory",required=True);a=parser.parse_args();run(a.source,a.subject,a.trajectory)
