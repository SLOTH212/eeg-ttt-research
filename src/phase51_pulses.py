from phase30_masked_graph import *
from phase29_forward_graph import LogitForward
from phase37_screen import load as load_carrier
import hashlib
import phase50_scope_online as scope_online
scope_online.OUT50=ROOT/"phase51"
OUT32=ROOT/"phase51"
def entropy(p):return float(-(p*np.log(np.clip(p,1e-12,1))).sum())
def run(sid,lr,trajectory):
 out=OUT32/f"lr{lr:g}_{trajectory}_s{sid:02d}";out.mkdir(exist_ok=True)
 if (out/"complete.json").exists():return
 model=scope_online.model_for("masked");x,y=prepared(sid);data_root=ROOT/("data/processed" if sid<20 else "phase2/processed");targets=np.load(data_root/f"s{sid:02d}_meta.npz")["features"][:,24:48];cache=OUT32/"cache"/f"s{sid:02d}.npz";checkpoint=ROOT/"phase28/mixed_joint_0/model.pt";sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
 if cache.exists():
  saved=np.load(cache);anchor=saved["p"];embeddings=saved["embeddings"];assert str(saved["checkpoint_sha256"])==sha
 else:
  captured=[];hook=model.head.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().cpu().numpy().copy()));anchor=probabilities(model,x);hook.remove();embeddings=np.concatenate(captured);np.savez_compressed(cache,p=anchor,embeddings=embeddings,checkpoint_sha256=sha)
 root=ROOT/("data/processed" if sid<20 else "phase2/processed");spectral=np.load(root/f"s{sid:02d}_meta.npz")["features"];histories={}
 for alpha in [.2,.8]:
  h=None;history=[]
  for q in anchor:h=q.copy() if h is None else alpha*q+(1-alpha)*h;history.append(h.copy())
  histories[alpha]=np.array(history)
 assert lr==.01;assert json.loads((OUT32/"graph_audit.json").read_text())["status"]=="PASS";graph=scope_online.SpectralGraph(model,"future_groups",x[:1].cuda());one=LogitForward(model,x[:1].cuda());five=LogitForward(model,x[:5].cuda());params=graph.params;frozen={n:v.detach().clone() for n,v in model.state_dict().items() if not dict(model.named_parameters()).get(n,torch.tensor(0.)).requires_grad};rows=[];pred=[];used=set();max_error=0.;start=time.perf_counter()
 def restore(state):
  with torch.no_grad():
   for p,v in zip(params,state):p.copy_(v)
 for t in range(len(y)):
  if trajectory=="reset64" and t%64==0:graph.reset()
  pulse=t%20==0 and t+4<len(y);before=[p.detach().clone() for p in params] if pulse else None;xt=x[t:t+1].cuda();q=graph.step(xt,torch.tensor(targets[t:t+1],device="cuda",dtype=torch.float32)).detach().cpu().numpy()[0].copy();pred.append(q);assert bool(graph.finite)
  if not pulse:continue
  proposal=[p.detach().clone() for p in params]
  with torch.no_grad():after=one.run(xt).softmax(1).cpu().numpy()[0]
  drift=float(torch.sqrt(sum((v-s).square().sum() for v,s in zip(before,graph.initial))))
  features=[entropy(q),float(graph.norm.detach()),float(np.abs(after-q).sum()),drift,float(q.max()),entropy(after),*q.tolist(),float(np.mean(np.abs(spectral[t]-spectral[t-1]))) if t else 0.,float(t==0),*anchor[t].tolist(),entropy(anchor[t]),float(np.abs(q-anchor[t]).sum()),*after.tolist(),*histories[.2][t].tolist(),*histories[.8][t].tolist(),float(graph.loss.detach())];assert len(features)==36
  if np.any(y[t:t+5]<0):continue
  views={}
  for action,state in [("commit",proposal),("keep",before),("reset",graph.initial)]:
   restore(state)
   with torch.no_grad():views[action]=five.run(x[t:t+5].cuda()).log_softmax(1).cpu().numpy().copy()
  restore(proposal);assert all(torch.equal(p,v) for p,v in zip(params,proposal));max_error=max(max_error,float(abs(np.exp(views["keep"][0])-q).max()));losses={k:-v[np.arange(5),y[t:t+5]] for k,v in views.items()};future=[losses["keep"][1:].mean()-losses["reset"][1:].mean(),losses["keep"][1:].mean()-losses["commit"][1:].mean()];current=[losses["keep"][0]-losses["reset"][0],losses["keep"][0]-losses["commit"][0]]
  if drift==0:assert future[0]==0 and current[0]==0
  rows.append(dict(t=t,features=features,future_gain=future,current_gain=current,current_class=int(y[t]),stable=bool(np.all(y[t+1:t+5]==y[t])),current_correct=bool(q.argmax()==y[t]),future_labels=y[t+1:t+5].tolist(),future_keep=np.exp(views["keep"][1:]).tolist(),future_commit=np.exp(views["commit"][1:]).tolist(),future_reset=np.exp(views["reset"][1:]).tolist()));used.update(range(t+1,t+5))
 assert max_error<2e-5;assert all(torch.equal(model.state_dict()[n],v) for n,v in frozen.items());np.savez_compressed(out/"actual_predictions.npz",y=y,p=np.array(pred));np.savez_compressed(out/"pulses.npz",features=np.array([r["features"] for r in rows]),future_gain=np.array([r["future_gain"] for r in rows]),current_gain=np.array([r["current_gain"] for r in rows]),t=np.array([r["t"] for r in rows]),current_class=np.array([r["current_class"] for r in rows]),stable=np.array([r["stable"] for r in rows]),current_correct=np.array([r["current_correct"] for r in rows]),future_labels=np.array([r["future_labels"] for r in rows]),future_keep=np.array([r["future_keep"] for r in rows]),future_commit=np.array([r["future_commit"] for r in rows]),future_reset=np.array([r["future_reset"] for r in rows]),future_labels_used=np.array(sorted(used)))
 (out/"complete.json").write_text(json.dumps(dict(subject=sid,lr=lr,trajectory=trajectory,pulses=len(rows),scope_selection_sha256=hashlib.sha256((OUT32/"selection.json").read_bytes()).hexdigest(),checkpoint_sha256=sha,feature_count=36,source_controller_only=True,proposal_restored_exact=True,nonwritable_unchanged=True,max_counterfactual_current_error=max_error,actual_metrics=metrics(y,np.array(pred)),seconds=time.perf_counter()-start),indent=2));print("MASKED RETENTION DIAGNOSTIC",sid,lr,trajectory,len(rows),flush=True)
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--subject",type=int,required=True);parser.add_argument("--lr",type=float,required=True);parser.add_argument("--trajectory",required=True);a=parser.parse_args();run(a.subject,a.lr,a.trajectory)
