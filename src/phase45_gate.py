from phase30_masked_graph import *
from phase37_screen import load as load_carrier
from phase29_forward_graph import LogitForward
from phase2_temporal import ema
import pickle,hashlib
OUT=ROOT/"phase45"
def entropy(q):return float(-(q*np.log(np.clip(q,1e-12,1))).sum())
def make_cache():
 model,_=load_carrier("mixed","fitted")
 for sid in [5,15,25,35]:
  path=OUT/"cache"/f"s{sid:02d}.npz"
  if path.exists():continue
  x=torch.from_numpy(np.array(np.load(ROOT/"phase17/cache"/f"s{sid:02d}_x.npy",mmap_mode="r")));captured=[];hook=model.head.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().cpu().numpy().copy()));p=probabilities(model,x);hook.remove();z=np.concatenate(captured);ref=np.load(ROOT/"phase35/mixed"/f"s{sid:02d}.npz");assert np.array_equal(p,ref["p"]);assert z.shape==(len(p),400);np.savez_compressed(path,p=p,embeddings=z);print("POLICY CACHE",sid,flush=True)
def rollout(policy,sid,n=None,alter_after=None,capture_before=None):
 saved=np.load(OUT/"cache"/f"s{sid:02d}.npz");base=saved["p"].copy();embedding=saved["embeddings"].copy();x=torch.from_numpy(np.array(np.load(ROOT/"phase17/cache"/f"s{sid:02d}_x.npy",mmap_mode="r")));tokens=np.load(ROOT/"phase28/tokens"/f"s{sid:02d}.npz")["tokens"].copy();root=ROOT/("data/processed" if sid<20 else "phase2/processed");spectral=np.load(root/f"s{sid:02d}_meta.npz")["features"].copy()
 if n is not None:x=x[:n];base=base[:n];embedding=embedding[:n];tokens=tokens[:n];spectral=spectral[:n]
 if alter_after is not None:
  x[alter_after:]*=-1.7;tokens[alter_after:]=(tokens[alter_after:].astype(np.int64)+1)%8192;base[alter_after:]=np.roll(base[alter_after:],1,axis=1);embedding[alter_after:]*=-1.3;spectral[alter_after:]*=2
 histories={}
 for alpha in [.2,.8]:
  h=None;rows=[]
  for q in base:h=q.copy() if h is None else alpha*q+(1-alpha)*h;rows.append(h.copy())
  histories[alpha]=np.array(rows)
 gate=None
 if policy in ["future36","current36"]:
  target="ordinary_future" if policy=="future36" else "ordinary_current"
  with (ROOT/"phase43/gates"/f"{target}_direction_history36.pkl").open("rb") as f:gate=pickle.load(f)
 model,_=load_carrier("mixed","fitted");graph=MaskedGraphAdapter(model,.01,x[:1].cuda());forward=LogitForward(model,x[:1].cuda()) if gate is not None else None;writable={n for n,p in model.named_parameters() if p.requires_grad};frozen={n:v.detach().clone() for n,v in model.state_dict().items() if n not in writable};pred=[];actions=[];prefix=None;start=time.perf_counter()
 for t in range(len(x)):
  if capture_before==t:prefix=torch.cat([p.detach().flatten() for p in graph.params]).cpu().numpy().copy()
  before=[p.detach().clone() for p in graph.params] if policy!="anchor_guard" else None;xt=x[t:t+1].cuda();q=graph.step(xt,torch.tensor(tokens[t:t+1].astype(np.int64),device="cuda")).detach().cpu().numpy()[0].copy();pred.append(q);assert bool(graph.finite)
  if gate is not None:
   with torch.no_grad():after=forward.run(xt).softmax(1).cpu().numpy()[0].copy()
   drift=float(torch.sqrt(sum((v-s).square().sum() for v,s in zip(before,graph.initial))));features=[entropy(q),float(graph.norm.detach()),float(abs(after-q).sum()),drift,float(q.max()),entropy(after),*q.tolist(),float(np.mean(abs(spectral[t]-spectral[t-1]))) if t else 0.,float(t==0),*base[t].tolist(),entropy(base[t]),float(abs(q-base[t]).sum()),*after.tolist(),*histories[.2][t].tolist(),*histories[.8][t].tolist(),*embedding[t].tolist(),float(graph.loss.detach())];assert len(features)==436;obs=np.array(features)[None];values=gate["model"].predict(gate["scaler"].transform(obs[:,gate["indices"]]));action=2 if values[0,1]>0 else 0
  elif policy=="low_entropy_write":action=2 if entropy(q)<.5 else 0
  elif policy=="anchor_guard":action=1 if abs(q-base[t]).sum()>.05 else 2
  elif policy.startswith("cadence"):action=2 if t%int(policy.removeprefix("cadence"))==0 else 0
  else:raise ValueError(policy)
  if action==1:graph.reset()
  elif action==0:
   with torch.no_grad():
    for p,v in zip(graph.params,before):p.copy_(v)
  actions.append(action)
 pred=np.array(pred);assert abs(pred[0]-base[0]).max()<2e-5;assert all(torch.equal(model.state_dict()[n],v) for n,v in frozen.items());assert all(float((p-v).detach().norm())<=radius+1e-5 for p,v,radius in zip(graph.params,graph.initial,graph.radii));return pred,np.array(actions,dtype=np.int8),prefix,time.perf_counter()-start
def audit():
 rows=[]
 for policy in ["future36","current36"]:
  a,aa,pa,_=rollout(policy,5,n=64,capture_before=32);b,bb,pb,_=rollout(policy,5,n=64,alter_after=32,capture_before=32);err=float(abs(a[:32]-b[:32]).max());perr=float(abs(pa-pb).max());assert err==0 and perr==0 and np.array_equal(aa[:32],bb[:32]);rows.append(dict(policy=policy,probability_prefix_error=err,parameter_prefix_error=perr,action_prefix_exact=True))
 (OUT/"closed_loop_audit.json").write_text(json.dumps(dict(status="PASS",future_perturbed_from_window=32,cases=rows,labels_not_passed_to_rollout=True),indent=2));print("CLOSED LOOP AUDIT PASS",flush=True)
def run(policy,sid):
 assert json.loads((OUT/"closed_loop_audit.json").read_text())["status"]=="PASS";out=OUT/policy;out.mkdir(exist_ok=True);done=out/f"s{sid:02d}.json"
 if done.exists():return
 pred,actions,_,seconds=rollout(policy,sid);root=ROOT/("data/processed" if sid<20 else "phase2/processed");y=np.load(root/f"s{sid:02d}_meta.npz")["y"];base=np.load(OUT/"cache"/f"s{sid:02d}.npz")["p"];np.savez_compressed(out/f"s{sid:02d}.npz",y=y,p=pred,actions=actions);rec=dict(subject=sid,policy=policy,metrics=metrics(y,pred),adapted_ema=metrics(y,ema(pred,.8)),frozen=metrics(y,base),weak_ema=metrics(y,ema(base,.8)),strong_ema=metrics(y,ema(base,.5)),write_fraction=float((actions==2).mean()),reset_fraction=float((actions==1).mean()),nonwritable_unchanged=True,radius_verified=True,seconds=seconds,source_validation_only=True);done.write_text(json.dumps(rec,indent=2));print("ACTUAL POLICY",policy,sid,rec["metrics"]["macro_f1"],rec["write_fraction"],seconds,flush=True)
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--prepare",action="store_true");parser.add_argument("--policy");parser.add_argument("--subject",type=int);a=parser.parse_args()
 if a.prepare:make_cache();audit()
 else:run(a.policy,a.subject)
