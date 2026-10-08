from phase30_masked_graph import *
from phase37_screen import load as load_carrier
def prefix_history(p,alpha):
 h=p[0].copy()
 for q in p[1:]:h=alpha*q+(1-alpha)*h
 return h
if __name__=="__main__":
 out=ROOT/"phase43";model=load_carrier("mixed","fitted")[0];captured={};hook=model.head.register_forward_pre_hook(lambda module,args:captured.update(z=args[0].detach().clone()));rows=[]
 for sid in [3,4,7,12,13,14,17,27,37]:
  cache=np.load(out/"cache"/f"s{sid:02d}.npz");raw=np.load(ROOT/("data/processed" if sid<20 else "phase2/processed")/f"s{sid:02d}_x.npy",mmap_mode="r")
  for t in np.linspace(0,len(raw)-5,3,dtype=int):
   x=preprocess(np.array(raw[t:t+5])).reshape(-1,2,2,15,200).cuda();changed=x.clone();changed[1:]*=-1.7
   with torch.no_grad():
    p=model(x).softmax(1);z=captured["z"][0].cpu().numpy().copy();pp=model(changed).softmax(1);zz=captured["z"][0].cpu().numpy().copy()
   assert torch.equal(p[0],pp[0]);assert np.array_equal(z,zz);err=float(abs(z-cache["embeddings"][t]).max());assert np.allclose(z,cache["embeddings"][t],atol=2e-4,rtol=2e-5);assert abs(p[0].cpu().numpy()-cache["p"][t]).max()<2e-5;rows.append(dict(subject=sid,t=int(t),future_perturbation_current_probability_error=0.,future_perturbation_current_embedding_error=0.,cache_embedding_max_error=err))
 hook.remove();checked=0
 for folder in out.glob("lr*_s*"):
  if not (folder/"complete.json").exists():continue
  meta=json.loads((folder/"complete.json").read_text());sid=meta["subject"];p=np.load(out/"cache"/f"s{sid:02d}.npz");d=np.load(folder/"pulses.npz")
  for idx in np.linspace(0,len(d["t"])-1,3,dtype=int):
   t=int(d["t"][idx]);f=d["features"][idx];assert np.array_equal(f[35:435],p["embeddings"][t]);assert np.array_equal(f[25:30],prefix_history(p["p"][:t+1],.2));assert np.array_equal(f[30:35],prefix_history(p["p"][:t+1],.8));checked+=1
 (out/"feature_causality_audit.json").write_text(json.dumps(dict(status="PASS",future_perturbation_cases=rows,stored_feature_prefix_checks=checked,task_labels_read=False,scope="Current fixed-source embedding/probabilities independent of altered future inputs; stored history equals prefix-only EMA. Dynamic update causality separately audited inphase30."),indent=2));print("FEATURE AUDIT PASS",len(rows),checked,flush=True)
