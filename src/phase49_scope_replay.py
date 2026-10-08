from phase47_pulses import model_for
from phase29_forward_graph import LogitForward
from phase46_spectral_head import *
OUT49=ROOT/"phase49"
def run(mode):
 assert json.loads((ROOT/"phase48/audit.json").read_text())["status"]=="PASS";out=OUT49/mode;out.mkdir(exist_ok=True)
 if (out/"complete.json").exists():return
 selection=json.loads((OUT49/"scopes.json").read_text());layout=json.loads((ROOT/"phase48/parameter_layout.json").read_text());model=model_for(mode);named=[(n,p) for n,p in model.named_parameters() if p.requires_grad];assert [n for n,p in named]==[p["name"] for p in layout["parameters"]];params=[p for n,p in named];initial=[p.detach().clone() for p in params];flat_initial=torch.cat([v.flatten() for v in initial]);state={k:v.clone() for k,v in model.state_dict().items()};forward=LogitForward(model,torch.zeros(5,2,2,15,200,device="cuda"));groups=np.concatenate([np.full(p["length"],p["group"],dtype=int) for p in layout["parameters"]]);cw=torch.tensor(json.loads((ROOT/"phase36/mixed_class_weights.json").read_text())["balanced_weights"],dtype=torch.float32,device="cuda");total=0;max_replay=0.;start=time.perf_counter()
 def apply(delta):
  with torch.no_grad():
   values=[v+delta[m["offset"]:m["offset"]+m["length"]].view(m["shape"]) for v,m in zip(initial,layout["parameters"])];torch._foreach_copy_(params,values)
 def restore():
  with torch.no_grad():torch._foreach_copy_(params,initial)
 for sid in [3,4,7,12,13,14,17,27,37]:
  done=out/f"s{sid:02d}.json"
  if done.exists():total+=len(json.loads(done.read_text())["rows"]);continue
  folds={f["objective"]:f for f in selection["folds"] if f["mode"]==mode and f["held_subject"]==sid};assert all(sid not in f["training_subjects"] and len(f["training_subjects"])==8 for f in folds.values());grad=np.load(ROOT/"phase48"/mode/f"s{sid:02d}_gradients.npz");metadata=json.loads((ROOT/"phase48"/mode/f"s{sid:02d}.json").read_text());meta={(r["t"],r["objective"]):r for r in metadata["rows"] if r["group"]==0};ref=json.loads((ROOT/"phase47"/mode/f"s{sid:02d}.json").read_text());reference={(r["t"],r["objective"]):r for r in ref["rows"] if r["lr"]==.01};root=ROOT/("data/processed" if sid<20 else "phase2/processed");raw=np.load(root/f"s{sid:02d}_x.npy",mmap_mode="r");y=np.load(root/f"s{sid:02d}_meta.npz")["y"];rows=[];errors=[]
  for index,tt in enumerate(grad["times"]):
   t=int(tt);restore();xx=preprocess(np.array(raw[t:t+5])).reshape(5,2,2,15,200).cuda();yy=torch.tensor(y[t:t+5],device="cuda")
   with torch.no_grad():logits=forward.run(xx).clone();bp=logits.softmax(1).cpu().numpy();base=-logits.log_softmax(1)[torch.arange(5,device="cuda"),yy]
   for objective,fold in folds.items():
    g=grad[objective][index];gg=g.astype(np.float64);norm=float(np.linalg.norm(gg));clip=float(meta[t,objective]["clip"]);masks={"allLN":np.ones(10960,bool),"finalLN":groups==12,"lastblock_plus_final":groups>=11,"last3blocks_plus_final":groups>=9,"future_positive_groups":np.isin(groups,fold["future_groups"]),"current_positive_groups":np.isin(groups,fold["current_groups"])};coeff={k:v.astype(np.float32) for k,v in masks.items()};coeff["future_normmatched"]=np.full(10960,float(np.linalg.norm(gg*masks["future_positive_groups"]))/norm if norm else 0.,dtype=np.float32);coeff["current_normmatched"]=np.full(10960,float(np.linalg.norm(gg*masks["current_positive_groups"]))/norm if norm else 0.,dtype=np.float32);g_gpu=torch.tensor(g,device="cuda")
    for scope,factor in coeff.items():
     delta=-.01*clip*g_gpu*torch.tensor(factor,device="cuda");apply(delta)
     with torch.no_grad():logits=forward.run(xx);pr=logits.softmax(1).cpu().numpy();post=-logits.log_softmax(1)[torch.arange(5,device="cuda"),yy];actual_norm=float(torch.linalg.vector_norm(torch.cat([p.flatten() for p in params])-flat_initial))
     proposed_norm=float(.01*clip*np.linalg.norm(gg*factor.astype(np.float64)));assert actual_norm<=.01001
     if scope=="allLN":
      error=float(abs(pr[1:]-np.array(reference[t,objective]["future_write_probabilities"])).max());assert error<2e-5;assert np.array_equal(pr[1:].argmax(1),np.array(reference[t,objective]["future_write_probabilities"]).argmax(1));errors.append(error);max_replay=max(max_replay,error)
     rows.append(dict(mode=mode,objective=objective,subject=sid,t=t,scope=scope,lr=.01,proposed_update_norm=proposed_norm,actual_update_norm=actual_norm,future_ce_gain=float((base[1:]-post[1:]).mean()),balanced_future_ce_gain=float(((base[1:]-post[1:])*cw[yy[1:]]).mean()),base_correct=int((bp[1:].argmax(1)==y[t+1:t+5]).sum()),write_correct=int((pr[1:].argmax(1)==y[t+1:t+5]).sum()),future_argmax_changes=int((bp[1:].argmax(1)!=pr[1:].argmax(1)).sum()),future_labels=y[t+1:t+5].tolist(),future_base_probabilities=bp[1:].tolist(),future_write_probabilities=pr[1:].tolist()))
   restore()
  assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items());done.write_text(json.dumps(dict(mode=mode,subject=sid,scope_folds=folds,scope_selection_sha256=hashlib.sha256((OUT49/"scopes.json").read_bytes()).hexdigest(),model_restored_exact=True,max_allLN_replay_probability_error=max(errors),rows=rows),indent=2));total+=len(rows);print("SCOPE REPLAY",mode,sid,len(rows),max(errors),flush=True)
 (out/"complete.json").write_text(json.dumps(dict(mode=mode,pulse_scope_rows=total,model_restored_exact=True,max_allLN_replay_probability_error=max_replay,seconds=time.perf_counter()-start),indent=2))
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--mode",required=True);a=parser.parse_args();run(a.mode)
