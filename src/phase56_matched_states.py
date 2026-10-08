from phase54_pulses import *
OUT56=ROOT/"phase56"
def run():
 assert json.loads((ROOT/"phase54/diagnostic_audit.json").read_text())["status"]=="PASS";assert json.loads((ROOT/"phase54/feature_causality_audit.json").read_text())["status"]=="PASS";assert hashlib.sha256((ROOT/"phase52/current/model.pt").read_bytes()).hexdigest()==json.loads((OUT56/"protocol.json").read_text())["head_sha256"];m=load_meta("current");x0,_=prepared(7);graph=scope_online.SpectralGraph(m,"allLN",x0[:1].cuda());forward=LogitForward(m,x0[:5].cuda());cw=np.array(json.loads((ROOT/"phase36/mixed_class_weights.json").read_text())["balanced_weights"]);state={n:v.clone() for n,v in m.state_dict().items()};total=0
 for sid in [7,17,27,37]:
  done=OUT56/f"s{sid:02d}.json"
  if done.exists():total+=len(json.loads(done.read_text())["rows"]);continue
  x,y=prepared(sid);root=ROOT/("data/processed" if sid<20 else "phase2/processed");target=np.load(root/f"s{sid:02d}_meta.npz")["features"][:,24:48];traces={name:np.load(ROOT/"phase54"/f"lr0.01_{name}_s{sid:02d}/pulses.npz") for name in ["continuous","reset64"]};assert np.array_equal(traces["continuous"]["t"],traces["reset64"]["t"]);times=traces["continuous"]["t"];rows=[];old={r["t"]:r for r in json.loads((ROOT/"phase52/evaluation/current"/f"s{sid:02d}.json").read_text())["rows"]}
  for index,t0 in enumerate(times):
   t=int(t0);graph.reset();xx=x[t:t+5].cuda()
   with torch.no_grad():base=forward.run(xx).softmax(1).cpu().numpy().copy()
   graph.step(xx[:1],torch.tensor(target[t:t+1],device="cuda",dtype=torch.float32))
   with torch.no_grad():write=forward.run(xx).softmax(1).cpu().numpy().copy()
   yy=y[t+1:t+5];before=-np.log(base[1:,][np.arange(4),yy]);after=-np.log(write[1:,][np.arange(4),yy]);cold_gain=float((before-after).mean());cold_bal=float(((before-after)*cw[yy]).mean());cold_correct=int((write[1:].argmax(1)==yy).sum())-int((base[1:].argmax(1)==yy).sum());replay=None
   if t in old:replay=float(abs(write[1:]-np.array(old[t]["future_write_probabilities"])).max());assert replay<2e-5
   for trajectory,z in traces.items():
    assert np.array_equal(yy,z["future_labels"][index]);reset=z["future_reset"][index];err=float(abs(reset-base[1:]).max());assert err<2e-5;keep=z["future_keep"][index];commit=z["future_commit"][index];kl=-np.log(keep[np.arange(4),yy]);cl=-np.log(commit[np.arange(4),yy]);adapted_gain=float((kl-cl).mean());history_gain=float((before-kl).mean());history_bal=float(((before-kl)*cw[yy]).mean());correct=int((commit.argmax(1)==yy).sum())-int((keep.argmax(1)==yy).sum())
    rows.append(dict(subject=sid,t=t,trajectory=trajectory,current_class=int(y[t]),transition=bool(np.any(yy!=y[t])),cold_write_gain=cold_gain,adapted_write_gain=adapted_gain,change_in_write_utility=adapted_gain-cold_gain,history_keep_gain=history_gain,cold_balanced_gain=cold_bal,adapted_balanced_gain=float(((kl-cl)*cw[yy]).mean()),history_balanced_gain=history_bal,cold_correct_gain=cold_correct,adapted_correct_gain=correct,drift=float(z["features"][index,3]),cold_aux_loss=float(graph.loss.detach()),adapted_aux_loss=float(z["features"][index,35]),cold_gradient_norm=float(graph.norm.detach()),adapted_gradient_norm=float(z["features"][index,1]),source_reset_probability_error=err,old_cold_replay_error=replay,future_labels=yy.tolist(),cold_keep_probabilities=base[1:].tolist(),cold_write_probabilities=write[1:].tolist(),adapted_keep_probabilities=keep.tolist(),adapted_write_probabilities=commit.tolist()))
  graph.reset();assert all(torch.equal(v,m.state_dict()[n]) for n,v in state.items());done.write_text(json.dumps(dict(subject=sid,matched_same_times_inputs_targets=True,model_restored_exact=True,rows=rows),indent=2));total+=len(rows);print("MATCHED STATE WRITES",sid,len(rows),flush=True)
 (OUT56/"complete.json").write_text(json.dumps(dict(rows=total,head="phase52/current",source_only=True),indent=2))
if __name__=="__main__":run()
