from phase54_pulses import *
OUT57=ROOT/"phase57"
def run():
 protocol=json.loads((OUT57/"protocol.json").read_text());assert hashlib.sha256((ROOT/"phase52/current/model.pt").read_bytes()).hexdigest()==protocol["initial_head_sha256"];m=load_meta("current");x0,_=prepared(3);graph=scope_online.SpectralGraph(m,"allLN",x0[:1].cuda());layout=json.loads((ROOT/"phase48/parameter_layout.json").read_text());names=[n for n,p in m.named_parameters() if p.requires_grad];assert names==[p["name"] for p in layout["parameters"]];fixed={n:v.clone() for n,v in m.state_dict().items() if n not in names};initial=torch.cat([p.flatten() for p in graph.initial]).cpu().numpy();np.save(OUT57/"initial_params.npy",initial);cases=[]
 for split,subjects in [("train",protocol["training_subjects"]),("held",protocol["held_subjects"])]:
  (OUT57/"state_pool"/split).mkdir(parents=True,exist_ok=True)
  for sid in subjects:
   x,y=prepared(sid);root=ROOT/("data/processed" if sid<20 else "phase2/processed");target=np.load(root/f"s{sid:02d}_meta.npz")["features"][:,24:48]
   times=np.load(ROOT/"phase48/masked"/f"s{sid:02d}_gradients.npz")["times"] if split=="train" else np.load(ROOT/"phase54"/f"lr0.01_continuous_s{sid:02d}/pulses.npz")["t"];selected=set(times.tolist())
   for trajectory in ["continuous","reset64"]:
    folder=OUT57/"state_pool"/split;path=folder/f"s{sid:02d}_{trajectory}.npz";done=path.with_suffix(".json")
    if done.exists():cases.append(json.loads(done.read_text()));continue
    graph.reset();states=[];pred=[];margins=[];start=time.perf_counter()
    for t in range(len(x)):
     if trajectory=="reset64" and t%64==0:graph.reset()
     if t in selected:
      states.append(torch.cat([p.detach().flatten() for p in graph.params]).cpu().numpy().copy());margins.append(min(r-float((p-v).norm()) for p,v,r in zip(graph.params,graph.initial,graph.radii)))
     q=graph.step(x[t:t+1].cuda(),torch.tensor(target[t:t+1],device="cuda",dtype=torch.float32)).detach().cpu().numpy()[0].copy();assert bool(graph.finite);pred.append(q)
    assert len(states)==len(times);assert all(torch.equal(v,m.state_dict()[n]) for n,v in fixed.items());error=None
    if split=="held":
     ref=np.load(ROOT/"phase54"/f"lr0.01_{trajectory}_s{sid:02d}/actual_predictions.npz")["p"];error=float(abs(np.array(pred)-ref).max());assert error<2e-5;assert np.array_equal(np.array(pred).argmax(1),ref.argmax(1))
    graph.reset();assert np.array_equal(torch.cat([p.detach().flatten() for p in graph.params]).cpu().numpy(),initial);np.savez_compressed(path,times=times,states=np.array(states),generator_head_sha256=protocol["initial_head_sha256"]);rec=dict(subject=sid,split=split,trajectory=trajectory,states=len(times),shape=list(np.array(states).shape),minimum_projection_margin=float(min(margins)),head_and_nonwritable_unchanged=True,source_state_restored_exact=True,held_prediction_replay_error=error,class_labels_not_used_in_rollout=True,seconds=time.perf_counter()-start);done.write_text(json.dumps(rec,indent=2));cases.append(rec);print("META STATE POOL",rec,flush=True)
 (OUT57/"state_pool_complete.json").write_text(json.dumps(dict(cases=cases,source_head="phase52/current",task_labels_not_used_to_generate_states=True,parameter_layout_sha256=hashlib.sha256((ROOT/"phase48/parameter_layout.json").read_bytes()).hexdigest()),indent=2))
if __name__=="__main__":run()
