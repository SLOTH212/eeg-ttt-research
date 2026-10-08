from phase54_pulses import *
from phase57_factorial_meta import set_flat,VARIANTS
OUT57=ROOT/"phase57"
def load_variant(variant):
 m=scope_online.model_for("masked");path=ROOT/"phase52/current/model.pt" if variant=="original" else OUT57/variant/"model.pt";m.load_state_dict(torch.load(path,map_location="cpu",weights_only=False)["state"]);ref=torch.load(ROOT/"phase52/current/model.pt",map_location="cpu",weights_only=False)["state"];assert all(torch.equal(v.cpu(),ref[n]) for n,v in m.state_dict().items() if not n.startswith("recon_head."));return m
def validity():
 rows=[]
 for variant in ["original"]+VARIANTS:
  m=load_variant(variant)
  for sid in [5,15,25,35]:
   z=np.load(ROOT/"phase46/cache"/f"s{sid:02d}.npz");xx=torch.tensor(z["masked"],device="cuda");target=torch.tensor(z["target"],device="cuda")
   with torch.no_grad():
    x=(xx-m.recon_input_mean)/m.recon_input_scale;y=(target-m.recon_target_mean)/m.recon_target_scale;pr=m.recon_head(x);sh=m.recon_head(torch.roll(x,len(x)//2,0));mse=float((pr-y).square().mean());mean=float(y.square().mean());shuffle=float((sh-y).square().mean())
   rows.append(dict(variant=variant,subject=sid,mse=mse,mean_mse=mean,shuffle_mse=shuffle,validity_pass=mse<mean and mse<shuffle))
  del m
 import pandas as pd
 pd.DataFrame(rows).to_csv(OUT57/"reconstruction_validity.csv",index=False);print(pd.DataFrame(rows).to_string(index=False),flush=True)
def run(variant):
 out=OUT57/"evaluation"/variant;out.mkdir(parents=True,exist_ok=True)
 if (out/"complete.json").exists():return
 m=load_variant(variant);x0,_=prepared(7);graph=scope_online.SpectralGraph(m,"allLN",x0[:1].cuda());forward=LogitForward(m,x0[:5].cuda());source={n:v.clone() for n,v in m.state_dict().items()};cw=np.array(json.loads((ROOT/"phase36/mixed_class_weights.json").read_text())["balanced_weights"]);total=0
 for sid in [7,17,27,37]:
  done=out/f"s{sid:02d}.json"
  if done.exists():total+=len(json.loads(done.read_text())["rows"]);continue
  x,y=prepared(sid);root=ROOT/("data/processed" if sid<20 else "phase2/processed");target=np.load(root/f"s{sid:02d}_meta.npz")["features"][:,24:48];pools={k:np.load(OUT57/"state_pool/held"/f"s{sid:02d}_{k}.npz") for k in ["continuous","reset64"]};traces={k:np.load(ROOT/"phase54"/f"lr0.01_{k}_s{sid:02d}/pulses.npz") for k in pools};cold={r["t"]:r for r in json.loads((ROOT/"phase56"/f"s{sid:02d}.json").read_text())["rows"] if r["trajectory"]=="continuous"};times=pools["continuous"]["times"];assert np.array_equal(times,pools["reset64"]["times"]);rows=[]
  for index,t0 in enumerate(times):
   t=int(t0);xx=x[t:t+5].cuda();yy=y[t+1:t+5]
   for state in ["fresh","continuous","reset64"]:
    if state=="fresh":graph.reset();expected=np.array(cold[t]["cold_keep_probabilities"]);original_write=np.array(cold[t]["cold_write_probabilities"])
    else:set_flat(m,pools[state]["states"][index]);expected=traces[state]["future_keep"][index];original_write=traces[state]["future_commit"][index]
    before_params=[p.detach().clone() for p in graph.params]
    with torch.no_grad():bp=forward.run(xx).softmax(1).cpu().numpy()[1:].copy()
    error=float(abs(bp-expected).max());assert error<2e-5;assert np.array_equal(bp.argmax(1),expected.argmax(1));graph.step(xx[:1],torch.tensor(target[t:t+1],device="cuda",dtype=torch.float32));assert bool(graph.finite)
    with torch.no_grad():wp=forward.run(xx).softmax(1).cpu().numpy()[1:].copy();stepnorm=float(torch.sqrt(sum((p-v).square().sum() for p,v in zip(graph.params,before_params))))
    assert stepnorm<.01001;before=-np.log(bp[np.arange(4),yy]);after=-np.log(wp[np.arange(4),yy]);gain=before-after;replay=None
    if variant=="original":replay=float(abs(wp-original_write).max());assert replay<2e-5;assert np.array_equal(wp.argmax(1),original_write.argmax(1))
    b=bp.argmax(1)==yy;w=wp.argmax(1)==yy;rows.append(dict(variant=variant,subject=sid,t=t,state=state,future_ce_gain=float(gain.mean()),balanced_future_ce_gain=float((gain*cw[yy]).mean()),accuracy_gain_pp=float((w.astype(float)-b).mean()*100),corrected=int((~b&w).sum()),harmed=int((b&~w).sum()),changed=int((bp.argmax(1)!=wp.argmax(1)).sum()),gradient_norm=float(graph.norm.detach()),aux_loss=float(graph.loss.detach()),actual_update_norm=stepnorm,base_probability_error=error,original_write_replay_error=replay,future_labels=yy.tolist(),base_probabilities=bp.tolist(),write_probabilities=wp.tolist()))
  graph.reset();assert all(torch.equal(v,m.state_dict()[n]) for n,v in source.items());done.write_text(json.dumps(dict(variant=variant,subject=sid,model_restored_exact=True,identical_held_state_inputs=True,rows=rows),indent=2));total+=len(rows);print("FACTORIAL HELD",variant,sid,len(rows),flush=True)
 (out/"complete.json").write_text(json.dumps(dict(variant=variant,rows=total),indent=2))
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--validity",action="store_true");parser.add_argument("--variant");a=parser.parse_args()
 if a.validity:validity()
 else:run(a.variant)
