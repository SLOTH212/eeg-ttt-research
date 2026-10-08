from phase22_source_screen import *
OUT24=ROOT/"phase24";checks=[]
for source in ["young","mixed"]:
 model=FullCarrier(source,"pretrained").eval().cuda();model.load_state_dict(torch.load(source_checkpoint(source),map_location="cpu",weights_only=False)["state"]);capture={};hook=model.head.register_forward_pre_hook(lambda module,args:capture.update(z=args[0].detach()))
 for sid in [7,17,27,37]:
  saved=np.load(OUT24/f"{source}_continuous_s{sid:02d}/pulses.npz");x,_=prepared(sid);anchor=np.load(ROOT/"phase21/evaluation"/f"{source}_pretrained"/f"s{sid:02d}.npz")["source_tuned"]
  for idx in [0,len(saved["t"])//2,len(saved["t"])-1]:
   t=int(saved["t"][idx]);block=x[t:t+5].cuda();changed=block.clone();changed[1:]*=1.7
   with torch.inference_mode():
    model(block);z=capture["z"][0].cpu().numpy().copy();model(changed);altered=capture["z"][0].cpu().numpy().copy()
   assert np.array_equal(z,altered);stored=saved["features"][idx,35:];error=float(np.max(np.abs(z-stored)));assert error<.002
   for alpha,start in [(.2,25),(.8,30)]:
    h=None
    for q in anchor[:t+1]:h=q.copy() if h is None else alpha*q+(1-alpha)*h
    assert np.array_equal(h,saved["features"][idx,start:start+5])
   checks.append(dict(source=source,subject=sid,t=t,current_embedding_future_perturbation_exact=True,stored_embedding_max_error=error,history_prefix_exact=True))
 hook.remove()
(OUT24/"feature_causality_audit.json").write_text(json.dumps(dict(status="PASS",checks=checks,scope="24 source-controller pulse samples; current source embedding invariant to changing future samples, source histories equal prefix-only computation"),indent=2));print("FEATURE CAUSALITY PASS",len(checks),max(v["stored_embedding_max_error"] for v in checks),flush=True)
