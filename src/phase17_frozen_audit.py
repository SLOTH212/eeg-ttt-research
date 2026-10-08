from phase16_masked_pulses import *
model=make_encoder(True);OUT=ROOT/"phase17";carriers={}
for source in ["young","mixed"]:
 with (ROOT/"phase14/probes"/f"{source}_pretrained400/model.pkl").open("rb") as f:carriers[source]=pickle.load(f)
rows=[]
for sid in [5,15,25,35]:
 xs=np.load(OUT/"cache"/f"s{sid:02d}_x.npy",mmap_mode="r");zs=[];start=time.perf_counter()
 with torch.inference_mode():
  for t in range(len(xs)):
   x=torch.from_numpy(np.array(xs[t]));z=model.forward_features(x,input_chans=[0,127,128],return_patch_tokens=True).mean(1).reshape(400);zs.append(z.numpy())
 z=np.array(zs);cached=np.load(ROOT/"phase14/features"/f"s{sid:02d}.npz")["pretrained"];root=ROOT/("data/processed" if sid<20 else "phase2/processed");y=np.load(root/f"s{sid:02d}_meta.npz")["y"];preds={}
 for source,carrier in carriers.items():
  scaler=carrier["scaler"];head=carrier["model"];baseline=head.predict_proba(scaler.transform(cached));W=torch.tensor(head.coef_,dtype=torch.float32);b=torch.tensor(head.intercept_,dtype=torch.float32);mean=torch.tensor(scaler.mean_,dtype=torch.float32);scale=torch.tensor(scaler.scale_,dtype=torch.float32)
  # Batch head GEMM can itself change rounding; mirror per-epoch online head call exactly.
  with torch.inference_mode():pr=np.array([nn.functional.linear((torch.from_numpy(v[None])-mean)/scale,W,b).softmax(1).numpy()[0] for v in z])
  row=dict(subject=sid,source=source,argmax_disagreements=int((baseline.argmax(1)!=pr.argmax(1)).sum()),max_probability_error=float(np.max(np.abs(baseline-pr))),cached_f1=metrics(y,baseline)["macro_f1"],streaming_f1=metrics(y,pr)["macro_f1"]);rows.append(row);preds[source]=pr;print("FROZEN LN AUDIT",row,flush=True)
 np.savez_compressed(OUT/f"frozen_stream_s{sid:02d}.npz",y=y,**preds);print("AUDIT SUBJECT",sid,time.perf_counter()-start,flush=True)
(OUT/"frozen_audit.json").write_text(json.dumps(dict(checks=rows,total_argmax_disagreements=sum(v["argmax_disagreements"] for v in rows)),indent=2))
