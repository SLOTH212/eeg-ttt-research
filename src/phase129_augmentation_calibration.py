from phase86_meta_waveform import *
from phase99_covariance import transform
import datetime
O=ROOT/"phase129";O.mkdir(exist_ok=True);P=O/"predictions";P.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def legacy(c,mode):
 d,s,m=c["domain"],c["subject"],c["montage"]
 if mode=="full":return ROOT/"phase100/predictions"/f"{d}_{s}_{m}_full.npz"
 if d in ["source","healthy"]:return ROOT/"phase79/predictions"/f'{"source" if d=="source" else "external"}_{s}_{m}_{mode}.npz'
 if d=="sleepedf":return ROOT/"phase81/predictions"/f"{s}_{mode}.npz"
 return ROOT/"phase83/predictions"/f"{s}_{m}_{mode}.npz"
def run():
 cases=json.loads((ROOT/"phase85/protocol.json").read_text())["cases"];parents=[s for s in json.loads((ROOT/"phase79/protocol.json").read_text())["models"] if s["model"]=="cbramod"];specs=[]
 for seed in [17,29,43]:
  folder=ROOT/"phase94" if seed==17 else ROOT/"phase96"/f"seed{seed}";selection=json.loads((folder/"selection.json").read_text())["selection"]
  for branch in ["repeat_raw","amplitude_augmented"]:
   path=folder/branch/"model.pt";assert sha(path)==selection[branch]["checkpoint_sha256"];specs.append(dict(seed=seed,branch=branch,path=str(path),sha256=sha(path),selected_epoch=selection[branch]["epoch"]))
 pr=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),purpose="Fixed 2x2 combination of alreadyselected sourceamplitudeaugmentation and alreadyfixedcausal fullshape inputguard; no newtraining or tuning.",cases=cases,models=specs,modes=["raw","full"],primary="KnownISRUCpatient19,left,EMA.5: selectedaugmented+fullshape minus selectedrepeatraw+fullshape. Same3seed metricsaveragedwithinperson; pairedperson10000bootstrapseed129017.",secondary="Allfour factorialcells and descriptive interaction (augfull-augraw)-(repeatfull-repeatraw), healthy10/SleepEDF30/source4 separate,rightmontage andEMA1/.8. No inputsetting or modelselectedby targetscore.",selection="Retain old source-selectedepochs17aug3/29aug0fallback/43aug3; allrepeatraw0. Existingselectionguard/budget unchanged; no final-epoch replacement.",scope="Allcohorts alreadyobserved. This checks composition of fixedmethods, notnewblindvalidation or newsourcefitting. Fullshape includescausal inputstate only; no gradientTTT. Rawaugmentedtargetpredictions independentlycomparephase95/97.",shape_reference_sha256=sha(ROOT/"phase99/source_reference.npz"),cpu_ids=list(range(8,24)),gpu=True)
 pp=O/"protocol.json"
 if pp.exists():
  old=json.loads(pp.read_text());pr["created_utc"]=old["created_utc"];assert old==pr
 else:pp.write_text(json.dumps(pr,indent=2));(O/"protocol.sha256").write_text(sha(pp))
 models={};states={};fallback={}
 for seed in [17,29,43]:
  repeat=next(s for s in specs if s["seed"]==seed and s["branch"]=="repeat_raw");aug=next(s for s in specs if s["seed"]==seed and s["branch"]=="amplitude_augmented");parent=next(s for s in parents if s["seed"]==seed)
  rs=torch.load(repeat["path"],map_location="cpu",weights_only=False)["state"];ps=torch.load(parent["path"],map_location="cpu",weights_only=False)["state"];assert all(torch.equal(v,ps[k]) for k,v in rs.items())
  ast=torch.load(aug["path"],map_location="cpu",weights_only=False)["state"];fallback[seed]=all(torch.equal(v,rs[k]) for k,v in ast.items());assert fallback[seed]==(seed==29)
  if not fallback[seed]:
   m=Carrier("pretrained").cuda();m.load_state_dict(ast);m.eval().requires_grad_(False);models[seed]=m;states[seed]=snapshot(m)
 reference=np.load(ROOT/"phase99/source_reference.npz")["reference"];records=[]
 for ci,c in enumerate(cases):
  name="_".join(c[k] for k in ["domain","subject","montage"]);raw=np.load(c["x"],mmap_mode="r")
  for mode in ["raw","full"]:
   dest=P/f"{name}_{mode}.npz";ap=P/f"{name}_{mode}_audit.json"
   if dest.exists() and ap.exists():
    rec=json.loads(ap.read_text());assert sha(dest)==rec["prediction_sha256"];records.append(rec);continue
   lp=legacy(c,mode);base=np.load(lp)
   if mode=="full":
    scalar=legacy(c,"guarded_median");gp=scalar.with_name(scalar.stem+"_normalizer.npz");gains=np.load(gp)["gains"];xx,aux,stats=transform(raw,gains,reference,"full");assert np.array_equal(xx[:65],transform(raw[:65],gains[:65],reference,"full")[0]);assert stats["max_relative_energy_error_beforeclip"]<1e-12
   else:xx=raw
   x=torch.from_numpy(np.concatenate([prep(np.array(xx[a:a+128]),[.1,45]) for a in range(0,len(xx),128)]));result={};checks=[]
   for seed in [17,29,43]:
    if fallback[seed]:p=base[f"cbramod_seed{seed}"].copy();single_error=0.
    else:
     m=models[seed]
     with torch.inference_mode():
      p=torch.cat([m(b.cuda()).softmax(1).cpu() for b in x.split(128)]).numpy();single=m(x[:1].cuda()).softmax(1).cpu().numpy()
     single_error=float(abs(single-p[:1]).max());assert single_error<2e-5;assert all(torch.equal(v.cpu(),states[seed][k]) for k,v in m.state_dict().items())
    olderr=None
    if mode=="raw" and c["domain"]!="source":
     old=np.load(ROOT/("phase95" if seed==17 else "phase97")/"predictions"/f"{name}.npz");key="amplitude_augmented_selected" if seed==17 else f"amplitude_augmented_selected_seed{seed}";ref=old[key];olderr=float(abs(p-ref).max());assert olderr<2e-5 and np.array_equal(p.argmax(1),ref.argmax(1))
    result[f"amplitude_augmented_seed{seed}"]=p;checks.append(dict(seed=seed,fallback_exact=fallback[seed],batch_one_probability_error=single_error,old_raw_probability_error=olderr))
   np.savez_compressed(dest,**result);rec=dict(case=c,mode=mode,windows=len(raw),prediction_sha256=sha(dest),baseline_path=str(lp),baseline_sha256=sha(lp),checks=checks,input_prefix_exact=True,model_states_exact=True);ap.write_text(json.dumps(rec,indent=2));records.append(rec)
  print("FACTORIAL",ci+1,len(cases),name,flush=True)
 assert len(records)==184
 (O/"audit.json").write_text(json.dumps(dict(status="PASS",case_input_conditions=len(records),source_selected_models_fixed=True,repeat_models_exact_original_parents=True,seed29_fallback_reused_bitwise=True,no_new_training=True,classification_labels_not_read=True,records=records),indent=2));print("COMPLETE",flush=True)
if __name__=="__main__":
 with sdpa_kernel(SDPBackend.MATH):run()
