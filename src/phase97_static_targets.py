from phase66_train import *
O=ROOT/"phase97";O.mkdir(exist_ok=True);(O/"predictions").mkdir(exist_ok=True)
def legacy(c,mode):
 sid=c["subject"];mo=c["montage"]
 return ROOT/"phase81/predictions"/f"{sid}_{mode}.npz" if c["domain"]=="sleepedf" else ROOT/"phase79/predictions"/f"external_{sid}_{mo}_{mode}.npz" if c["domain"]=="healthy" else ROOT/"phase83/predictions"/f"{sid}_{mo}_{mode}.npz"
def run():
 for seed in [29,43]:assert json.loads((ROOT/"phase96"/f"seed{seed}"/"audit.json").read_text())["status"]=="PASS"
 cases=json.loads((ROOT/"phase95/protocol.json").read_text())["cases"];choices=json.loads((ROOT/"phase71/selection_lock.json").read_text())["selections"];specs=[]
 for seed in [29,43]:
  selection=json.loads((ROOT/"phase96"/f"seed{seed}"/"selection.json").read_text())
  for kind in ["repeat_raw","amplitude_augmented"]:
   for version in ["selected","final5"]:
    path=ROOT/"phase96"/f"seed{seed}"/kind/("model.pt" if version=="selected" else "resume.pt");specs.append(dict(seed=seed,method=kind+"_"+version,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),selected_epoch=selection["selection"][kind]["epoch"] if version=="selected" else 5))
 pr=dict(purpose="Complete fixed source-selected static recipe evaluation across classifierseeds17/29/43; seed17phase95reused, seed29sourceguardfallbackretained.",new_seeds=[29,43],cases=cases,models=specs,primary="Same asphase95, first average metrics withinperson across3seeds then participantpairedcomparison. No probability ensemble or seedpseudoreplication.",ema=[1.,.8,.5],methods=["repeat_raw_selected","amplitude_augmented_selected","repeat_raw_final5","amplitude_augmented_final5","initial_guarded_median","initial_floor"],source_choices_fixed_before_targetscoring=True,all_targets_previously_observed=True,no_online_updates=True,seed17_metrics_sha256=hashlib.sha256((ROOT/"phase95/participants.csv").read_bytes()).hexdigest())
 (O/"protocol.json").write_text(json.dumps(pr,indent=2));(O/"protocol.sha256").write_text(hashlib.sha256((O/"protocol.json").read_bytes()).hexdigest());models={};states={};reuse={}
 for spec in specs:
  m=Carrier("pretrained").cuda().eval().requires_grad_(False);m.load_state_dict(torch.load(spec["path"],map_location="cpu",weights_only=False)["state"]);key=(spec["seed"],spec["method"]);models[key]=m;states[key]=snapshot(m);basepath=Path(choices[str(spec["seed"])]["selected"]["folder"])/"model.pt";base=torch.load(basepath,map_location="cpu",weights_only=False)["state"];reuse[key]=all(torch.equal(v,base[n]) for n,v in states[key].items())
 audits=[]
 for ci,c in enumerate(cases):
  name="_".join(c[k] for k in ["domain","subject","montage"]);ap=O/"predictions"/(name+"_audit.json")
  if ap.exists():audits.append(json.loads(ap.read_text()));continue
  raw=np.load(c["x"],mmap_mode="r");x=torch.from_numpy(prep(np.array(raw),[.1,45]));preds={};checks=[];old=np.load(legacy(c,"raw"))
  for (seed,method),m in models.items():
   if reuse[seed,method]:
    p=old[f"cbramod_seed{seed}"];verify=probabilities(m,x[:32]);err=float(abs(verify-p[:32]).max());assert err<2e-5 and np.array_equal(verify.argmax(1),p[:32].argmax(1))
   else:p=probabilities(m,x);err=None
   assert p.shape==(len(raw),5) and np.isfinite(p).all() and abs(p.sum(1)-1).max()<2e-6
   with torch.no_grad():one=m(x[:1].cuda()).softmax(1).cpu().numpy();oneerr=float(abs(one-p[:1]).max());assert oneerr<2e-5
   assert all(torch.equal(v.cpu(),states[seed,method][n]) for n,v in m.state_dict().items());preds[f"{method}_seed{seed}"]=p;checks.append(dict(seed=seed,method=method,windows=len(raw),initial_state_reused=reuse[seed,method],reuse_first32_error=err,batch_one_error=oneerr,state_exact=True))
  for seed in [29,43]:
   for mode in ["guarded_median","floor"]:preds[f"initial_{mode}_seed{seed}"]=np.load(legacy(c,mode))[f"cbramod_seed{seed}"]
  np.savez_compressed(O/"predictions"/(name+".npz"),**preds);rec=dict(case=c,checks=checks,labels_notused_inprediction=True,previous_label_exposure_disclosed=True);ap.write_text(json.dumps(rec,indent=2));audits.append(rec);print("MULTISEED TARGET",ci+1,len(cases),name,flush=True)
 (O/"prediction_audit.json").write_text(json.dumps(dict(status="PASS",records=audits,allstatesexact=True,sourceguardfallback_kept=True),indent=2));(O/"complete.json").write_text(json.dumps(dict(cases=88,seeds=2,new_method_streams=3168)))
if __name__=="__main__":run()
