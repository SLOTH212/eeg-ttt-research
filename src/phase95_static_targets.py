from phase66_train import *
O=ROOT/"phase95";O.mkdir(exist_ok=True);(O/"predictions").mkdir(exist_ok=True)
def legacy(case,mode):
 sid=case["subject"];montage=case["montage"]
 if case["domain"]=="sleepedf":p=ROOT/"phase81/predictions"/f"{sid}_{mode}.npz"
 elif case["domain"]=="healthy":p=ROOT/"phase79/predictions"/f"external_{sid}_{montage}_{mode}.npz"
 else:p=ROOT/"phase83/predictions"/f"{sid}_{montage}_{mode}.npz"
 return p
def run():
 assert json.loads((ROOT/"phase94/audit.json").read_text())["status"]=="PASS";sel=json.loads((ROOT/"phase94/selection.json").read_text());assert sel["advance"];cases=[c for c in json.loads((ROOT/"phase85/protocol.json").read_text())["cases"] if c["domain"]!="source"]
 specs=[]
 for kind in ["repeat_raw","amplitude_augmented"]:
  for version in ["selected","final5"]:
   p=ROOT/"phase94"/kind/("model.pt" if version=="selected" else "resume.pt");specs.append(dict(method=kind+"_"+version,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),epoch=sel["selection"][kind]["epoch"] if version=="selected" else 5))
 pr=dict(purpose="Exploratory previously-observed target evaluation of source-selected static amplitude augmentation; singleclassifierseed17, not new blind validation.",cases=cases,models=specs,ema=[1.,.8,.5],primary="amplitude_augmented_selected minus repeat_raw_selected, sameEMA.5; report SleepEDF30,healthy10,patient19 separately, externalleft primary.",secondary="Fixed5epoch amplitude_augmented_final5 vs repeat_raw_final5 matches actualtrainingupdates; diagnostic only, no target-based modelselection. Original fixedguard/floor controls reused.",source_selection=sel,selection_scope="No choices made from these target scores; all modelstates/checkpoints fixed before this evaluation.",normalizer_protocol_sha256=hashlib.sha256((ROOT/"phase79/protocol.json").read_bytes()).hexdigest(),no_test_time_parameter_updates=True,extra_unique_class_labels=0,interpretation="Raw selected control is epoch0: both branches had same attemptedbudget, but selectedcheckpoints have different effectiveepochs. Fixed5epoch secondary makes that distinction explicit.")
 (O/"protocol.json").write_text(json.dumps(pr,indent=2));(O/"protocol.sha256").write_text(hashlib.sha256((O/"protocol.json").read_bytes()).hexdigest());models={};states={}
 for spec in specs:
  ck=torch.load(spec["path"],map_location="cpu",weights_only=False);m=Carrier("pretrained").cuda().eval().requires_grad_(False);m.load_state_dict(ck["state"]);models[spec["method"]]=m;states[spec["method"]]=snapshot(m)
 initial=torch.load(ROOT/"phase66/pretrained_0.0001/model.pt",map_location="cpu",weights_only=False)["state"];assert all(torch.equal(v,initial[k]) for k,v in states["repeat_raw_selected"].items())
 audits=[]
 for i,case in enumerate(cases):
  name="_".join(case[k] for k in ["domain","subject","montage"]);ap=O/"predictions"/(name+"_audit.json")
  if ap.exists():audits.append(json.loads(ap.read_text()));continue
  raw=np.load(case["x"],mmap_mode="r");x=torch.from_numpy(prep(np.array(raw),[.1,45]));preds={};checks=[];oldraw=np.load(legacy(case,"raw"))["cbramod_seed17"]
  for method,m in models.items():
   if method=="repeat_raw_selected":
    verify=probabilities(m,x[:32]);err=float(abs(verify-oldraw[:32]).max());assert err<2e-5 and np.array_equal(verify.argmax(1),oldraw[:32].argmax(1));p=oldraw
   else:p=probabilities(m,x);err=None
   assert p.shape==(len(raw),5) and np.isfinite(p).all() and abs(p.sum(1)-1).max()<2e-6
   with torch.no_grad():one=m(x[:1].cuda()).softmax(1).cpu().numpy();oneerr=float(abs(p[:1]-one).max());assert oneerr<2e-5
   assert all(torch.equal(v.cpu(),states[method][k]) for k,v in m.state_dict().items());preds[method]=p;checks.append(dict(method=method,windows=len(raw),batch_one_error=oneerr,legacy_reuse_first32_error=err,model_state_exact=True))
  for mode in ["guarded_median","floor"]:preds["initial_"+mode]=np.load(legacy(case,mode))["cbramod_seed17"];assert preds["initial_"+mode].shape==(len(raw),5)
  np.savez_compressed(O/"predictions"/(name+".npz"),**preds);rec=dict(case=case,checks=checks,labels_not_used_by_prediction=True,previous_label_exposure_disclosed=True);ap.write_text(json.dumps(rec,indent=2));audits.append(rec);print("TARGET STATIC",i+1,len(cases),name,flush=True)
 (O/"prediction_audit.json").write_text(json.dumps(dict(status="PASS",records=audits,all_model_states_exact=True,initial_selected_raw_state_identical=True),indent=2));(O/"complete.json").write_text(json.dumps(dict(cases=len(cases),method_streams=len(cases)*6*3,no_online_updates=True)))
if __name__=="__main__":run()
