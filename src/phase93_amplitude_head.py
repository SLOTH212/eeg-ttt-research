from phase66_train import *
O=ROOT/"phase93";O.mkdir(exist_ok=True);(O/"cache").mkdir(exist_ok=True)
def feats(m,x):return (m.encoder.encoder(m.encoder.patch_embedding(x)).mean(2).flatten(1)-m.mean)/m.scale
def run():
 train=[0,1,2,10,11,20,21,30,31,32];val=[5,15,25,35];path=ROOT/"phase66/pretrained_0.0001/model.pt"
 pr=dict(purpose="Source-only static amplitude-augmented classifier head; stronger frozen-input control motivated by observed amplitude-transfer failures, not blind discovery.",train=train,validation=val,classifier_seed=17,checkpoint=str(path),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),gains=[.1,1.,10.],input="Multiply both source raw channels by same fixed gain, clip to+-20, then exact existing .1-45Hz CBraMod prep; independent current windows.",methods=["repeat_raw","amplitude_augmented"],trainable="Only400->5classifierhead,2005params; frozen encoder and normalization buffers.",optimizer="Warmstart existing head, AdamWlr1e-4wd.01,batch128,10epochs,clip1,dedicatedgenerator17; identical3views/window presentations and permutation across methods.",weights="Equal source-person inverse count times sqrt inverse global class count; normalized mean1.",selection="Across epoch0..10 choose largest source-val mean MacroF1 over fixed3gains, only if native mean no worse than initial by1pp and each source-valperson no worse thaninitial by2pp. Ties keep earlier. Same rule for both controls.",primary="Selected amplitude augmentation versus selected repeat-raw control on source robust score; native tradeoff fullyreported.",advance="Only if selected augmentedepoch>0 and robust source score exceeds repeat-raw; no guarantee of target benefit. Target people previously observed; any later comparison exploratory.",extra_unique_class_labels=0,validation_labels_reused_for_selection=True,external_observations_motivated_design=True,cpu_ids=list(range(8,24)),gpu=0)
 p=O/"protocol.json"
 if p.exists():assert json.loads(p.read_text())==pr
 else:p.write_text(json.dumps(pr,indent=2));(O/"protocol.sha256").write_text(hashlib.sha256(p.read_bytes()).hexdigest())
 m=Carrier("pretrained").cuda().eval();m.load_state_dict(torch.load(path,map_location="cpu",weights_only=False)["state"]);m.requires_grad_(False);state=snapshot(m);checks=[]
 for sid in train+val:
  dest=O/"cache"/f"s{sid:02d}.npz";ap=dest.with_suffix(".json")
  if dest.exists() and ap.exists():checks.append(json.loads(ap.read_text()));continue
  folder=ROOT/("data/processed" if sid<20 else "phase2/processed");raw=np.load(folder/f"s{sid:02d}_x.npy",mmap_mode="r");y=np.load(folder/f"s{sid:02d}_meta.npz")["y"];allz=[];clipped=[];errors=[];started=time.perf_counter()
  for gain in pr["gains"]:
   zparts=[];clipcount=0
   with torch.no_grad():
    for i in range(0,len(raw),128):
     scaled=np.array(raw[i:i+128])*gain;clipcount+=int((abs(scaled)>20).sum());scaled=np.clip(scaled,-20,20).astype(np.float32);x=torch.tensor(prep(scaled,[.1,45]),device="cuda");z=feats(m,x);zparts.append(z.cpu().numpy())
     if i==0:
      one=feats(m,x[:1]);err=float(abs(m.head(one).softmax(1)-m.head(z[:1]).softmax(1)).max());assert err<2e-5;errors.append(err)
   allz.append(np.concatenate(zparts));clipped.append(clipcount)
  z=np.stack(allz,axis=1);assert z.shape==(len(y),3,400) and np.isfinite(z).all();assert all(torch.equal(v.cpu(),state[n]) for n,v in m.state_dict().items());np.savez_compressed(dest,z=z,y=y)
  rec=dict(subject=sid,windows=len(y),scored_windows=int((y>=0).sum()),clipped_samples_by_gain=clipped,batch_one_probability_errors=errors,encoder_state_exact=True,seconds=time.perf_counter()-started,cache_sha256=hashlib.sha256(dest.read_bytes()).hexdigest());ap.write_text(json.dumps(rec,indent=2));checks.append(rec);print("FEATURE CACHE",sid,len(y),round(rec["seconds"],2),flush=True)
 (O/"cache_audit.json").write_text(json.dumps(dict(status="PASS",records=checks,all_encoder_states_exact=True),indent=2))
 arrays=[];ys=[];subjects=[]
 for sid in train:
  a=np.load(O/"cache"/f"s{sid:02d}.npz");good=a["y"]>=0;arrays.append(a["z"][good]);ys.append(a["y"][good]);subjects.extend([sid]*int(good.sum()))
 z=torch.tensor(np.concatenate(arrays),device="cuda");y=torch.tensor(np.concatenate(ys),device="cuda");subjects=np.array(subjects);counts=np.bincount(y.cpu().numpy(),minlength=5);cw=np.sqrt(len(y)/np.maximum(counts,1));nper={sid:int((subjects==sid).sum()) for sid in train};weight=np.array([1/nper[int(sid)] for sid in subjects])*cw[y.cpu().numpy()];weight=torch.tensor(weight/weight.mean(),dtype=torch.float32,device="cuda")
 validation={sid:np.load(O/"cache"/f"s{sid:02d}.npz") for sid in val}
 def evaluate(head):
  scores=[]
  with torch.no_grad():
   for sid,a in validation.items():
    zz=torch.tensor(a["z"].reshape(-1,400),device="cuda");p=torch.cat([head(b).softmax(1) for b in zz.split(512)]).cpu().numpy().reshape(len(a["y"]),3,5)
    for j,gain in enumerate(pr["gains"]):scores.append(dict(subject=sid,gain=gain,macro_f1=100*metrics(a["y"],p[:,j])["macro_f1"]))
  return scores
 base_head=nn.Linear(400,5).cuda();base_head.load_state_dict(m.head.state_dict());baseline=evaluate(base_head);native={v["subject"]:v["macro_f1"] for v in baseline if v["gain"]==1};base_native=float(np.mean(list(native.values())));base_robust=float(np.mean([v["macro_f1"] for v in baseline]));selection={};histories={}
 for method in pr["methods"]:
  out=O/method;out.mkdir(exist_ok=True);head=nn.Linear(400,5).cuda();head.load_state_dict(m.head.state_dict());opt=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=.01);rng=torch.Generator(device="cpu").manual_seed(17);best=base_robust;bestepoch=0;beststate={k:v.detach().cpu().clone() for k,v in head.state_dict().items()};history=[dict(epoch=0,robust_f1=base_robust,native_f1=base_native,eligible=True,person_scores=baseline)];orders_hash=[];start=time.perf_counter()
  for epoch in range(10):
   order=torch.randperm(len(y)*3,generator=rng);orders_hash.append(hashlib.sha256(order.numpy().tobytes()).hexdigest());losses=[]
   for idx in order.split(128):
    idx=idx.cuda();baseidx=idx//3;view=idx%3 if method=="amplitude_augmented" else torch.ones_like(baseidx);opt.zero_grad();loss=(nn.functional.cross_entropy(head(z[baseidx,view]),y[baseidx],reduction="none")*weight[baseidx]).mean();loss.backward();gn=nn.utils.clip_grad_norm_(head.parameters(),1.);assert torch.isfinite(gn);opt.step();losses.append(float(loss.detach()))
   scores=evaluate(head);nat=[v for v in scores if v["gain"]==1];nm=float(np.mean([v["macro_f1"] for v in nat]));robust=float(np.mean([v["macro_f1"] for v in scores]));eligible=nm>=base_native-1 and all(v["macro_f1"]>=native[v["subject"]]-2 for v in nat);row=dict(epoch=epoch+1,loss=float(np.mean(losses)),robust_f1=robust,native_f1=nm,eligible=eligible,person_scores=scores);history.append(row)
   if eligible and robust>best:best=robust;bestepoch=epoch+1;beststate={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
   print("HEAD AUG",method,epoch+1,round(robust,4),round(nm,4),eligible,flush=True)
  torch.save(dict(head=beststate,epoch=bestepoch,robust_f1=best),out/"head.pt");selection[method]=dict(epoch=bestepoch,robust_f1=best,selected_metrics=history[bestepoch],checkpoint_sha256=hashlib.sha256((out/"head.pt").read_bytes()).hexdigest());histories[method]=dict(history=history,permutation_hashes=orders_hash,seconds=time.perf_counter()-start,training_source_windows=len(y),label_presentations=len(y)*3*10)
  (out/"history.json").write_text(json.dumps(histories[method],indent=2))
 assert histories["repeat_raw"]["permutation_hashes"]==histories["amplitude_augmented"]["permutation_hashes"];assert all(torch.equal(v.cpu(),state[n]) for n,v in m.state_dict().items())
 advance=selection["amplitude_augmented"]["epoch"]>0 and selection["amplitude_augmented"]["robust_f1"]>selection["repeat_raw"]["robust_f1"]
 (O/"selection.json").write_text(json.dumps(dict(selection=selection,advance=advance,source_only=True,no_new_unique_training_labels=True),indent=2));(O/"complete.json").write_text(json.dumps(dict(status="PASS",frozen_encoder_exact=True,same_permutations_and_presentations=True,advance=advance)));print("SELECTION",selection,"ADVANCE",advance,flush=True)
if __name__=="__main__":run()
