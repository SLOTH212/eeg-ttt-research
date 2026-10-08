from common import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import hashlib,time,datetime
from phase79_amplitude import normalize
O=ROOT/"phase85";O.mkdir(exist_ok=True);(O/"predictions").mkdir(exist_ok=True)
cases=[]
for sid in [5,15,25,35]+json.loads((ROOT/"phase81/protocol.json").read_text())["subjects"]:
 folder=ROOT/("data/processed" if sid<20 else "phase2/processed" if sid<40 else "phase6/processed")
 cases.append(dict(domain="source" if sid in [5,15,25,35] else "sleepedf",subject=f"s{sid:02d}",montage="native",x=str(folder/f"s{sid:02d}_x.npy"),labels=str(folder/f"s{sid:02d}_meta.npz")))
for domain,phase,ids in [("healthy","phase76",[f"sub-III{i:03d}" for i in range(1,11)]),("patient","phase83",json.loads((ROOT/"phase83/cohort_lock_after_input_audit.json").read_text())["included"])]:
 for sid in ids:
  for montage in ["left","right"]:
   lab=ROOT/("phase77" if domain=="healthy" else "phase83")/"labels"/(sid+"_y.npy")
   cases.append(dict(domain=domain,subject=sid,montage=montage,x=str(ROOT/phase/"processed"/f"{sid}_{montage}_x.npy"),labels=str(lab)))
specs=[]
for seed in [17,29,43]:
 p=ROOT/f"phase3/results/eegnet_mixed_{seed}/model.pt"
 ck=torch.load(p,map_location="cpu",weights_only=False)
 assert ck["config"]["folds"]["mixed"]["base_train"]==[0,1,2,10,11,20,21,30,31,32]
 assert ck["config"]["folds"]["mixed"]["base_val"]==[5,15,25,35]
 specs.append(dict(seed=seed,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),source_selected_epoch=ck["epoch"],source_val_f1=ck["val_f1"]))
pr=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),purpose="Existing frozen EEGNet-style lightweight external comparator; exploratory after target labels already exposed. No new blind claim.",architecture="Historical common.EEGNet, EEGNet-style 8,2 with adaptive average pool8; not exact official reproduction.",models=specs,cases=cases,modes=["raw","guarded_median","floor"],ema=[1.,.8,.5],normalizer_protocol_sha256=hashlib.sha256((ROOT/"phase79/protocol.json").read_bytes()).hexdigest(),selection="No training, tuning, method selection or target-based checkpoint selection. Source-selected historical checkpoints.",cpu_ids=list(range(8,24)),gpu=True)
if (O/"protocol.json").exists():
 old=json.loads((O/"protocol.json").read_text());pr["created_utc"]=old["created_utc"];assert pr==old
else:
 (O/"protocol.json").write_text(json.dumps(pr,indent=2));(O/"protocol.sha256").write_text(hashlib.sha256((O/"protocol.json").read_bytes()).hexdigest())
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
assert torch.cuda.is_available()
@torch.inference_mode()
def gpu_pred(m,x,batch=128):
 return np.concatenate([m(torch.as_tensor(np.array(x[a:a+batch]),dtype=torch.float32,device="cuda")).softmax(1).cpu().numpy() for a in range(0,len(x),batch)])
models=[];compat=[]
reference=np.load(cases[0]["x"],mmap_mode="r")[:32]
for sp in specs:
 m,ck=load_model(sp["path"]);pc=predict(m,reference,batch=32);m=m.cuda().eval();pg=gpu_pred(m,reference)
 err=float(abs(pc-pg).max());assert err<2e-5 and np.array_equal(pc.argmax(1),pg.argmax(1))
 compat.append(dict(seed=sp["seed"],cpu_gpu_max_probability_error=err,argmax_exact=True,parameters=sum(p.numel() for p in m.parameters())))
 models.append((sp["seed"],m,{k:v.detach().cpu().clone() for k,v in m.state_dict().items()}))
reference_rms=json.loads((ROOT/"phase79/protocol.json").read_text())["source_reference_rms"]
audit=[]
for ci,case in enumerate(cases):
 name="_".join(case[k] for k in ["domain","subject","montage"]);ap=O/"predictions"/(name+"_audit.json")
 if ap.exists():audit.append(json.loads(ap.read_text()));continue
 raw=np.load(case["x"],mmap_mode="r");rec=dict(case=case,checks=[]);start=time.perf_counter()
 for mode in pr["modes"]:
  if mode=="raw":x=raw
  else:
   x,gains,states,rms=normalize(raw,reference_rms,mode)
   aux=dict(gains=gains,log_rms_state=states,current_rms=rms)
   np.savez_compressed(O/"predictions"/(name+"_"+mode+"_normalizer.npz"),**aux)
  probs={}
  for seed,m,snap in models:
   p=gpu_pred(m,x);one=gpu_pred(m,x[:1],1);err=float(abs(one-p[:1]).max());assert err<2e-5
   assert p.shape==(len(raw),5) and np.isfinite(p).all() and abs(p.sum(1)-1).max()<2e-6
   assert all(torch.equal(v.cpu(),snap[k]) for k,v in m.state_dict().items())
   probs[f"eegnet_seed{seed}"]=p;rec["checks"].append(dict(mode=mode,seed=seed,windows=len(raw),batch_one_error=err,state_exact=True))
  np.savez_compressed(O/"predictions"/(name+"_"+mode+".npz"),**probs)
 rec["seconds"]=time.perf_counter()-start;ap.write_text(json.dumps(rec,indent=2));audit.append(rec);print("CASE",ci+1,len(cases),name,round(rec["seconds"],2),flush=True)
(O/"prediction_audit.json").write_text(json.dumps(dict(status="PASS",cpu_gpu=compat,records=audit,all_frozen=True,labels_not_used_for_prediction=True,previous_label_exposure_disclosed=True),indent=2))
(O/"complete.json").write_text(json.dumps(dict(cases=len(cases),trajectories=len(cases)*9)))
print("COMPLETE",len(cases),flush=True)
