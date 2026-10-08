from common import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from phase77_predict import context
import hashlib,datetime,time,copy
O=ROOT/"phase106";O.mkdir(exist_ok=True)
train=[0,1,2,10,11,20,21,30,31,32];val=[5,15,25,35];variants=["relative","relative_repeat11","relative_context11","full_context11"];seeds=[17,29,43]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
pr=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),purpose="Source-only fixed static relative-spectral plus causal-history controls. No adaptation or hyperparametergrid.",train=train,validation=val,variants=variants,seeds=seeds,features="relative=features24:48 and50:52 (26dim). Relativecontext=11past/current windows286dim. Repeat11 repeats current26features11times286dim as samecapacitycontrol. Fullcontext=52featuresx11=572dim.",architecture="Historical SpectralNet64/32 BN/ELU/dropout.25; source-onlymean/std floor.1; frozenBN for evaluation.",recipe="Sameoriginalphase4:30epochs,AdamWlr.001wd.001,batch128,sqrtinverseclassweights normalizedmean1; allsource labels, mean4sourceperson rawMacroF1 selects earliestbestepoch1..30.",rng="DedicatedCPU torchGenerator(seed) for permutation, identical withinseed acrossvariants. Trainingdropout seed100000+seed reset afterconstructor. Same-dimension relativehistory/repeat models share exactinitial parameters, apartfrom sourcefeature scalingstatistics.",scope="Allsource and existingtargets repeatedlyobserved. Sourceonly fitting/checkpointselection, no newblindclaim. Parametercount differs for26/286/572 inputs; historycontrast286vs286 controls thatdifference.",cpu_ids=list(range(8,24)),gpu=True,source_features_hashes={f"s{s:02d}":sha(ROOT/("data/processed" if s<20 else "phase2/processed")/f"s{s:02d}_meta.npz") for s in train+val})
if (O/"protocol.json").exists():
 old=json.loads((O/"protocol.json").read_text());pr["created_utc"]=old["created_utc"];assert old==pr
else:(O/"protocol.json").write_text(json.dumps(pr,indent=2));(O/"protocol.sha256").write_text(sha(O/"protocol.json"))
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
def transform(ff,v):
 if v=="full_context11":return context(ff)
 rel=np.concatenate([ff[:,24:48],ff[:,50:52]],axis=1)
 return rel if v=="relative" else np.tile(rel,(1,11)) if v=="relative_repeat11" else context(rel)
def subject(s,v):
 p=ROOT/("data/processed" if s<20 else "phase2/processed")/f"s{s:02d}_meta.npz";z=np.load(p);ff=z["features"];x=transform(ff,v);assert np.array_equal(x[:65],transform(ff[:65],v));return x,z["y"]
@torch.inference_mode()
def probabilities(m,x):
 m.eval();return np.concatenate([m(x[i:i+1024]).softmax(1).cpu().numpy() for i in range(0,len(x),1024)])
selections=[]
for seed in seeds:
 for variant in variants:
  out=O/f"{variant}_seed{seed}";out.mkdir(exist_ok=True)
  if (out/"complete.json").exists():selections.append(json.loads((out/"complete.json").read_text()));continue
  seed_all(seed);xy=[subject(s,variant) for s in train];xx=np.concatenate([x[y>=0] for x,y in xy]);yy=np.concatenate([y[y>=0] for x,y in xy]);x=torch.from_numpy(xx);y=torch.from_numpy(yy);m=SpectralNet(x.shape[1]);initial_parameters=hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in m.parameters())).hexdigest();m.mean.copy_(x.mean(0));m.scale.copy_(x.std(0).clamp(min=.1));m.cuda();x=x.cuda();y=y.cuda();counts=torch.bincount(y,minlength=5).float();weights=(counts.sum()/counts.clamp(min=1)).sqrt();weights/=weights.mean();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001);g=torch.Generator().manual_seed(seed);torch.manual_seed(100000+seed);torch.cuda.manual_seed_all(100000+seed)
  validation={s:(torch.from_numpy(subject(s,variant)[0]).cuda(),subject(s,variant)[1]) for s in val};best=-1.;history=[];orders=[];start=0
  if (out/"resume.pt").exists():
   ck=torch.load(out/"resume.pt",map_location="cuda",weights_only=False);m.load_state_dict(ck["state"]);opt.load_state_dict(ck["opt"]);g.set_state(ck["order_rng"].cpu());torch.set_rng_state(ck["cpu_rng"].cpu());torch.cuda.set_rng_state(ck["cuda_rng"].cpu());best=ck["best"];history=ck["history"];orders=ck["orders"];start=ck["epoch"]
  for epoch in range(start,30):
   m.train();perm=torch.randperm(len(y),generator=g);orders.append(hashlib.sha256(perm.numpy().tobytes()).hexdigest());total=0.;begin=time.perf_counter()
   for idx in perm.split(128):
    assert len(idx)>=2;idx=idx.cuda();opt.zero_grad();loss=nn.functional.cross_entropy(m(x[idx]),y[idx],weight=weights);loss.backward();opt.step();total+=float(loss.detach())*len(idx)
   scores={s:metrics(vy,probabilities(m,vx))["macro_f1"] for s,(vx,vy) in validation.items()};score=float(np.mean(list(scores.values())));history.append(dict(epoch=epoch+1,loss=total/len(y),val_f1=score,val=scores,seconds=time.perf_counter()-begin))
   state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
   if score>best:
    best=score;torch.save(dict(state=state,dim=xx.shape[1],variant=variant,seed=seed,epoch=epoch+1,val_f1=score),out/"model.pt")
   torch.save(dict(state=state,opt=opt.state_dict(),order_rng=g.get_state(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),best=best,history=history,orders=orders,epoch=epoch+1),out/"resume.pt")
  ck=torch.load(out/"model.pt",map_location="cpu",weights_only=False);m.load_state_dict(ck["state"]);replay={s:metrics(vy,probabilities(m,vx))["macro_f1"] for s,(vx,vy) in validation.items()};assert abs(np.mean(list(replay.values()))-best)<1e-12
  for s,(vx,vy) in validation.items():np.savez_compressed(out/f"source_s{s:02d}.npz",p=probabilities(m,vx))
  rec=dict(seed=seed,variant=variant,dim=xx.shape[1],parameters=sum(p.numel() for p in m.parameters()),initial_parameters_sha256=initial_parameters,train_scored_windows=len(y),label_presentations=len(y)*30,selected_epoch=ck["epoch"],selected_val_f1=best,source_replay_f1=replay,checkpoint_sha256=sha(out/"model.pt"),order_hashes=orders,history=history);(out/"complete.json").write_text(json.dumps(rec,indent=2));selections.append(rec);print("STATIC SOURCE",seed,variant,ck["epoch"],best,flush=True)
for seed in seeds:
 records=[r for r in selections if r["seed"]==seed];assert len(records)==4 and all(r["order_hashes"]==records[0]["order_hashes"] for r in records)
 a=next(r for r in records if r["variant"]=="relative_repeat11");b=next(r for r in records if r["variant"]=="relative_context11");assert a["initial_parameters_sha256"]==b["initial_parameters_sha256"] and a["parameters"]==b["parameters"]
(O/"selection.json").write_text(json.dumps(selections,indent=2));(O/"audit.json").write_text(json.dumps(dict(status="PASS",models=12,source_only=True,orders_identical_within_seed=True,label_presentations_equal=True,relative_history_repeat_initial_parameters_exact=True,source_selected_checkpoint_replay_exact=True,causal_feature_prefix_exact=True),indent=2));print("COMPLETE",flush=True)
