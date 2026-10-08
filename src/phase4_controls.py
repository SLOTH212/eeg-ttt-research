from common import *
from phase2_temporal import ema
import itertools,time,copy,hashlib
OUT=ROOT/"phase4_controls";OUT.mkdir(exist_ok=True);(OUT/"results").mkdir(exist_ok=True)
CFG4=json.loads((ROOT/"phase3/protocol.json").read_text());CFG4.update(purpose="Exploratory capacity controls for context11: current-only wide network matched to~39k parameters, and current window repeated11times using identical context11 architecture. Same splits and training asphase4; no historical information in either control.",variants=["wide","repeat11"])
(OUT/"protocol.json").write_text(json.dumps(CFG4,indent=2))
def transform(x,variant):
 if variant=="wide":return x
 if variant=="repeat11":return np.tile(x,(1,11))
 k=int(variant.removeprefix("context"));padded=np.pad(x,((k-1,0),(0,0)),mode="edge")
 return np.concatenate([padded[j:j+len(x)] for j in range(k)],axis=1)
def subj(s,variant):
 base=ROOT/("data/processed" if s<20 else "phase2/processed");d=np.load(base/f"s{s:02d}_meta.npz");x=d["features"];z=transform(x,variant);assert np.array_equal(z[:31],transform(x[:31],variant));return z,d["y"]
for variant,fold,seed in itertools.product(CFG4["variants"],["young","middle","mixed"],[17,29,43]):
 name=f"{variant}_{fold}_{seed}";out=OUT/"results"/name;out.mkdir(exist_ok=True)
 if (out/"evaluation.json").exists():continue
 seed_all(seed);split=CFG4["folds"][fold];xs=[];ys=[]
 for sid in split["base_train"]:
  x,y=subj(sid,variant);xs.append(x[y>=0]);ys.append(y[y>=0])
 x=torch.from_numpy(np.concatenate(xs));y=torch.from_numpy(np.concatenate(ys));del xs,ys
 m=SpectralNet(x.shape[1])
 if variant=="wide":m.body=nn.Sequential(nn.Linear(52,448),nn.BatchNorm1d(448),nn.ELU(),nn.Dropout(.25),nn.Linear(448,32),nn.BatchNorm1d(32),nn.ELU())
 m.mean.copy_(x.mean(0));m.scale.copy_(x.std(0).clamp(min=.1));counts=torch.bincount(y,minlength=5).float();weights=(counts.sum()/counts.clamp(min=1)).sqrt();weights/=weights.mean();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001)
 val={s:subj(s,variant) for s in split["base_val"]};best=-1.;history=[];start_epoch=0
 if (out/"resume.pt").exists():
  ck=torch.load(out/"resume.pt",map_location="cpu",weights_only=False);m.load_state_dict(ck["state"]);opt.load_state_dict(ck["opt"]);torch.set_rng_state(ck["rng"]);best=ck["best"];history=ck["history"];start_epoch=ck["epoch"]
 for epoch in range(start_epoch,30):
  m.train();perm=torch.randperm(len(y));total=0.;t=time.perf_counter()
  for idx in perm.split(128):
   if len(idx)<2:continue
   opt.zero_grad();loss=nn.functional.cross_entropy(m(x[idx]),y[idx],weight=weights);loss.backward();opt.step();total+=float(loss.detach())*len(idx)
  scores={s:metrics(vy,predict(m,vx))["macro_f1"] for s,(vx,vy) in val.items()};score=float(np.mean(list(scores.values())));history.append(dict(epoch=epoch+1,loss=total/len(y),val_f1=score,val=scores,seconds=time.perf_counter()-t))
  if score>best:
   best=score;torch.save(dict(state=copy.deepcopy(m.state_dict()),dim=x.shape[1],variant=variant,fold=fold,seed=seed,epoch=epoch+1,val_f1=best),out/"model.pt")
  torch.save(dict(state=m.state_dict(),opt=opt.state_dict(),rng=torch.get_rng_state(),best=best,history=history,epoch=epoch+1),out/"resume.tmp");(out/"resume.tmp").replace(out/"resume.pt")
 ck=torch.load(out/"model.pt",map_location="cpu",weights_only=False);m.load_state_dict(ck["state"]);results=[]
 for s in split["evaluation"]:
  vx,vy=subj(s,variant);pr=predict(m,vx);preds=dict(frozen=pr,ema_08=ema(pr,.8))
  for method,pp in preds.items():results.append(dict(subject=s,cohort="young" if s<20 else "middle",method=method,**metrics(vy,pp)))
  np.savez_compressed(out/f"predictions_s{s:02d}.npz",y=vy,**preds)
 (out/"evaluation.json").write_text(json.dumps(dict(name=name,variant=variant,source=fold,seed=seed,results=results,history=history,selected_epoch=ck["epoch"],parameters=sum(p.numel() for p in m.parameters()),prefix_audit=True,affinity=sorted(os.sched_getaffinity(0))),indent=2));print("FEATURE COMPLETE",name,ck["epoch"],flush=True)
