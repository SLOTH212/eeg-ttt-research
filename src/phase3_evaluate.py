from common import *
from phase2_temporal import temporal,ema,score
from phase2_head import stream as head_stream
import itertools,time
OUT=ROOT/"phase3";CFG3=json.loads((OUT/"protocol.json").read_text())
def subject(s,kind):
 root=ROOT/("data/processed" if s<20 else "phase2/processed");d=np.load(root/f"s{s:02d}_meta.npz");return (d["features"] if kind=="spectral" else np.load(root/f"s{s:02d}_x.npy",mmap_mode="r")),d["y"]
for kind,fold,seed in itertools.product(["spectral","eegnet"],["young","middle","mixed"],[17,29,43]):
 name=f"{kind}_{fold}_{seed}";out=OUT/"results"/name;dest=out/"evaluation.json"
 if dest.exists():continue
 while not (out/"training_complete.json").exists():time.sleep(5)
 m,ck=load_model(out/"model.pt");split=CFG3["folds"][fold];data={};counts=np.ones((5,5));freq=np.ones(5)
 for s in split["base_train"]:
  _,y=subject(s,kind);freq+=np.bincount(y[y>=0],minlength=5);k=(y[:-1]>=0)&(y[1:]>=0);counts+=np.bincount(y[:-1][k]*5+y[1:][k],minlength=25).reshape(5,5)
 A=counts/counts.sum(1,keepdims=True);prior=freq/freq.sum()
 for s in split["base_val"]+split["evaluation"]:
  x,y=subject(s,kind);zs=[]
  with torch.no_grad():
   for a in range(0,len(x),128):zs.append(m.features(torch.tensor(np.array(x[a:a+128]),dtype=torch.float32)).numpy())
   z=np.concatenate(zs);p=m.head(torch.tensor(z)).softmax(1).numpy()
  data[s]=dict(z=z,p=p,y=y);np.savez_compressed(out/f"cache_s{s:02d}.npz",z=z,p=p,y=y)
 grid=[]
 for strength,temp,corr in itertools.product([0.,.25,.5,1.],[.75,1.,1.5],[0.,.5,1.]):
  opts=dict(strength=strength,temperature=temp,correction=corr);val=np.mean([score(data[s]["y"],temporal(data[s]["p"],A,prior,**opts)) for s in split["base_val"]]);grid.append(dict(**opts,val_f1=float(val)))
 best=max(grid,key=lambda x:x["val_f1"]);topts={k:v for k,v in best.items() if k!="val_f1"};hselect={};hgrids={};W=m.head.weight.detach().numpy();b=m.head.bias.detach().numpy()
 for objective in ["entropy","temporal","pseudo"]:
  hg=[]
  for lr,anchor in itertools.product([0.,.0001,.001,.01],[0.,.1,1.]):
   if lr==0 and anchor!=0:continue
   opts=dict(objective=objective,lr=lr,anchor=anchor);val=np.mean([score(data[s]["y"],head_stream(data[s]["z"],data[s]["p"],W,b,**opts)[0]) for s in split["base_val"]]);hg.append(dict(**opts,val_f1=float(val)))
  hselect[objective]=max(hg,key=lambda x:x["val_f1"]);hgrids[objective]=hg
 results=[]
 for s in split["evaluation"]:
  d=data[s];y=d["y"];p=d["p"];preds=dict(frozen=p,ema_08=ema(p,.8),ema_05=ema(p,.5),source_selected_filter=temporal(p,A,prior,**topts))
  for objective,sel in hselect.items():preds["head_"+objective],_=head_stream(d["z"],p,W,b,**{k:v for k,v in sel.items() if k!="val_f1"})
  for method,pr in preds.items():results.append(dict(subject=s,cohort="young" if s<20 else "middle",method=method,**metrics(y,pr)))
  np.savez_compressed(out/f"predictions_s{s:02d}.npz",y=y,**preds)
 dest.write_text(json.dumps(dict(name=name,results=results,temporal_selection=best,temporal_grid=grid,head_selection=hselect,head_grids=hgrids,development_only=True,affinity=sorted(os.sched_getaffinity(0))),indent=2));print("EVALUATED",name,flush=True)
