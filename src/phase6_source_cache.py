from common import *
from adaptation import stream
from controllers import fit_gate
import itertools,time
OUT=ROOT/"phase6_source_cache";OUT.mkdir(exist_ok=True);(OUT/"results").mkdir(exist_ok=True)
protocol=json.loads((ROOT/"phase5/protocol.json").read_text());protocol.update(purpose="Exploratory loss-alignment test: controller learns source-frequency-weighted future/current CE gains; target labels never enter policy",powers=[.5,1.],weight_rule="w_c=source_prior_c^(-power), normalized so sum(source_prior*w)=1; source training labels only")
(OUT/"protocol.json").write_text(json.dumps(protocol,indent=2));cfg3=json.loads((ROOT/"phase3/protocol.json").read_text())
def subj(s):
 base=ROOT/("data/processed" if s<20 else "phase2/processed");d=np.load(base/f"s{s:02d}_meta.npz");hf=d["features"];pad=np.pad(hf,((10,0),(0,0)),mode="edge");return np.concatenate([pad[j:j+len(hf)] for j in range(11)],axis=1),d["y"],hf
for source,seed in itertools.product(protocol["sources"],protocol["seeds"]):
 name=f"context11_{source}_{seed}";out=OUT/"results"/name;out.mkdir(exist_ok=True)
 if (out/"evaluation.json").exists():continue
 selection=ROOT/"phase5/results"/name/"selection.json"
 while not selection.exists():time.sleep(5)
 lr=json.loads(selection.read_text())["best_positive"]["lr"];ck=torch.load(ROOT/"phase4/results"/name/"model.pt",map_location="cpu",weights_only=False);m=SpectralNet(ck["dim"]);m.load_state_dict(ck["state"]);m.eval();freq=np.ones(5)
 for s in cfg3["folds"][source]["base_train"]:
  _,y,_=subj(s);freq+=np.bincount(y[y>=0],minlength=5)
 prior=freq/freq.sum();cache=out/"source_counterfactuals.npz"
 if not cache.exists():
  collect={k:[] for k in ["subject","t","trajectory","features","keep","write","reset","current_keep","current_write","current_reset","future_y","current_y"]}
  for s in protocol["controller_train"]:
   x,y,hf=subj(s)
   for mode in ["continuous","periodic"]:
    _,ds,_=stream(m,x,mode,lr,history_features=hf,collect=True,subject=s)
    for d in ds:
     for key in ["t","features","keep","write","reset","current_keep","current_write","current_reset"]:collect[key].append(d[key])
     collect["subject"].append(s);collect["trajectory"].append(mode);collect["future_y"].append(y[d["t"]+1:d["t"]+11]);collect["current_y"].append(y[d["t"]])
  np.savez_compressed(cache,**{k:np.array(v) for k,v in collect.items()});print("COUNTERFACTUAL CACHE",name,flush=True)
 print("PRECOMPUTED",name,flush=True)
