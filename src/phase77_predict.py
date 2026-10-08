from phase66_train import Carrier,prep,ROOT,snapshot,probabilities
from phase21_full_encoder import FullCarrier,preprocess
from common import SpectralNet
import torch,numpy as np,json,hashlib
OUT77=ROOT/"phase77"
def context(x):
 padded=np.pad(x,((10,0),(0,0)),mode="edge");return np.concatenate([padded[j:j+len(x)] for j in range(11)],axis=1)
def run():
 pr=json.loads((OUT77/"protocol.json").read_text());assert hashlib.sha256((OUT77/"protocol.json").read_bytes()).hexdigest()==(OUT77/"protocol.sha256").read_text();models={};states={};checks=[]
 for spec in pr["models"]:
  path=ROOT/spec["path"];assert hashlib.sha256(path.read_bytes()).hexdigest()==spec["sha256"];ck=torch.load(path,map_location="cpu",weights_only=False)
  m=Carrier("pretrained") if spec["model"]=="cbramod" else FullCarrier("mixed","pretrained") if spec["model"]=="labram" else SpectralNet(ck["dim"]);m.load_state_dict(ck["state"]);m.cuda().eval().requires_grad_(False);key=f'{spec["model"]}_seed{spec["seed"]}';models[key]=m;states[key]=snapshot(m)
 out=OUT77/"predictions";out.mkdir(exist_ok=True)
 for sid in pr["participants"]:
  for montage in pr["montages"]:
   dest=out/(sid+"_"+montage+".npz");raw=np.load(ROOT/"phase76/processed"/(sid+"_"+montage+"_x.npy"));features=np.load(ROOT/"phase76/processed"/(sid+"_"+montage+"_features.npy"));ct=context(features);assert np.array_equal(ct[:33],context(features[:33]))
   xcb=torch.from_numpy(prep(raw,[.1,45]));xlb=preprocess(raw).reshape(-1,2,2,15,200);xc=torch.from_numpy(ct);xs={"cbramod":xcb,"labram":xlb,"context11":xc};preds={}
   for key,m in models.items():
    kind=key.rsplit("_seed",1)[0];x=xs[kind];p=probabilities(m,x);assert p.shape==(len(raw),5) and np.isfinite(p).all();preds[key]=p
    with torch.no_grad():
     t=len(x)//2;single=m(x[t:t+1].cuda()).softmax(1).cpu().numpy()[0];err=float(abs(single-p[t]).max());assert err<2e-5 and single.argmax()==p[t].argmax()
     if kind=="context11":
      short=context(features[:33]);replay=m(torch.from_numpy(short).cuda()).softmax(1).cpu().numpy();prefixerr=float(abs(replay-p[:33]).max());assert prefixerr<2e-5 and np.array_equal(replay.argmax(1),p[:33].argmax(1))
     else:prefixerr=0.
    assert all(torch.equal(v.cpu(),states[key][n]) for n,v in m.state_dict().items());checks.append(dict(subject=sid,montage=montage,model=key,windows=len(raw),batch_one_error=err,context_prefix_error=prefixerr,state_unchanged=True))
   np.savez_compressed(dest,**preds);print("EXTERNAL STATIC PREDICTIONS",sid,montage,len(raw),flush=True)
 (OUT77/"prediction_audit.json").write_text(json.dumps(dict(status="PASS",cases=checks,source_checkpoints_verified=True,no_target_labels_read=True,models=9,person_montage_cases=20,all_model_states_unchanged=True),indent=2));(OUT77/"predictions_complete.json").write_text(json.dumps(dict(complete=True,model_person_montage_cases=180,no_target_updates=True)))
if __name__=="__main__":run()
