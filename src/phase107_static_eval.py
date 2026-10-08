from common import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from phase77_predict import context,snapshot
import hashlib,datetime
O=ROOT/"phase107";O.mkdir(exist_ok=True);(O/"predictions").mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((ROOT/"phase106/audit.json").read_text())["status"]=="PASS"
selections=json.loads((ROOT/"phase106/selection.json").read_text());cases=json.loads((ROOT/"phase85/protocol.json").read_text())["cases"]
modelspec=[dict(variant=r["variant"],seed=r["seed"],path=str(ROOT/"phase106"/f'{r["variant"]}_seed{r["seed"]}'/"model.pt"),sha256=r["checkpoint_sha256"]) for r in selections]
pr=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),purpose="Fixed previouslyobserved full-record evaluation of source-only relative-history static controls.",cases=cases,models=modelspec,source_protocol_sha256=sha(ROOT/"phase106/protocol.json"),input="Original raw cached52features, unchanged. No amplitude/shape adaptation.",ema=[1.,.8,.5],primary="SleepEDF30 native EMA.5: relative_context11 versus same-dimensional relative_repeat11; three seed metrics averaged withinperson.",secondary="Relativehistory versus current26featuremodel and newlytrained fullcontext11; source4,healthy10,patient19 separately, rightmontage sensitivity.",scope="Allpeople alreadyobserved, exploratory; phase101new20 not included because their transientwaveforms/features were not retained. No new blindclaim, no selection amongvariants or EMA on targets.")
if (O/"protocol.json").exists():
 old=json.loads((O/"protocol.json").read_text());pr["created_utc"]=old["created_utc"];assert pr==old
else:(O/"protocol.json").write_text(json.dumps(pr,indent=2));(O/"protocol.sha256").write_text(sha(O/"protocol.json"))
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def trans(ff,v):
 if v=="full_context11":return context(ff)
 rel=np.concatenate([ff[:,24:48],ff[:,50:52]],axis=1);return rel if v=="relative" else np.tile(rel,(1,11)) if v=="relative_repeat11" else context(rel)
models={};snaps={}
for sp in modelspec:
 assert sha(Path(sp["path"]))==sp["sha256"];ck=torch.load(sp["path"],map_location="cpu",weights_only=False);m=SpectralNet(ck["dim"]);m.load_state_dict(ck["state"]);m.cuda().eval().requires_grad_(False);key=f'{sp["variant"]}_seed{sp["seed"]}';models[key]=m;snaps[key]=snapshot(m)
@torch.inference_mode()
def pred(m,x):
 return np.concatenate([m(torch.from_numpy(x[i:i+1024]).cuda()).softmax(1).cpu().numpy() for i in range(0,len(x),1024)])
records=[]
for c in cases:
 name="_".join(c[k] for k in ["domain","subject","montage"]);ap=O/"predictions"/(name+"_audit.json")
 if ap.exists():records.append(json.loads(ap.read_text()));continue
 if c["labels"].endswith(".npz"):fp=Path(c["labels"]);ff=np.load(fp)["features"]
 else:fp=Path(c["x"].replace("_x.npy","_features.npy"));ff=np.load(fp)
 xs={v:trans(ff,v) for v in ["relative","relative_repeat11","relative_context11","full_context11"]};assert all(np.array_equal(x[:65],trans(ff[:65],v)) for v,x in xs.items());probs={};checks=[]
 for key,m in models.items():
  variant=key.rsplit("_seed",1)[0];x=xs[variant];p=pred(m,x);short=pred(m,x[:33]);err=float(abs(short-p[:33]).max());assert err<2e-5 and np.array_equal(short.argmax(1),p[:33].argmax(1));assert p.shape==(len(ff),5) and np.isfinite(p).all() and abs(p.sum(1)-1).max()<3e-6
  assert all(torch.equal(v.cpu(),snaps[key][k]) for k,v in m.state_dict().items());probs[key]=p;checks.append(dict(model=key,prefix_probability_error=err,prefix_argmax_exact=True,frozen_state_exact=True))
 np.savez_compressed(O/"predictions"/(name+".npz"),**probs);rec=dict(case=c,features_path=str(fp),features_sha256=sha(fp),checks=checks,causal_feature_prefix_exact=True);ap.write_text(json.dumps(rec,indent=2));records.append(rec);print("STATIC CASE",name,len(ff),flush=True)
(O/"audit.json").write_text(json.dumps(dict(status="PASS",cases=len(cases),models=12,records=records,target_labels_not_used_for_prediction=True),indent=2));(O/"complete.json").write_text(json.dumps(dict(cases=len(cases),models=12)));print("COMPLETE",flush=True)
