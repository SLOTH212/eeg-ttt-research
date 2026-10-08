from common import *
from controllers import fit_gate
import pickle,hashlib,datetime,itertools
OUT=ROOT/"phase6";assert not (OUT/"evaluation_lock.json").exists();(OUT/"controllers").mkdir(exist_ok=True)
models={};ginfo={};hashes={};cfg3=json.loads((ROOT/"phase3/protocol.json").read_text())
for variant,source,seed in itertools.product(["full52","wide","repeat11","context5","context11"],["young","middle","mixed"],[17,29,43]):
 name=f"{variant}_{source}_{seed}";phase="phase3" if variant=="full52" else ("phase4_controls" if variant in ["wide","repeat11"] else "phase4");dirname=f"spectral_{source}_{seed}" if variant=="full52" else name;f=ROOT/phase/"results"/dirname/"model.pt";models[name]=dict(path=str(f.relative_to(ROOT)),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),variant=variant,source=source,seed=seed)
for source,seed in itertools.product(["young","mixed"],[17,29,43]):
 name=f"context11_{source}_{seed}";old=ROOT/"phase5/results"/name;raw=old/"controller_rows.json";rows=json.loads(raw.read_text());gates={"identity_"+typ:fit_gate(rows,typ) for typ in ["future","confidence","current"]};hashes[str(raw.relative_to(ROOT))]=hashlib.sha256(raw.read_bytes()).hexdigest();sel=old/"selection.json";selection=json.loads(sel.read_text());hashes[str(sel.relative_to(ROOT))]=hashlib.sha256(sel.read_bytes()).hexdigest();cache=ROOT/"phase6_source_cache/results"/name/"source_counterfactuals.npz";hashes[str(cache.relative_to(ROOT))]=hashlib.sha256(cache.read_bytes()).hexdigest();d=np.load(cache);freq=np.ones(5)
 for s in cfg3["folds"][source]["base_train"]:
  base=ROOT/("data/processed" if s<20 else "phase2/processed");f=base/f"s{s:02d}_meta.npz";y=np.load(f)["y"];freq+=np.bincount(y[y>=0],minlength=5);hashes[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
 prior=freq/freq.sum()
 for power in [.5,1.]:
  w=prior**(-power);w/=prior@w;balanced=[]
  for i in range(len(d["t"])):
   yy=d["future_y"][i,:4];k=yy>=0
   if not k.any():continue
   loss={key:-np.log(np.maximum(d[key][i,:4][k,yy[k]],1e-9)) for key in ["keep","write","reset"]};cy=int(d["current_y"][i]);cur={key:-np.log(max(float(d["current_"+key][i,cy]),1e-9)) if cy>=0 else None for key in ["keep","write","reset"]}
   balanced.append(dict(subject=int(d["subject"][i]),h=4,features=d["features"][i].tolist(),write_gain=float(((loss["keep"]-loss["write"])*w[yy[k]]).mean()),reset_gain=float(((loss["keep"]-loss["reset"])*w[yy[k]]).mean()),current_write_gain=(cur["keep"]-cur["write"])*w[cy] if cy>=0 else None,current_reset_gain=(cur["keep"]-cur["reset"])*w[cy] if cy>=0 else None))
  for typ in ["future","confidence","current"]:gates[f"balanced_{power}_{typ}"]=fit_gate(balanced,typ)
 payload=dict(gates=gates,lr=selection["best_positive"]["lr"],best_lr=selection["best"]["lr"],prior=prior.tolist());f=OUT/"controllers"/(name+".pkl")
 with open(f,"wb") as fh:pickle.dump(payload,fh)
 ginfo[name]=dict(path=str(f.relative_to(ROOT)),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),lr=payload["lr"],best_lr=payload["best_lr"],methods=list(gates))
for name in ["common.py","adaptation.py","controllers.py","phase2_temporal.py","phase6_eval.py"]:
 f=ROOT/"src"/name;hashes[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
lock=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cohort=list(range(40,59)),age_range=[66,74],models=models,controllers=ginfo,sha256=hashes,primary_comparison="Frozen context11 young-source versus parameter-matched current-only wide young-source; 3seeds averaged within each of19 new subjects",secondary_comparisons="Context versus full52/repeat11/context5; other source compositions; unweighted/weighted future controller versus same-weight current/confidence, weak EMA0.8, source-selected no-update/simple updates, periodic and reset",evaluation="Same-dataset fresh cohort; participant MacroF1 with fixed5classes; bootstrap paired differences; stage-transition and negative-transfer risks secondary; no target label tuning; no pooling seeds as independent participants",allowed_cpu_ids=list(range(8,24)),gpu_allowed=False)
f=OUT/"evaluation_lock.json";f.write_text(json.dumps(lock,indent=2));print("LOCKED",len(models),"models",len(ginfo),"controller sets",hashlib.sha256(f.read_bytes()).hexdigest())
