from pathlib import Path
import json,hashlib,datetime,itertools
r=Path(__file__).resolve().parents[1];out=r/"phase2";path=out/"evaluation_lock.json"
assert not path.exists(),"Lock already exists; never silently overwrite"
models={};hashes={}
for kind,fold,seed in itertools.product(["spectral","eegnet"],["A","B"],[17,29,43]):
 name=f"{kind}_{fold}_{seed}";parts={}
 for typ in ["temporal","head","prototype"]:
  f=out/"results"/f"{name}_{typ}.json";d=json.loads(f.read_text());assert d["prefix_audit"];hashes[str(f.relative_to(r))]=hashlib.sha256(f.read_bytes()).hexdigest();parts[typ]=d
 original=r/"results"/name
 for file in ["model.pt","selection.json","controller_diagnostics.json","source_replay_diagnostics.json"]:
  f=original/file;hashes[str(f.relative_to(r))]=hashlib.sha256(f.read_bytes()).hexdigest()
 models[name]=dict(temporal={k:parts["temporal"][k] for k in ["selected","transition_matrix","prior"]},head=parts["head"]["selections"],prototype={k:parts["prototype"][k] for k in ["fixed_selection","online_selection","centroids"]},controller_lr=json.loads((original/"selection.json").read_text())["lr"])
for file in ["common.py","adaptation.py","controllers.py","phase2_temporal.py","phase2_head.py","phase2_prototype.py"]:
 f=r/"src"/file;hashes[str(f.relative_to(r))]=hashlib.sha256(f.read_bytes()).hexdigest()
lock=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cohort=list(range(20,39)),models=models,sha256=hashes,primary_comparison="EEGNet causal source-selected filter versus fixed EMA alpha0.8",secondary_comparisons=["identity_future versus EMA0.8 and identity_confidence/current","temporal-head versus frozen and EMA0.8","online prototype versus source prototype"],metrics="Five-class per-participant MacroF1; average seeds and source folds within each new participant before cohort averaging; delta<-0.01 participant risk; transition +/-2 windows reported separately; CE and recalls",interpretation="Same-dataset fresh subjects, not external validation. Development selection complete before decoding fresh labels. Primary paired participant bootstrap descriptive 95% CI; other comparisons exploratory, no multiplicity-adjusted confirmation.",allowed_cpu_ids=list(range(8,24)),gpu_allowed=False)
path.write_text(json.dumps(lock,indent=2));print("LOCK",path,hashlib.sha256(path.read_bytes()).hexdigest())
