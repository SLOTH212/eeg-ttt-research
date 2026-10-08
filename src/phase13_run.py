from pathlib import Path
import os,json,itertools,subprocess,concurrent.futures
r=Path(__file__).resolve().parents[1];out=r/"phase13";out.mkdir(exist_ok=True);cases=[f"{k}_{f}_{s}" for k,f,s in itertools.product(["spectral","eegnet"],["A","B"],[17,29,43])]+[f"context11_{f}_{s}" for f,s in itertools.product(["young","mixed"],[17,29,43])]
(out/"protocol.json").write_text(json.dumps(dict(purpose="Static supervised calibration using precisely the H4 controller label windows; separately match writable BN-affine and head subspaces plus original0.1per-tensor projection; source-validation epoch0 included",cases=cases,training="Only controller-source matched label windows,20epochs AdamW0.001,weight_decay0.001,model.eval fixed running stats/dropout; no target labels",interpretation="Exploratory positive control for parameter capacity and source-label-use fairness, not unsupervised test-time learning",allowed_cpu_ids=list(range(8,24)),gpu_allowed=False),indent=2))
def run(case):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES="",EEG_THREADS="4" if case.startswith("eegnet") else "1",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="4" if case.startswith("eegnet") else "1")
 with open(out/f"{case}.log","a") as f:p=subprocess.run([str(r/".venv/bin/python"),str(r/"src/phase13_static_scope.py"),"--case",case],env=env,stdout=f,stderr=subprocess.STDOUT)
 print(case,p.returncode,flush=True)
 if p.returncode:raise RuntimeError(case)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(run,cases))
