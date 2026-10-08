from pathlib import Path
import os,subprocess,concurrent.futures,json,itertools
r=Path(__file__).resolve().parents[1];out=r/"phase12";out.mkdir(exist_ok=True);cases=[f"{k}_{f}_{z}" for k,f,z in itertools.product(["spectral","eegnet"],["A","B"],[17,29,43])]+[f"context11_{f}_{z}" for f,z in itertools.product(["young","mixed"],[17,29,43])]
(out/"protocol.json").write_text(json.dumps(dict(purpose="New write objective: BN-affine temporal teacher distillation instead of entropy; frozen teacher EMA is causal and retains history across parameter-only resets",cases=cases,features="Original13 plus teacher5probabilities,teacher entropy,current-teacher L1 disagreement; future/current controllers share all20; confidence control sees current+teacher entropy/disagreement",baselines="Direct teacherEMA/source-selectedEMA/weakEMA must be compared, not only frozen",selection="Source validation only, including no-update baseline; exploratory after previous cohorts observed",allowed_cpu_ids=list(range(8,24)),gpu_allowed=False),indent=2))
def run(case):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES="",EEG_THREADS="1",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
 with open(out/f"{case}.log","a") as f:p=subprocess.run([str(r/".venv/bin/python"),str(r/"src/phase12_temporal_bn.py"),"--case",case],env=env,stdout=f,stderr=subprocess.STDOUT)
 print(case,p.returncode,flush=True)
 if p.returncode:raise RuntimeError(case)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:list(ex.map(run,cases))
