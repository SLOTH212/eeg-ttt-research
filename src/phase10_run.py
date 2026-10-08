from pathlib import Path
import os,json,subprocess,concurrent.futures
r=Path(__file__).resolve().parents[1];out=r/"phase10";out.mkdir(exist_ok=True);cases=["spectral_A_17","spectral_B_17","eegnet_A_17","eegnet_B_17","context11_young_17","context11_mixed_17"]
(out/"protocol.json").write_text(json.dumps(dict(purpose="Source-only horizon sensitivity for future-benefit controllers, fixed models/LR/features;1,4,10futurewindows, paired future updates suspended as in original protocol",cases=cases,allowed_cpu_ids=list(range(8,24)),gpu_allowed=False),indent=2))
def run(case):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES="",EEG_THREADS="1",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
 with open(out/f"{case}.log","a") as f:p=subprocess.run([str(r/".venv/bin/python"),str(r/"src/phase10_horizons.py"),"--case",case],env=env,stdout=f,stderr=subprocess.STDOUT)
 print(case,p.returncode,flush=True)
 if p.returncode:raise RuntimeError(case)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(run,cases))
