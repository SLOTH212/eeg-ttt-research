from pathlib import Path
import os,subprocess,concurrent.futures,json
r=Path(__file__).resolve().parents[1];out=r/"phase11";out.mkdir(exist_ok=True);cases=["spectral_A_17","eegnet_A_17","context11_young_17"]
(out/"protocol.json").write_text(json.dumps(dict(purpose="Offline mechanism diagnostic only: entropy-gradient alignment with future H4 supervised-task gradient; reset to source before each pulse; distinguish current correctness and future transitions",cases=cases,stride=20,target_labels="Used only to explain completed experiments, never fed to deployed policies or locked selections",allowed_cpu_ids=list(range(8,24)),gpu_allowed=False),indent=2))
def run(case):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES="",EEG_THREADS="1",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
 with open(out/f"{case}.log","a") as f:p=subprocess.run([str(r/".venv/bin/python"),str(r/"src/phase11_alignment.py"),"--case",case],env=env,stdout=f,stderr=subprocess.STDOUT)
 print(case,p.returncode,flush=True)
 if p.returncode:raise RuntimeError(case)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(run,cases))
