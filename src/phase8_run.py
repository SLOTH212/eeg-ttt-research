from pathlib import Path
import os,subprocess,concurrent.futures,json
r=Path(__file__).resolve().parents[1];out=r/"phase8";out.mkdir(exist_ok=True);cases=["spectral_A_17","eegnet_A_17","context11_young_17","context11_mixed_17"]
(out/"protocol.json").write_text(json.dumps(dict(cases=cases,scope="Source-selected exploratory scoped ETA/EATA/SAR variants. Fixed running stats and BN1d support are deliberate EEG protocol changes; no claim of full paper reproduction",references=["https://github.com/mr-eggplant/EATA","https://github.com/mr-eggplant/SAR"],fisher="200 uniformly sampled unlabeled source windows per training participant, pseudo-label CE gradient squares averaged per window",allowed_cpu_ids=list(range(8,24)),gpu_allowed=False),indent=2))
def run(case):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES="",EEG_THREADS="1",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
 with open(out/f"{case}.log","a") as f:p=subprocess.run([str(r/".venv/bin/python"),str(r/"src/phase8_screen.py"),"--case",case],env=env,stdout=f,stderr=subprocess.STDOUT)
 print(case,p.returncode,flush=True)
 if p.returncode:raise RuntimeError(case)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(run,cases))
