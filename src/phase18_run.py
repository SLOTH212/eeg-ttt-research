from pathlib import Path
import subprocess,os,itertools
r=Path(__file__).resolve().parents[1];env=dict(os.environ,CUDA_VISIBLE_DEVICES="",EEG_THREADS="4",OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="1")
for kind,source in itertools.product(["pretrained","random"],["young","mixed"]):
 with (r/"phase18"/f"{source}_{kind}.log").open("w") as f:result=subprocess.run(["taskset","-c","8-23",str(r/".venv/bin/python"),str(r/"src/phase18_source_tune.py"),"--source",source,"--kind",kind],env=env,stdout=f,stderr=subprocess.STDOUT)
 print(source,kind,result.returncode,flush=True)
 if result.returncode:raise SystemExit(result.returncode)
