from pathlib import Path
import subprocess,os,time
r=Path(__file__).resolve().parents[1]
while not (r/"phase43/pulses_complete.json").exists():time.sleep(15)
env=dict(os.environ,EEG_THREADS="2",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1",CUDA_VISIBLE_DEVICES="")
with (r/"phase43/summary.log").open("a") as f:
 p=subprocess.run(["taskset","-c","8-23",str(r/".venv/bin/python"),str(r/"src/phase43_diversity.py")],env=env,stdout=f,stderr=subprocess.STDOUT)
print("diversity analysis",p.returncode,flush=True)
raise SystemExit(p.returncode)
