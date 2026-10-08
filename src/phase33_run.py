from pathlib import Path
import subprocess,os,json
r=Path(__file__).resolve().parents[1];p=json.loads((r/"phase33/protocol.json").read_text());env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",EEG_THREADS="4",OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="1")
for seed in p["seeds"]:
 for weight in p["weights"]:
  name=f"young_joint_{weight:g}_seed{seed}"
  if (r/"phase33"/name/"complete.json").exists():continue
  with (r/"phase33"/f"{name}.log").open("a") as f:
   result=subprocess.run(["taskset","-c","8-23",str(r/".venv/bin/python"),str(r/"src/phase33_train.py"),"--source","young","--weight",str(weight),"--seed",str(seed)],env=env,stdout=f,stderr=subprocess.STDOUT)
  print(name,result.returncode,flush=True)
  if result.returncode:raise SystemExit(result.returncode)
