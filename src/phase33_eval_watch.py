from pathlib import Path
import subprocess,os,time
r=Path(__file__).resolve().parents[1];env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",EEG_THREADS="2",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1")
while True:
 for seed in [29,43]:
  for weight in [0.,.01]:
   name=f"young_joint_{weight:g}_seed{seed}"
   if not (r/"phase33"/name/"validation_audit.json").exists() or (r/"phase33/evaluation"/name/"complete.json").exists():continue
   with (r/"phase33"/f"eval_{name}.log").open("a") as f:
    result=subprocess.run(["taskset","-c","8-23",str(r/".venv/bin/python"),str(r/"src/phase33_evaluate.py"),"--weight",str(weight),"--seed",str(seed)],env=env,stdout=f,stderr=subprocess.STDOUT)
   print(name,result.returncode,flush=True)
   if result.returncode:raise SystemExit(result.returncode)
 if len(list((r/"phase33/evaluation").glob("*/complete.json")))==4:break
 time.sleep(20)
