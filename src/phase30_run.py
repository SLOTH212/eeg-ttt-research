from pathlib import Path
import subprocess,os,json
r=Path(__file__).resolve().parents[1];p=json.loads((r/"phase30/protocol.json").read_text());env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",EEG_THREADS="2",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1")
for weight in p["weights"]:
 for objective in p["objectives"]:
  for lr in p["lrs"]:
   name=f"young_w{weight:g}_{objective}_{lr:g}"
   if (r/"phase30"/name/"complete.json").exists():continue
   with (r/"phase30"/f"{name}.log").open("a") as f:
    result=subprocess.run(["taskset","-c","8-23",str(r/".venv/bin/python"),str(r/"src/phase30_screen.py"),"--weight",str(weight),"--objective",objective,"--lr",str(lr)],env=env,stdout=f,stderr=subprocess.STDOUT)
   print(name,result.returncode,flush=True)
   if result.returncode:raise SystemExit(result.returncode)
