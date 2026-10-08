from pathlib import Path
import subprocess,os,json,time
r=Path(__file__).resolve().parents[1];p=json.loads((r/"phase41/protocol.json").read_text());env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",EEG_THREADS="2",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1")
while not (r/"phase40/complete.json").exists():time.sleep(15)
for lr in p["lrs"]:
 for trajectory in p["trajectories"]:
  for sid in p["controllers"]:
   name=f"lr{lr:g}_{trajectory}_s{sid:02d}"
   if (r/"phase41"/name/"complete.json").exists():continue
   with (r/"phase41"/f"{name}.log").open("a") as f:
    result=subprocess.run(["taskset","-c","8-23",str(r/".venv/bin/python"),str(r/"src/phase41_pulses.py"),"--subject",str(sid),"--lr",str(lr),"--trajectory",trajectory],env=env,stdout=f,stderr=subprocess.STDOUT)
   print(name,result.returncode,flush=True)
   if result.returncode:raise SystemExit(result.returncode)

for script in ["phase41_feature_audit.py","phase41_observability.py","phase41_report.py"]:
 with (r/"phase41"/(script+".log")).open("a") as f:
  q=subprocess.run(["taskset","-c","8-23",str(r/".venv/bin/python"),str(r/"src"/script)],env=env,stdout=f,stderr=subprocess.STDOUT)
 print(script,q.returncode,flush=True)
 if q.returncode:raise SystemExit(q.returncode)
