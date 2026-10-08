from pathlib import Path
import subprocess,os,json,time
r=Path(__file__).resolve().parents[1];p=json.loads((r/"phase43/protocol.json").read_text());env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",EEG_THREADS="2",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1")
while not (r/"phase42/audit.json").exists():time.sleep(15)
assert json.loads((r/"phase42/audit.json").read_text())["status"]=="PASS"
for sid in p["original_controllers"]:
 cache=r/"phase43/cache"/f"s{sid:02d}.npz"
 if not cache.exists():cache.symlink_to(r/"phase41/cache"/f"s{sid:02d}.npz")
 for trajectory in p["trajectories"]:
  name=f"lr0.01_{trajectory}_s{sid:02d}";dest=r/"phase43"/name
  if not dest.exists():dest.symlink_to(r/"phase41"/name,target_is_directory=True)
for trajectory in p["trajectories"]:
 for sid in p["additional_controllers"]:
  name=f"lr0.01_{trajectory}_s{sid:02d}"
  with (r/"phase43"/(name+".log")).open("a") as f:
   q=subprocess.run(["taskset","-c","8-23",str(r/".venv/bin/python"),str(r/"src/phase43_pulses.py"),"--subject",str(sid),"--lr",".01","--trajectory",trajectory],env=env,stdout=f,stderr=subprocess.STDOUT)
  print(name,q.returncode,flush=True)
  if q.returncode:raise SystemExit(q.returncode)
with (r/"phase43/feature_audit.log").open("a") as f:
 q=subprocess.run(["taskset","-c","8-23",str(r/".venv/bin/python"),str(r/"src/phase43_feature_audit.py")],env=env,stdout=f,stderr=subprocess.STDOUT)
print("feature audit",q.returncode,flush=True)
if q.returncode:raise SystemExit(q.returncode)
(r/"phase43/pulses_complete.json").write_text(json.dumps(dict(complete=True,original_cases=8,new_cases=10),indent=2))
