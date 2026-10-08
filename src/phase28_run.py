from pathlib import Path
import subprocess,os,json,time
r=Path(__file__).resolve().parents[1];protocol=json.loads((r/"phase28/protocol.json").read_text());env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",EEG_THREADS="4",OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="1")
while len(list((r/"phase28/tokens").glob("*.npz")))<23:time.sleep(10)
for source in protocol["sources"]:
 for weight in protocol["auxiliary_weights"]:
  name=f"{source}_joint_{weight:g}"
  if (r/"phase28"/name/"complete.json").exists():continue
  with (r/"phase28"/f"{name}.log").open("a") as f:
   result=subprocess.run(["taskset","-c","8-23",str(r/".venv/bin/python"),str(r/"src/phase28_joint_train.py"),"--source",source,"--weight",str(weight)],env=env,stdout=f,stderr=subprocess.STDOUT)
  print(name,result.returncode,flush=True)
  if result.returncode:raise SystemExit(result.returncode)
