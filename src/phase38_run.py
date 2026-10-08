from pathlib import Path
import os,subprocess,time,json
r=Path(__file__).resolve().parents[1];env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",EEG_THREADS="2",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1")
while len(list((r/"phase37").glob("*/complete.json")))<12:
 time.sleep(15)
print("PHASE37 COMPLETE; START FIXED SEED REPLICATION",flush=True)
for seed in [29,43]:
 tasks=[("phase38_head_train.py",["--source","young","--classifier-seed",str(seed)])]+[("phase38_screen.py",["--source","young","--classifier-seed",str(seed),"--variant",v,"--lr",".01"]) for v in ["fitted","unfitted"]]
 for script,args in tasks:
  name=script.replace(".py","")+"_"+"_".join(args).replace("--","")
  with (r/"phase38"/(name+".log")).open("a") as f:
   result=subprocess.run(["taskset","-c","8-23",str(r/".venv/bin/python"),str(r/"src"/script)]+args,env=env,stdout=f,stderr=subprocess.STDOUT)
  print(name,result.returncode,flush=True)
  if result.returncode:raise SystemExit(result.returncode)
(r/"phase38/run_complete.json").write_text(json.dumps(dict(complete=True),indent=2))
