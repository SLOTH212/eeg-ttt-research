from pathlib import Path
import subprocess,os,json
r=Path(__file__).resolve().parents[1];env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",EEG_THREADS="2",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1")
commands=[["--audit"]] if not (r/"phase37/graph_audit.json").exists() else []
commands += [["--source",s,"--variant",v,"--lr",str(lr)] for s in ["young","mixed"] for v in ["fitted","unfitted"] for lr in [.0001,.001,.01]]
for args in commands:
 name="_".join(args).replace("--","")
 with (r/"phase37"/(name+".log")).open("a") as f:
  result=subprocess.run(["taskset","-c","8-23",str(r/".venv/bin/python"),str(r/"src/phase37_screen.py")]+args,env=env,stdout=f,stderr=subprocess.STDOUT)
 print(name,result.returncode,flush=True)
 if result.returncode:raise SystemExit(result.returncode)
