from pathlib import Path
import os,json,subprocess,concurrent.futures,argparse
r=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument("--mode",required=True);p.add_argument("--workers",type=int,default=4);args=p.parse_args();names=list(json.loads((r/"phase2/evaluation_lock.json").read_text())["models"])
def run(name):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES="",EEG_THREADS="1",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
 with open(r/"phase2"/f"fresh_{args.mode}_{name}.log","a") as f:
  result=subprocess.run([str(r/".venv/bin/python"),str(r/"src/phase2_fresh.py"),"--name",name,"--mode",args.mode],env=env,stdout=f,stderr=subprocess.STDOUT)
 return name,result.returncode
with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
 for name,code in ex.map(run,names):
  print(name,code,flush=True)
  if code:raise RuntimeError(name)
