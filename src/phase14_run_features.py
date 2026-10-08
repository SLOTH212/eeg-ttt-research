from pathlib import Path
import subprocess,os,concurrent.futures
r=Path(__file__).resolve().parents[1];env=dict(os.environ,CUDA_VISIBLE_DEVICES="",EEG_THREADS="2",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1")
def run(sid):
 with (r/"phase14"/f"features_s{sid:02d}.log").open("w") as f:
  result=subprocess.run(["taskset","-c","8-23",str(r/".venv/bin/python"),str(r/"src/phase14_features.py"),"--subject",str(sid)],env=env,stdout=f,stderr=subprocess.STDOUT)
 print(sid,result.returncode,flush=True);return result.returncode
subjects=list(range(20))+[25,35]+[s for s in range(20,59) if s not in [25,35,39]]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:codes=list(pool.map(run,subjects))
assert not any(codes)
