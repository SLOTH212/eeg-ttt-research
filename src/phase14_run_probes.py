from pathlib import Path
import subprocess,os,concurrent.futures
r=Path(__file__).resolve().parents[1];env=dict(os.environ,CUDA_VISIBLE_DEVICES="",EEG_THREADS="1",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
def run(spec):
 source,representation=spec
 with (r/"phase14"/f"probe_{source}_{representation}.log").open("w") as f:result=subprocess.run(["taskset","-c","8-23",str(r/".venv/bin/python"),str(r/"src/phase14_probe.py"),"--source",source,"--representation",representation],env=env,stdout=f,stderr=subprocess.STDOUT)
 print(source,representation,result.returncode,flush=True);return result.returncode
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:codes=list(pool.map(run,[(s,v) for s in ["young","mixed"] for v in ["pretrained400","random400","spectral52","pretrained_plus_spectral452"]]))
assert not any(codes)
