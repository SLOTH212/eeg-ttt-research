from pathlib import Path
import subprocess,os,itertools,concurrent.futures,json
r=Path(__file__).resolve().parents[1];out=r/"phase17";env=dict(os.environ,CUDA_VISIBLE_DEVICES="",EEG_THREADS="2",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1");protocol=json.loads((out/"protocol.json").read_text())
def run(spec):
 source,obj,lr=spec
 with (out/f"{source}_{obj}_{lr}.log").open("w") as f:result=subprocess.run(["taskset","-c","8-23",str(r/".venv/bin/python"),str(r/"src/phase17_source_screen.py"),"--source",source,"--objective",obj,"--lr",str(lr)],env=env,stdout=f,stderr=subprocess.STDOUT)
 print(*spec,result.returncode,flush=True);return result.returncode
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:codes=list(pool.map(run,itertools.product(protocol["sources"],protocol["objectives"],protocol["extension_lrs"])))
assert not any(codes)
