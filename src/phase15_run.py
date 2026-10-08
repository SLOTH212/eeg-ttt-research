from pathlib import Path
import subprocess,os,concurrent.futures,itertools,json
r=Path(__file__).resolve().parents[1];env=dict(os.environ,CUDA_VISIBLE_DEVICES="",EEG_THREADS="1",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1");out=r/"phase15"
protocol=dict(scope="Exploratory head-only retention on frozen LaBraM and linear spectral probes, existing observed cohorts; no new blinded claim.",sources=["young","mixed"],representations=["pretrained400","pretrained_plus_spectral452","spectral52"],objectives=["entropy","temporal"],controller_subjects=[7,17,27,37],validation_subjects=[5,15,25,35],evaluation_development=[6,8,9,16,18,19,26,28,29,36,38],evaluation_older=list(range(40,59)),lr_grid=[0.,1e-5,1e-4,.001,.01],anchors=[0.,.1],teacher_alpha=.8,projection="Same as phase2 head: each ofW,b norm ball radius .1*max(source norm,1);globalgradientnormclip1;no momentum",online_labels=False,controller_targets="Source-only counterfactual H4 future CE gain; current and confidence controls; confidence also trained on future gain",order="Predict current epoch before parameter action; RESET parameters only",cpu_ids=list(range(8,24)),gpu=False)
(out/"protocol.json").write_text(json.dumps(protocol,indent=2))
def run(spec):
 source,rep,obj=spec
 with (out/f"{source}_{rep}_{obj}.log").open("w") as f:result=subprocess.run(["taskset","-c","8-23",str(r/".venv/bin/python"),str(r/"src/phase15_head_retention.py"),"--source",source,"--representation",rep,"--objective",obj],env=env,stdout=f,stderr=subprocess.STDOUT)
 print(*spec,result.returncode,flush=True);return result.returncode
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:codes=list(pool.map(run,itertools.product(protocol["sources"],protocol["representations"],protocol["objectives"])))
assert not any(codes)
