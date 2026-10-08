from pathlib import Path
import subprocess,os,itertools,concurrent.futures,json
r=Path(__file__).resolve().parents[1];out=r/"phase17";env=dict(os.environ,CUDA_VISIBLE_DEVICES="",EEG_THREADS="2",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1");protocol=dict(scope="Source-validation-only continuous LayerNorm adaptation screen, existing pretrained400 probes",sources=["young","mixed"],objectives=["entropy","masked_token"],lrs=[.0001,.001,.01],validation_subjects=[5,15,25,35],prediction_order="Current30s prediction before update;reset sourceparameters perperson",selection="Mean perperson fixed5classMacroF1, include frozen fallback",writable="Existing LayerNorm affine only;other parameters fixed",projection="Per-tensor .1*max(sourceRMS,1)*sqrt(numel);globalgradientnormclip1",auxiliary="Complementary50%mask currentepoch tokens;fixed official tokenizer cached without labels",cpu_ids=list(range(8,24)),gpu=False)
if not (out/"protocol.json").exists():(out/"protocol.json").write_text(json.dumps(protocol,indent=2))
def run(spec):
 source,obj,lr=spec
 with (out/f"{source}_{obj}_{lr}.log").open("w") as f:result=subprocess.run(["taskset","-c","8-23",str(r/".venv/bin/python"),str(r/"src/phase17_source_screen.py"),"--source",source,"--objective",obj,"--lr",str(lr)],env=env,stdout=f,stderr=subprocess.STDOUT)
 print(*spec,result.returncode,flush=True);return result.returncode
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:codes=list(pool.map(run,itertools.product(protocol["sources"],protocol["objectives"],protocol["lrs"])))
assert not any(codes)
