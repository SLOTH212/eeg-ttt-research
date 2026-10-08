from pathlib import Path
import subprocess,os,json
r=Path(__file__).resolve().parents[1];protocol=json.loads((r/"phase21/protocol.json").read_text());env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",EEG_THREADS="4",OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="1")
for source in protocol["sources"]:
 for kind in protocol["kinds"]:
  for lr in protocol["encoder_lrs"][kind]:
   name=f"{source}_{kind}_{lr:g}"
   if (r/"phase21"/name/"complete.json").exists():continue
   with (r/"phase21"/f"{name}.log").open("a") as f:
    result=subprocess.run(["taskset","-c","8-23",str(r/".venv/bin/python"),str(r/"src/phase21_full_encoder.py"),"--source",source,"--kind",kind,"--lr",str(lr)],env=env,stdout=f,stderr=subprocess.STDOUT)
   print(name,result.returncode,flush=True)
   if result.returncode:raise SystemExit(result.returncode)
