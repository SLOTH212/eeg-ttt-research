from pathlib import Path
import subprocess,os,json
r=Path(__file__).resolve().parents[1];o=r/"phase66";pr=json.loads((o/"protocol.json").read_text());env=dict(os.environ,OPENBLAS_NUM_THREADS="1",EEG_THREADS="2",OMP_NUM_THREADS="2")
for kind in pr["kinds"]:
 for lr in pr["encoder_lrs"][kind]:
  name=f"{kind}_{lr:g}"
  with (o/(name+".log")).open("a") as f:code=subprocess.call([str(r/".venv/bin/python"),"src/phase66_train.py","--kind",kind,"--lr",str(lr)],cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT)
  print(name,code,flush=True)
  if code:raise SystemExit(code)
(o/"training_complete.json").write_text(json.dumps(dict(complete=True,configs=4)))
