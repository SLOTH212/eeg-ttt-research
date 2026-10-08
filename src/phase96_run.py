import subprocess,json,os
from pathlib import Path
r=Path(__file__).resolve().parents[1];assert os.sched_getaffinity(0)<=set(range(8,24))
assert json.loads((r/"phase94/audit.json").read_text())["advance"]
for seed in [29,43]:
 with (r/"phase96"/f"seed{seed}_train.log").open("w") as f:
  subprocess.run([str(r/".venv/bin/python"),str(r/"src/phase96_full_amplitude.py"),"--seed",str(seed)],cwd=r,stdout=f,stderr=subprocess.STDOUT,check=True)
 print("REPLICATE COMPLETE",seed,flush=True)
