import os,subprocess,json
from pathlib import Path
r=Path(__file__).resolve().parents[1]
assert os.sched_getaffinity(0)<=set(range(8,24))
assert json.loads((r/"phase86/implementation_audit.json").read_text())["status"]=="PASS"
for which in ["future","current","reconstruction"]:
 with (r/"phase86"/(which+"_train.log")).open("w") as f:
  subprocess.run([str(r/".venv/bin/python"),str(r/"src/phase86_meta_waveform.py"),"--target",which],cwd=r,stdout=f,stderr=subprocess.STDOUT,check=True)
 print("FINISHED",which,flush=True)
