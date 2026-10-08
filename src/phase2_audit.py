from common import *
from phase2_head import stream
import hashlib
out=ROOT/"phase2";checks=[]
for kind in ["spectral","eegnet"]:
 name=f"{kind}_A_17";m,_=load_model(ROOT/"results"/name/"model.pt");d=np.load(out/"cache"/f"{name}_s10.npz");z=d["z"][:31];pr=d["p"][:31];W0=m.head.weight.detach().numpy();b0=m.head.bias.detach().numpy()
 for objective in ["entropy","pseudo","temporal"]:
  actual,_=stream(z,pr,W0,b0,objective,.01,1.)
  W=torch.tensor(W0,dtype=torch.float64,requires_grad=True);b=torch.tensor(b0,dtype=torch.float64,requires_grad=True);Ws=W.detach().clone();bs=b.detach().clone();teacher=pr[0].astype(float).copy();expected=[]
  for v,p in zip(z,pr):
   q=(W@torch.tensor(v,dtype=torch.float64)+b).softmax(0);expected.append(q.detach().numpy().copy());teacher=.8*p+.2*teacher
   if objective=="entropy":loss=-(q*q.log()).sum()
   elif objective=="pseudo":loss=-q[int(p.argmax())].log()
   else:loss=-(torch.tensor(teacher)*q.log()).sum()
   loss=loss+.5*((W-Ws).square().sum()+(b-bs).square().sum());loss.backward()
   with torch.no_grad():
    norm=(W.grad.square().sum()+b.grad.square().sum()).sqrt();scale=min(1.,1/max(float(norm),1e-12));W-=.01*scale*W.grad;b-=.01*scale*b.grad
    for value,source,original in [(W,Ws,W0),(b,bs,b0)]:
     delta=value-source;limit=.1*max(float(np.linalg.norm(original)),1.);value.copy_(source+delta*min(1.,limit/max(float(delta.norm()),1e-12)))
   W.grad=None;b.grad=None
  err=float(np.max(abs(np.array(expected)-actual)))
  # Float32 input probability mass differs from 1 by at most O(1e-7); CE gradient
  # mathematically is mass*q-target. Manual q-target differs only by that rounding.
  assert err<2e-6,(kind,objective,err)
  checks.append(dict(kind=kind,objective=objective,torch_autograd_max_probability_error=err))
lock=json.loads((out/"evaluation_lock.json").read_text())
for name,h in lock["sha256"].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
threads=[]
for pd in Path("/proc").iterdir():
 if not pd.name.isdigit():continue
 try:
  cmd=(pd/"cmdline").read_bytes().replace(b"\0",b" ").decode(errors="replace")
  if "/src/phase2_" not in cmd or "python" not in cmd or "bash -c" in cmd:continue
  for tid in (pd/"task").iterdir():
   affinity=sorted(os.sched_getaffinity(int(tid.name)));assert set(affinity)<=set(range(8,24));threads.append(dict(pid=int(pd.name),tid=int(tid.name),affinity=affinity))
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
(out/"results/phase2_audit.json").write_text(json.dumps(dict(head_reference=checks,locked_hashes_valid=True,threads=threads,gpu_visible=os.environ.get("CUDA_VISIBLE_DEVICES")),indent=2));print(json.dumps(dict(checks=checks,audited_threads=len(threads))))
