"""Fixed causal input controls, one completed 30-second window at a time."""
import json
import numpy as np
from phase99_covariance import root
class CausalInputState:
 def __init__(self,reference_rms,lower_rms,reference_shape,mode="full"):
  if mode not in ("scalar","floor","diagonal","full"):raise ValueError("Unsupported fixed mode")
  shape=np.asarray(reference_shape,dtype=np.float64)
  if shape.shape!=(2,2) or not np.isfinite(shape).all() or not np.allclose(shape,shape.T) or np.linalg.eigvalsh(shape).min()<=0:raise ValueError("Positive definite source shape required")
  if not 0<lower_rms<reference_rms:raise ValueError("Invalid fixed source RMS constants")
  self.config=dict(version=1,mode=mode,reference_rms=float(reference_rms),lower_rms=float(lower_rms),reference_shape=shape.tolist())
  self.source=.99*shape+.01*np.eye(2);self.source_root=root(self.source,.5);self.reset()
 def reset(self):
  self.log_rms=None;self.shape=None;self.windows=0
 def export_state(self):
  return dict(config=json.loads(json.dumps(self.config)),log_rms=self.log_rms,shape=None if self.shape is None else self.shape.tolist(),windows=self.windows)
 def restore_state(self,state):
  if state["config"]!=self.config:raise ValueError("State configuration differs from fixed method")
  log=state["log_rms"];shape=None if state["shape"] is None else np.asarray(state["shape"],dtype=np.float64);n=state["windows"]
  if not isinstance(n,int) or n<0 or (log is not None and not np.isfinite(log)):raise ValueError("Invalid state")
  if shape is not None and (shape.shape!=(2,2) or not np.isfinite(shape).all() or not np.allclose(shape,shape.T) or np.linalg.eigvalsh(shape).min()< -1e-12):raise ValueError("Invalid shape state")
  self.log_rms=log;self.shape=shape;self.windows=n
 def step(self,window):
  raw=np.asarray(window)
  if raw.shape!=(2,3000) or raw.dtype!=np.float32 or not np.isfinite(raw).all():raise ValueError("Expected finite float32 2x3000 input, centered/scaled/clipped by locked upstream preprocessing")
  if abs(raw).max()>20:raise ValueError("Input exceeds locked upstream clipping range")
  x=raw.astype(np.float64);rms=float(np.sqrt(np.mean(x*x)));gain=1.
  if rms>=1e-6:
   log=float(np.log(rms));self.log_rms=log if self.log_rms is None else .99*self.log_rms+.01*log;estimate=float(np.exp(self.log_rms))
   gain=max(1.,self.config["lower_rms"]/estimate) if self.config["mode"]=="floor" else self.config["reference_rms"]/estimate if estimate<self.config["lower_rms"] else 1.
   gain=float(np.clip(gain,1,100))
  cov=np.einsum("ct,dt->cd",x,x,optimize=False)/x.shape[-1];tr=float(np.trace(cov));valid=tr>=2e-12
  if valid:
   sh=2*cov/tr;self.shape=sh.copy() if self.shape is None else .99*self.shape+.01*sh
  mode=self.config["mode"]
  if mode in ("scalar","floor"):out=np.clip(raw*np.float64(gain),-20,20).astype(np.float32)
  elif gain>1 and valid:
   target=.99*self.shape+.01*np.eye(2);matrix=self.source_root@root(target,-.5) if mode=="full" else np.diag(np.sqrt(np.diag(self.source)/np.diag(target)));z=matrix@x;z*=np.sqrt(float(np.mean(x*x))/float(np.mean(z*z)))*gain;out=np.clip(z,-20,20).astype(np.float32)
  else:out=raw.copy()
  self.windows+=1
  return out,dict(gain=gain,log_rms=self.log_rms,shape=None if self.shape is None else self.shape.copy(),window_index=self.windows-1)
