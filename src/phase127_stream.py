from phase119_models import ResidualLogitHead
import numpy as np,torch,json,hashlib
from pathlib import Path
class CausalLogitReadout:
 """Frozen head, ten cached centered parent logits, and fixed causal EMA."""
 def __init__(self,checkpoint,expected_parent_sha256,input_pipeline_id,ema_alpha=.5):
  path=Path(checkpoint);ck=torch.load(path,map_location="cpu",weights_only=False)
  if ck.get("variant")!="history11" or ck.get("dim")!=55:raise ValueError("Expected fixed 11-window history head")
  if ck.get("parent_sha256")!=expected_parent_sha256:raise ValueError("Parent checkpoint identity mismatch")
  if not isinstance(input_pipeline_id,str) or not input_pipeline_id:raise ValueError("Input pipeline identity required")
  if not 0<float(ema_alpha)<=1:raise ValueError("EMA alpha out of range")
  self._model=ResidualLogitHead(55);self._model.load_state_dict(ck["state"]);self._model.eval().requires_grad_(False)
  self._config=dict(version=1,head_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),parent_sha256=expected_parent_sha256,input_pipeline_id=input_pipeline_id,ema_alpha=float(ema_alpha))
  self.reset()
 def reset(self):
  self._history=[];self._ema=None;self._count=0
 @staticmethod
 def _check_logit(x):
  if x.shape!=(5,) or x.dtype!=np.float32 or not np.isfinite(x).all():raise ValueError("Expected finite float32 centered logit vector of length5")
  if abs(float(x.mean(dtype=np.float64)))>1e-5*max(1.,float(abs(x).max())):raise ValueError("Parent logits must already be centered")
 def step(self,centered_parent_logits):
  x=np.asarray(centered_parent_logits);self._check_logit(x);x=x.copy()
  first=self._history[0] if self._history else x
  sequence=[first]*(10-len(self._history))+self._history+[x]
  features=np.concatenate(sequence).astype(np.float32,copy=False)
  with torch.inference_mode():p=self._model(torch.from_numpy(features)[None]).softmax(1)[0].numpy().copy()
  if not np.isfinite(p).all():raise ValueError("Nonfinite model prediction")
  previous=p if self._ema is None else self._ema;alpha=self._config["ema_alpha"];q=alpha*p+(1-alpha)*previous
  self._history=(self._history+[x])[-10:];self._ema=q.copy();self._count+=1
  return p.copy(),q.copy()
 def export_state(self):
  return json.loads(json.dumps(dict(config=self._config,count=self._count,history=[x.tolist() for x in self._history],ema=None if self._ema is None else self._ema.tolist()),allow_nan=False))
 def restore_state(self,state):
  st=json.loads(json.dumps(state,allow_nan=False))
  if st.get("config")!=self._config:raise ValueError("State/model/EMA/input-pipeline configuration mismatch")
  count=st.get("count")
  if isinstance(count,bool) or not isinstance(count,int) or count<0:raise ValueError("Invalid count")
  history=st.get("history")
  if not isinstance(history,list) or len(history)!=min(count,10):raise ValueError("Invalid history length")
  new=[]
  for value in history:
   x=np.asarray(value,dtype=np.float32);self._check_logit(x);new.append(x.copy())
  q=st.get("ema")
  if count==0:
   if q is not None:raise ValueError("Empty stream must not have EMA state")
  else:
   q=np.asarray(q,dtype=np.float32)
   if q.shape!=(5,) or not np.isfinite(q).all() or (q<0).any() or abs(float(q.sum())-1)>2e-5:raise ValueError("Invalid EMA state")
   q=q.copy()
  self._history=new;self._ema=q;self._count=count
