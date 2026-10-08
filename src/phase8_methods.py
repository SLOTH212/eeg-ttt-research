"""Scoped causal ETA/EATA/SAR mechanisms. Fixed BN running statistics, not exact paper reproduction.
References: https://github.com/mr-eggplant/EATA ; https://github.com/mr-eggplant/SAR
Reimplemented for batch1 EEG, predict-before-update, SGD momentum0.9, no projection.
"""
from common import *
from adaptation import Adapter,entropy
import time

def fisher_diagonal(model,records,per_subject=200,seed=17):
 a=Adapter(model);f=[torch.zeros_like(p) for p in a.params];rng=np.random.default_rng(seed);n=0
 for x in records:
  for index in rng.choice(len(x),min(per_subject,len(x)),replace=False):
   a.model.zero_grad(set_to_none=True);logits=a.model(torch.tensor(np.array(x[index:index+1]),dtype=torch.float32));label=logits.detach().argmax(1);nn.functional.cross_entropy(logits,label).backward()
   for fi,p in zip(f,a.params):fi.add_(p.grad.detach().square())
   n+=1
 return [v/n for v in f],n

def robust_stream(model,x,method,lr,fisher=None,alpha=0.,diversity=.05,reset_entropy=.2,rho=.05):
 a=Adapter(model,lr);opt=torch.optim.SGD(a.params,lr=lr,momentum=.9);margin=.4*np.log(5);prob_memory=None;entropy_ema=None;pred=[];accepted=0;reliable=0;resets=0;start=time.perf_counter()
 for t in range(len(x)):
  xt=torch.tensor(np.array(x[t:t+1]),dtype=torch.float32);a.model.zero_grad(set_to_none=True);logits=a.model(xt)
  if not torch.isfinite(logits).all():raise FloatingPointError(f"nonfinite logits at {t}")
  p=logits.softmax(1);h=entropy(p).mean();pred.append(p.detach().numpy()[0].copy())
  if float(h.detach())<margin:
   reliable+=1
   if method in ["eta","eata"]:
    allowed=prob_memory is None or float(nn.functional.cosine_similarity(prob_memory,p.detach()[0],dim=0))<diversity
    if allowed:
     prob_memory=p.detach()[0].clone() if prob_memory is None else .9*prob_memory+.1*p.detach()[0]
     loss=h*torch.exp(torch.tensor(margin)-h.detach())
     if method=="eata":loss=loss+alpha*sum((fi*(v-s).square()).sum() for fi,v,s in zip(fisher,a.params,a.source))
     loss.backward();opt.step();accepted+=1
   elif method=="sar":
    h.backward();old=a.snapshot();norm=sum(v.grad.square().sum() for v in a.params).sqrt()
    with torch.no_grad():
     for v in a.params:v.add_(rho*v.grad/(norm+1e-12))
    a.model.zero_grad(set_to_none=True);h2=entropy(a.model(xt).softmax(1)).mean()
    if torch.isfinite(h2) and float(h2.detach())<margin:
     h2.backward();a.restore(old);opt.step();accepted+=1;value=float(h2.detach());entropy_ema=value if entropy_ema is None else .9*entropy_ema+.1*value
    else:a.restore(old)
   else:raise ValueError(method)
  # Match source-recovery semantics: source parameters+optimizer restored, entropy EMA retained.
  if method=="sar" and entropy_ema is not None and entropy_ema<reset_entropy:
   a.restore(a.source);opt.state.clear();resets+=1
  if any(not torch.isfinite(v).all() for v in a.params):raise FloatingPointError(f"nonfinite parameters at {t}")
 return np.array(pred),dict(accepted=accepted,reliable=reliable,resets=resets,seconds=time.perf_counter()-start,final_parameter_drift=a.drift())
