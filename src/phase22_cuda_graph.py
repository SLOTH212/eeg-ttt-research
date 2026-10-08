from phase22_source_screen import *
# Whole-step graph follows https://docs.pytorch.org/docs/2.8/notes/cuda.html
# This is an execution optimization; it must pass eager trajectory equivalence before use.
class GraphAdapter:
 def __init__(self,model,objective,lr,example):
  self.model=model;self.objective=objective;self.params=[p for p in model.parameters() if p.requires_grad];self.initial=[p.detach().clone() for p in self.params];self.radii=[.1*max(float(v.square().mean().sqrt()),1)*v.numel()**.5 for v in self.initial];self.opt=torch.optim.SGD(self.params,lr=lr);self.x=example.clone().cuda();self.teacher=torch.ones(5,device="cuda")/5;self.channels=torch.tensor([0,127,128],device="cuda")
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for _ in range(3):self._step()
  torch.cuda.current_stream().wait_stream(stream);self.reset();self.opt.zero_grad(set_to_none=True);self.graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(self.graph):self._step()
  self.reset()
 def reset(self):
  with torch.no_grad():
   for p,v in zip(self.params,self.initial):p.copy_(v)
 def _step(self):
  self.opt.zero_grad(set_to_none=True);z=self.model.encoder.forward_features(self.x.flatten(0,1),input_chans=self.channels,return_patch_tokens=True).mean(1).reshape(-1,400);logits=self.model.head((z-self.model.mean)/self.model.scale);self.q=logits.softmax(1);self.loss=-(self.q*logits.log_softmax(1)).sum() if self.objective=="entropy" else -(self.teacher*logits.log_softmax(1)[0]).sum();self.loss.backward();self.norm=nn.utils.clip_grad_norm_(self.params,1.);self.opt.step()
  with torch.no_grad():
   for p,v,radius in zip(self.params,self.initial,self.radii):
    delta=p-v;factor=torch.clamp(radius/torch.linalg.vector_norm(delta).clamp_min(1e-12),max=1.);p.copy_(v+factor*delta)
  self.finite=torch.isfinite(self.loss)&torch.isfinite(self.norm)
 def step(self,x,teacher):
  self.x.copy_(x);self.teacher.copy_(teacher);self.graph.replay();return self.q

def writable_model():
 model=FullCarrier("young","pretrained").eval().cuda();model.load_state_dict(torch.load(source_checkpoint("young"),map_location="cpu",weights_only=False)["state"]);model.requires_grad_(False)
 for m in model.encoder.modules():
  if isinstance(m,nn.LayerNorm):
   for p in m.parameters():p.requires_grad_(True)
 return model

def benchmark():
 x,_=prepared(5);x=x[1280:1536].cuda();baseline=writable_model();base=probabilities(baseline,x);teacher=[];h=None
 for q in base:h=q.copy() if h is None else .8*q+.2*h;teacher.append(h.copy())
 teacher=torch.tensor(np.array(teacher),device="cuda");results=[]
 for objective in ["entropy","temporal"]:
  for lr in [1e-4,.01]:
   eager=writable_model();params=[p for p in eager.parameters() if p.requires_grad];initial=[p.detach().clone() for p in params];radii=[.1*max(float(v.square().mean().sqrt()),1)*v.numel()**.5 for v in initial];opt=torch.optim.SGD(params,lr=lr);pred=[];start=time.perf_counter()
   for t in range(len(x)):
    opt.zero_grad(set_to_none=True);z=eager(x[t:t+1]);q=z.softmax(1);pred.append(q.detach().cpu().numpy()[0]);loss=-(q*z.log_softmax(1)).sum() if objective=="entropy" else -(teacher[t]*z.log_softmax(1)[0]).sum();loss.backward();nn.utils.clip_grad_norm_(params,1.);opt.step()
    with torch.no_grad():
     for p,v,r in zip(params,initial,radii):
      delta=p-v;factor=torch.clamp(r/torch.linalg.vector_norm(delta).clamp_min(1e-12),max=1.);p.copy_(v+factor*delta)
   torch.cuda.synchronize();eager_seconds=time.perf_counter()-start;reference=np.array(pred);end=[p.detach().clone() for p in params];del eager,opt
   model=writable_model();graph=GraphAdapter(model,objective,lr,x[:1]);pred=[];start=time.perf_counter()
   for t in range(len(x)):
    pred.append(graph.step(x[t:t+1],teacher[t]).detach().cpu().numpy()[0].copy());assert bool(graph.finite)
   torch.cuda.synchronize();graph_seconds=time.perf_counter()-start;pred=np.array(pred);err=float(np.max(np.abs(pred-reference)));param_error=max(float(torch.max(torch.abs(p-v))) for p,v in zip(graph.params,end));disagreements=int((pred.argmax(1)!=reference.argmax(1)).sum());assert err<2e-5 and param_error<2e-6 and disagreements==0
   result=dict(objective=objective,lr=lr,windows=len(x),probability_error=err,parameter_error=param_error,argmax_disagreements=disagreements,eager_seconds=eager_seconds,graph_seconds=graph_seconds);results.append(result);print("GRAPH CHECK",result,flush=True);del graph,model
 (OUT22/"graph_benchmark.json").write_text(json.dumps(dict(status="PASS",numerical_execution_check_only=True,segment="subject5 windows1280:1536 reset-to-source; not a performance evaluation",results=results),indent=2))
if __name__=="__main__":benchmark()
