from phase28_joint_train import *
class LogitForward:
 def __init__(self,model,example):
  self.model=model;self.x=example.clone().cuda();self.channels=torch.tensor([0,127,128],device="cuda");stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream),torch.inference_mode():
   for _ in range(3):self.forward()
  torch.cuda.current_stream().wait_stream(stream);self.graph=torch.cuda.CUDAGraph()
  with torch.inference_mode(),torch.cuda.graph(self.graph):self.forward()
 def forward(self):
  z=self.model.encoder.forward_features(self.x.flatten(0,1),input_chans=self.channels,return_patch_tokens=True).mean(1).reshape(-1,400);self.logits=self.model.head((z-self.model.mean)/self.model.scale)
 def run(self,x):self.x.copy_(x);self.graph.replay();return self.logits
if __name__=="__main__":
 x,_=prepared(7);results=[]
 for weight in [0.,.01]:
  m=JointCarrier("young").cuda().eval();m.load_state_dict(torch.load(ROOT/"phase28"/f"young_joint_{weight:g}"/"model.pt",map_location="cpu",weights_only=False)["state"]);m.requires_grad_(False);params=[p for module in m.encoder.modules() if isinstance(module,nn.LayerNorm) for p in module.parameters()];initial=[p.clone() for p in params];forward=LogitForward(m,x[:5].cuda());errors=[]
  with torch.no_grad():
   for i in range(12):
    for p,v in zip(params,initial):p.copy_(v+(.0001*i)*torch.sin(v*17))
    xx=x[i*113:i*113+5].cuda();expected=m(xx);actual=forward.run(xx).clone();err=float((expected-actual).abs().max());assert err<1e-5 and torch.equal(expected.argmax(1),actual.argmax(1));errors.append(err)
   for p,v in zip(params,initial):p.copy_(v)
  results.append(dict(weight=weight,batches=12,windows=60,max_logit_error=max(errors),argmax_disagreements=0))
 (ROOT/"phase29/forward_graph_audit.json").write_text(json.dumps(dict(status="PASS",results=results,scope="Eager versus captured 5-window logit forward, including changed writable parameter values"),indent=2));print(results)
