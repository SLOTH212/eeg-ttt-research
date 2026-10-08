from phase22_cuda_graph import GraphAdapter
from phase28_joint_train import JointCarrier
from phase21_full_encoder import *
OUT30=ROOT/"phase30"
class MaskedGraphAdapter(GraphAdapter):
 def __init__(self,model,lr,example):
  self.tokens=torch.zeros((1,2,30),dtype=torch.long,device="cuda");self.positions=torch.arange(0,30,2,device="cuda");self.mask=(torch.arange(30,device="cuda")[None]%2==0).expand(2,-1)
  super().__init__(model,"masked_token",lr,example)
 def _step(self):
  self.opt.zero_grad(set_to_none=True)
  with torch.no_grad():
   z=self.model.encoder.forward_features(self.x.flatten(0,1),input_chans=self.channels,return_patch_tokens=True).mean(1).reshape(-1,400);self.q=self.model.head((z-self.model.mean)/self.model.scale).softmax(1)
  e=self.model.encoder;raw=self.x.flatten(0,1);z=e.patch_embed(raw);z=torch.where(self.mask[:,:,None],self.model.mask_token.expand(2,30,-1),z);z=torch.cat([e.cls_token.expand(2,-1,-1),z],1);pos=e.pos_embed[:,self.channels];spatial=pos[:,1:].unsqueeze(2).expand(2,-1,15,-1).flatten(1,2);spatial=torch.cat([pos[:,:1].expand(2,-1,-1),spatial],1);z=z+spatial;z[:,1:]+=e.time_embed[:,:15].unsqueeze(1).expand(2,2,-1,-1).flatten(1,2);z=e.pos_drop(z)
  for block in e.blocks:z=block(z,rel_pos_bias=None)
  z=e.norm(z)[:,1:];logits=self.model.aux_head(z.index_select(1,self.positions).flatten(0,1));targets=self.tokens.flatten(0,1).index_select(1,self.positions).reshape(-1);self.loss=nn.functional.cross_entropy(logits,targets);self.loss.backward();self.norm=nn.utils.clip_grad_norm_(self.params,1.);self.opt.step()
  with torch.no_grad():
   for p,v,radius in zip(self.params,self.initial,self.radii):
    delta=p-v;factor=torch.clamp(radius/torch.linalg.vector_norm(delta).clamp_min(1e-12),max=1.);p.copy_(v+factor*delta)
  self.finite=torch.isfinite(self.loss)&torch.isfinite(self.norm)
 def step(self,x,tokens):
  self.x.copy_(x);self.tokens.copy_(tokens);self.graph.replay();return self.q
def load_model(weight):
 model=JointCarrier("young").cuda().eval();model.load_state_dict(torch.load(ROOT/"phase28"/f"young_joint_{weight:g}"/"model.pt",map_location="cpu",weights_only=False)["state"]);model.requires_grad_(False)
 for module in model.encoder.modules():
  if isinstance(module,nn.LayerNorm):
   for p in module.parameters():p.requires_grad_(True)
 return model
def benchmark():
 x,_=prepared(5);x=x[1280:1344].cuda();tokens=torch.tensor(np.load(ROOT/"phase28/tokens/s05.npz")["tokens"][1280:1344].astype(np.int64),device="cuda");results=[]
 for weight in [0.,.01]:
  for lr in [.001,.01]:
   eager=load_model(weight);params=[p for p in eager.parameters() if p.requires_grad];initial=[p.detach().clone() for p in params];radii=[.1*max(float(v.square().mean().sqrt()),1)*v.numel()**.5 for v in initial];opt=torch.optim.SGD(params,lr=lr);pred=[];start=time.perf_counter()
   for t in range(len(x)):
    with torch.no_grad():pred.append(eager(x[t:t+1]).softmax(1).cpu().numpy()[0])
    opt.zero_grad(set_to_none=True);loss,_=eager.auxiliary(x[t:t+1],tokens[t:t+1],False);loss.backward();nn.utils.clip_grad_norm_(params,1.);opt.step()
    with torch.no_grad():
     for p,v,r in zip(params,initial,radii):
      delta=p-v;factor=torch.clamp(r/torch.linalg.vector_norm(delta).clamp_min(1e-12),max=1.);p.copy_(v+factor*delta)
   torch.cuda.synchronize();eager_seconds=time.perf_counter()-start;reference=np.array(pred);end=[p.detach().clone() for p in params];del eager,opt
   model=load_model(weight);graph=MaskedGraphAdapter(model,lr,x[:1]);pred=[];start=time.perf_counter()
   for t in range(len(x)):pred.append(graph.step(x[t:t+1],tokens[t:t+1]).detach().cpu().numpy()[0].copy());assert bool(graph.finite)
   torch.cuda.synchronize();seconds=time.perf_counter()-start;pred=np.array(pred);err=float(abs(pred-reference).max());perr=max(float(abs(p-v).max()) for p,v in zip(graph.params,end));disagreements=int((pred.argmax(1)!=reference.argmax(1)).sum());assert err<2e-5 and perr<2e-6 and disagreements==0
   result=dict(weight=weight,lr=lr,windows=len(x),max_probability_error=err,max_parameter_error=perr,argmax_disagreements=disagreements,eager_seconds=eager_seconds,graph_seconds=seconds);results.append(result);print("MASKED GRAPH AUDIT",result,flush=True);del graph,model
 (OUT30/"graph_audit.json").write_text(json.dumps(dict(status="PASS",results=results),indent=2))
if __name__=="__main__":benchmark()
