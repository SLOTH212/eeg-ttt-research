from phase14_features import *
import modeling_vqnsp as vqnsp
import pickle,copy
OUT=ROOT/"phase16"
def midpoint_positions(model):
 tree=ast.parse((ROOT/"phase14/utils.py").read_text());names=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="standard_1020" for t in n.targets))
 with torch.no_grad():
  for dest,(a,b) in zip([127,128],[("FPZ","CZ"),("PZ","OZ")]):model.pos_embed[:,dest].copy_((model.pos_embed[:,names.index(a)+1]+model.pos_embed[:,names.index(b)+1])/2)
def preprocess(raw):
 sos=butter(4,[.1,45.],fs=100,btype="bandpass",output="sos");x=resample_poly(sosfiltfilt(sos,raw,axis=-1),2,1,axis=-1).astype(np.float32);return torch.from_numpy(np.ascontiguousarray(x.reshape(len(x),2,2,15,200).transpose(0,2,1,3,4).reshape(-1,2,15,200)))
def masked_tokens(model,x,mask,mask_token):
 batch,n,t,_=x.shape;z=model.patch_embed(x)
 if mask is not None:z=torch.where(mask[:,:,None],mask_token.expand(batch,n*t,-1),z)
 z=torch.cat([model.cls_token.expand(batch,-1,-1),z],1);pos=model.pos_embed[:,[0,127,128]];spatial=pos[:,1:,:].unsqueeze(2).expand(batch,-1,t,-1).flatten(1,2);spatial=torch.cat([pos[:,:1,:].expand(batch,-1,-1),spatial],1);z=z+spatial;z[:,1:,:]+=model.time_embed[:,:t,:].unsqueeze(1).expand(batch,n,-1,-1).flatten(1,2);z=model.pos_drop(z)
 for block in model.blocks:z=block(z,rel_pos_bias=None)
 return model.norm(z)[:,1:,:]
def tokenizer_indices(tokenizer,x):
 with torch.no_grad():
  z=tokenizer.encoder(x,input_chans=[0,127,128],return_patch_tokens=True);z=nn.functional.normalize(tokenizer.encode_task_layer(z),dim=-1);code=tokenizer.quantize.embedding.weight;dist=z.square().sum(-1,keepdim=True)+code.square().sum(1)-2*(z@code.T);return dist.argmin(-1)
def run(source):
 out=OUT/source;out.mkdir(exist_ok=True);model=make_encoder(True);state={k:v.clone() for k,v in model.state_dict().items()};sd=torch.load(ROOT/"phase14/weights_only.pt",map_location="cpu",weights_only=True);mask_token=sd["student.mask_token"];lmw=sd["student.lm_head.weight"];lmb=sd["student.lm_head.bias"]
 for module in model.modules():
  if isinstance(module,nn.LayerNorm):
   for param in module.parameters():param.requires_grad_(True)
 params=[p for p in model.parameters() if p.requires_grad];initial=[p.detach().clone() for p in params];writable=[n for n,p in model.named_parameters() if p.requires_grad];tokenizer=vqnsp.vqnsp_encoder_base_decoder_3x200x12(code_dim=64).eval();tokenizer.load_state_dict(torch.load(ROOT/"phase14/tokenizer_weights_only.pt",map_location="cpu",weights_only=True),strict=True);tokenizer.requires_grad_(False);midpoint_positions(tokenizer.encoder);tokenizer_state={k:v.clone() for k,v in tokenizer.state_dict().items()}
 with (ROOT/"phase14/probes"/f"{source}_pretrained400/model.pkl").open("rb") as f:carrier=pickle.load(f)
 scale=torch.tensor(carrier["scaler"].scale_,dtype=torch.float32);mean=torch.tensor(carrier["scaler"].mean_,dtype=torch.float32);W=torch.tensor(carrier["model"].coef_,dtype=torch.float32);b=torch.tensor(carrier["model"].intercept_,dtype=torch.float32)
 def logits(x):
  z=model.forward_features(x,input_chans=[0,127,128],return_patch_tokens=True).mean(1).reshape(-1,400);return nn.functional.linear((z-mean)/scale,W,b)
 def restore():
  with torch.no_grad():
   for p,v in zip(params,initial):p.copy_(v)
 rows=[];audits=[];start=time.perf_counter()
 for sid in [7,17,27,37]:
  dest=out/f"s{sid:02d}.json"
  if dest.exists():rows.extend(json.loads(dest.read_text())["rows"]);continue
  data=ROOT/("data/processed" if sid<20 else "phase2/processed");raw=np.load(data/f"s{sid:02d}_x.npy",mmap_mode="r");y=np.load(data/f"s{sid:02d}_meta.npz")["y"];times=np.linspace(0,len(raw)-5,64,dtype=int);subject_rows=[]
  for t in times:
   restore();x=preprocess(np.array(raw[t:t+5]));current=x[:2];future=x[2:];yy=torch.tensor(y[t+1:t+5],dtype=torch.long);valid=yy>=0
   if not bool(valid.any()):continue
   with torch.no_grad():
    expected=model.forward_features(current,input_chans=[0,127,128],return_patch_tokens=True);manual=masked_tokens(model,current,None,mask_token);error=float((manual-expected).abs().max());assert error==0.;base_logits=logits(future);base_loss=nn.functional.cross_entropy(base_logits[valid],yy[valid]);targets=tokenizer_indices(tokenizer,current)
   future_loss=nn.functional.cross_entropy(logits(future)[valid],yy[valid]);task_grads=torch.autograd.grad(future_loss,params);task_norm=torch.sqrt(sum(g.square().sum() for g in task_grads));rng=np.random.default_rng(100000*sid+int(t));mask=torch.zeros(2,30,dtype=torch.bool)
   for j in range(2):mask[j,rng.permutation(30)[:15]]=True
   for objective in ["entropy","masked_token"]:
    restore()
    if objective=="entropy":
     lp=logits(current).log_softmax(1);loss=-(lp.exp()*lp).sum(1).mean()
    else:
     losses=[]
     for mk in [mask,~mask]:
      tokens=masked_tokens(model,current,mk,mask_token);pred=nn.functional.linear(tokens[mk],lmw,lmb);losses.append(nn.functional.cross_entropy(pred,targets[mk]))
     loss=(losses[0]+losses[1])/2
    grads=torch.autograd.grad(loss,params);norm=torch.sqrt(sum(g.square().sum() for g in grads));clip=min(1.,1/max(float(norm),1e-12));dot=float(sum((g*h).sum() for g,h in zip(grads,task_grads)));cosine=dot/max(float(norm*task_norm),1e-12)
    for lr in [.0001,.001,.01,.1]:
     restore()
     with torch.no_grad():
      for p,v,g in zip(params,initial,grads):
       p.copy_(v-lr*clip*g);delta=p-v;limit=.1*max(float(v.square().mean().sqrt()),1.)*np.sqrt(v.numel());d=float(delta.norm())
       if d>limit:p.copy_(v+delta*(limit/d))
      updated=logits(future);new_loss=nn.functional.cross_entropy(updated[valid],yy[valid]);gain=float(base_loss-new_loss)
     subject_rows.append(dict(subject=sid,t=int(t),source=source,objective=objective,lr=lr,aux_loss=float(loss.detach()),gradient_norm=float(norm),gradient_cosine=cosine,first_order_gain=lr*clip*dot,future_ce_gain=gain,base_ce=float(base_loss),future_label_count=int(valid.sum()),base_correct=int((base_logits[valid].argmax(1)==yy[valid]).sum()),write_correct=int((updated[valid].argmax(1)==yy[valid]).sum())))
   restore()
  assert all(torch.equal(state[k],v) for k,v in model.state_dict().items());assert all(torch.equal(tokenizer_state[k],v) for k,v in tokenizer.state_dict().items());dest.write_text(json.dumps(dict(subject=sid,rows=subject_rows,model_restored_exact=True,tokenizer_unchanged_exact=True,unmasked_forward_matches_exact=True),indent=2));rows.extend(subject_rows);print("MASKED PULSES",source,sid,len(subject_rows),time.perf_counter()-start,flush=True)
 (out/"complete.json").write_text(json.dumps(dict(source=source,pulse_rows=len(rows),seconds=time.perf_counter()-start,writable_parameters=sum(p.numel() for p in params),writable_names=writable,source_only=True),indent=2));print("MASKED DIAGNOSTIC COMPLETE",source,flush=True)
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--source",required=True);a=parser.parse_args();run(a.source)
