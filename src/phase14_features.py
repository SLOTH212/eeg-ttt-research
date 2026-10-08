from common import *
import sys,ast,time,argparse,hashlib
from scipy.signal import butter,sosfiltfilt,resample_poly
OUT=ROOT/"phase14";sys.path.insert(0,str(OUT))
import modeling_finetune as upstream

def make_encoder(pretrained=True):
 seed_all(17)
 model=upstream.labram_base_patch200_200(num_classes=0,init_values=.1,use_mean_pooling=False).eval()
 if pretrained:
  sd=torch.load(OUT/"weights_only.pt",map_location="cpu",weights_only=True);sd={k[8:]:v for k,v in sd.items() if k.startswith("student.") and k[8:] not in ["mask_token","lm_head.weight","lm_head.bias"]}
  model.load_state_dict(sd,strict=True)
 tree=ast.parse((OUT/"utils.py").read_text());names=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="standard_1020" for t in n.targets))
 with torch.no_grad():
  for dest,(a,b) in zip([127,128],[("FPZ","CZ"),("PZ","OZ")]):model.pos_embed[:,dest].copy_((model.pos_embed[:,names.index(a)+1]+model.pos_embed[:,names.index(b)+1])/2)
 for param in model.parameters():param.requires_grad_(False)
 return model

def encode(m,x,batch=32):
 result=[]
 with torch.inference_mode():
  for i in range(0,len(x),batch):result.append(m.forward_features(torch.from_numpy(np.ascontiguousarray(x[i:i+batch])),input_chans=[0,127,128],return_patch_tokens=True).mean(1).numpy())
 return np.concatenate(result)

def run(sid):
 out=OUT/"features";out.mkdir(exist_ok=True);dest=out/f"s{sid:02d}.npz"
 if dest.exists():return
 data=ROOT/("data/processed" if sid<20 else ("phase2/processed" if sid<40 else "phase6/processed"));raw=np.load(data/f"s{sid:02d}_x.npy",mmap_mode="r");sos=butter(4,[.1,45.],fs=100,btype="bandpass",output="sos");features={};audits={};start=time.perf_counter()
 for label,pretrained in [("pretrained",True),("random",False)]:
  model=make_encoder(pretrained);zs=[]
  for i in range(0,len(raw),64):
   block=np.asarray(raw[i:i+64]);x=resample_poly(sosfiltfilt(sos,block,axis=-1),2,1,axis=-1).astype(np.float32);x=x.reshape(len(x),2,2,15,200).transpose(0,2,1,3,4).reshape(-1,2,15,200);z=encode(model,x);zs.append(z.reshape(len(block),400))
   if i==0:
    single=encode(model,x[:2],batch=1);diff=np.abs(single-z[:2]);relative=float(np.linalg.norm(diff)/max(np.linalg.norm(single),1e-9));assert float(diff.max())<.002 and relative<.0002;altered=x[:32].copy();altered[2:]=altered[2:][::-1]*1.7;check=encode(model,altered)[:2];assert np.array_equal(check,z[:2]);audits[label]=dict(batch1_max_absolute_error=float(diff.max()),batch1_relative_error=relative,other_sample_perturbation_exact=True);assert np.isfinite(z).all()
  features[label]=np.concatenate(zs);assert np.isfinite(features[label]).all()
  del model
 np.savez_compressed(dest,**features)
 (out/f"s{sid:02d}.json").write_text(json.dumps(dict(subject=sid,windows=len(raw),seconds=time.perf_counter()-start,features_shape=list(features["pretrained"].shape),no_labels_read=True,numerical_audit=audits,cpu_ids=sorted(os.sched_getaffinity(0)),checkpoint_sha256=hashlib.sha256((OUT/"weights_only.pt").read_bytes()).hexdigest()),indent=2));print("FEATURES",sid,len(raw),time.perf_counter()-start,flush=True)
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--subject",type=int,required=True);args=parser.parse_args();run(args.subject)
