from phase21_full_encoder import *
from contextlib import contextmanager
import pandas as pd
OUT26=ROOT/"phase26"
@contextmanager
def cls_bias(encoder,divisor):
 def hook(module,args,kwargs):
  if divisor!=1:
   z=args[0];bias=torch.zeros(z.shape[1:2]*2,device=z.device);bias[:,0]=-np.log(divisor);kwargs["rel_pos_bias"]=bias
  return args,kwargs
 handles=[b.register_forward_pre_hook(hook,with_kwargs=True) for b in encoder.blocks]
 try:yield
 finally:
  for h in handles:h.remove()
if __name__=="__main__":
 t=vqnsp.vqnsp_encoder_base_decoder_3x200x12(code_dim=64).eval();t.load_state_dict(torch.load(ROOT/"phase14/tokenizer_weights_only.pt",map_location="cpu",weights_only=True),strict=True);t.requires_grad_(False);t.cuda();midpoint_positions(t.encoder);midpoint_positions(t.decoder);state={k:v.clone() for k,v in t.state_dict().items()};rows=[]
 (OUT26/"reconstruction_protocol.json").write_text(json.dumps(dict(subjects=[5,15,25],sampled_epochs=64,labels_read=False,encoder_divisors=[1,8,16],decoder_divisors=[1,8],controls=["constant5288","shuffle8"],target="Official per-half standardized 200-bin FFT magnitude and angle; record separate MSE, same frozen decoder. Zero prediction MSE reference.",scope="Source-only diagnostic. Attention variants alter pretrained model; code diversity alone is not validity. No training or target outcome selection."),indent=2))
 with torch.inference_mode():
  for sid in [5,15,25]:
   raw=np.load(ROOT/("data/processed" if sid<20 else "phase2/processed")/f"s{sid:02d}_x.npy",mmap_mode="r");x=preprocess(np.array(raw[np.linspace(0,len(raw)-1,64,dtype=int)])).cuda();fft=torch.fft.fft(x,dim=-1);amp=t.std_norm(fft.abs()).flatten(1,2);angle=t.std_norm(fft.angle()).flatten(1,2);codes={}
   for divisor in [1,8,16]:
    with cls_bias(t.encoder,divisor):codes[str(divisor)]=torch.cat([tokenizer_indices(t,b) for b in x.split(8)])
   codes["constant5288"]=torch.full_like(codes["1"],5288);rng=torch.Generator(device="cuda");rng.manual_seed(17);codes["shuffle8"]=codes["8"].flatten()[torch.randperm(codes["8"].numel(),generator=rng,device="cuda")].reshape_as(codes["8"])
   for key,ids in codes.items():
    q=t.quantize.embedding.weight[ids].reshape(-1,2,15,64).permute(0,3,1,2).contiguous()
    for divisor in [1,8]:
     rec=[];ph=[]
     with cls_bias(t.decoder,divisor):
      for b in q.split(8):
       rr,pp=t.decode(b,input_chans=[0,127,128]);rec.append(rr);ph.append(pp)
     rr=torch.cat(rec);pp=torch.cat(ph);row=dict(subject=sid,encoder_variant=key,decoder_divisor=divisor,amplitude_mse=float((rr-amp).square().mean()),phase_mse=float((pp-angle).square().mean()),zero_amplitude_mse=float(amp.square().mean()),zero_phase_mse=float(angle.square().mean()));rows.append(row);print(row,flush=True)
 assert all(torch.equal(v,t.state_dict()[k]) for k,v in state.items())
 pd.DataFrame(rows).to_csv(OUT26/"reconstruction.csv",index=False);(OUT26/"reconstruction_audit.json").write_text(json.dumps(dict(status="PASS",cases=len(rows),weights_unchanged=True,labels_read=False),indent=2))
