from phase21_full_encoder import *
import pandas as pd
from scipy.signal import butter,sosfiltfilt
OUT26=ROOT/"phase26"
if __name__=="__main__":
 tokenizer=vqnsp.vqnsp_encoder_base_decoder_3x200x12(code_dim=64).eval();tokenizer.load_state_dict(torch.load(ROOT/"phase14/tokenizer_weights_only.pt",map_location="cpu",weights_only=True),strict=True);tokenizer.requires_grad_(False);tokenizer.cuda();original={k:v.clone() for k,v in tokenizer.state_dict().items()};midpoint_positions(tokenizer.encoder);mid=tokenizer.encoder.pos_embed.clone();rows=[]
 def infer(xx,chans):
  ids=[]
  with torch.inference_mode():
   for b in xx.split(8):
    z=nn.functional.normalize(tokenizer.encode_task_layer(tokenizer.encoder(b,input_chans=chans,return_patch_tokens=True)),dim=-1);code=tokenizer.quantize.embedding.weight;distance=z.square().sum(-1,keepdim=True)+code.square().sum(1)-2*(z@code.T);ids.append(distance.argmin(-1).cpu().numpy())
  return np.concatenate(ids)
 protocol=dict(subjects=[5,15,25],sampled_epochs=32,seed=17,labels_read=False,purpose="Separate input bandwidth, token count and artificial spatial diversity; all synthetic conditions diagnostic only",variants="2/16/32 channels, real/zero signal and duplicated/distinct positions; two-channel matched-RMS Gaussian with lowpass10/20/30/45/60/75Hz or white; no production preprocessing selected")
 (OUT26/"token_controls_protocol.json").write_text(json.dumps(protocol,indent=2))
 for sid in protocol["subjects"]:
  folder=ROOT/("data/processed" if sid<20 else "phase2/processed");raw=np.load(folder/f"s{sid:02d}_x.npy",mmap_mode="r");idx=np.linspace(0,len(raw)-1,32,dtype=int);x=preprocess(np.array(raw[idx])).cuda();specs=[]
  for n in [2,16,32]:
   for spatial in ["duplicated","distinct"]:
    for signal in ["real","zero"]:specs.append(dict(channels=n,spatial=spatial,signal=signal,cutoff=0))
  for cutoff in [10,20,30,45,60,75,100]:specs.append(dict(channels=2,spatial="duplicated",signal="gaussian",cutoff=cutoff))
  for spec in specs:
   n=spec["channels"];xx=x.repeat(1,n//2,1,1);tokenizer.load_state_dict(original)
   if spec["spatial"]=="duplicated":
    with torch.no_grad():tokenizer.encoder.pos_embed.copy_(mid)
    chans=[0]+[127,128]*(n//2)
   else:chans=list(range(n+1))
   if spec["signal"]=="zero":xx=torch.zeros_like(xx)
   elif spec["signal"]=="gaussian":
    noise=np.random.default_rng(17).normal(size=(len(x),2,3000))
    if spec["cutoff"]<100:noise=sosfiltfilt(butter(4,spec["cutoff"],fs=200,output="sos"),noise,axis=-1)
    noise-=noise.mean(-1,keepdims=True);noise/=noise.std(-1,keepdims=True);rms=x.flatten(2).std(-1,keepdim=True).cpu().numpy();xx=torch.tensor((noise*rms).reshape(-1,2,15,200),dtype=torch.float32,device="cuda")
   ids=infer(xx,chans);counts=np.bincount(ids.ravel(),minlength=8192);p=counts[counts>0]/counts.sum();row=dict(subject=sid,**spec,unique_codes=int((counts>0).sum()),dominant_code=int(counts.argmax()),dominant_fraction=float(counts.max()/counts.sum()),entropy_bits=float(-(p*np.log2(p)).sum()));rows.append(row);print(row,flush=True)
   pd.DataFrame(rows).to_csv(OUT26/"token_controls.csv",index=False)
 assert torch.equal(tokenizer.quantize.embedding.weight,original["quantize.embedding.weight"])
 (OUT26/"token_controls_complete.json").write_text(json.dumps(dict(status="PASS",conditions=len(rows),labels_read=False,codebook_unchanged=True,cpu_affinity=sorted(os.sched_getaffinity(0))),indent=2))
