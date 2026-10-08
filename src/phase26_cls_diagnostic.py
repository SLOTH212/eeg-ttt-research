from phase21_full_encoder import *
import pandas as pd
OUT26=ROOT/"phase26"
if __name__=="__main__":
 t=vqnsp.vqnsp_encoder_base_decoder_3x200x12(code_dim=64).eval();t.load_state_dict(torch.load(ROOT/"phase14/tokenizer_weights_only.pt",map_location="cpu",weights_only=True),strict=True);t.requires_grad_(False);t.cuda();midpoint_positions(t.encoder);original={k:v.clone() for k,v in t.state_dict().items()};e=t.encoder;rows=[];layers=[];equiv=[]
 (OUT26/"cls_diagnostic_protocol.json").write_text(json.dumps(dict(subjects=[5,15,25],sampled_epochs=32,variants=["official","cls_key_bias_minus_log8","cls_key_bias_minus_log16","remove_cls","duplicated16_same_positions"],scope="Unlabelled diagnostic only. Modified attention is not official tokenizer, not selected production preprocessing. Test duplicated-patch equivalence to downweighting CLS key mass.",labels_read=False),indent=2))
 def manual(x,mode,sid):
  n=x.shape[1];b=len(x);z=e.patch_embed(x);pos=e.pos_embed[:,[127,128]*(n//2)].unsqueeze(2).expand(b,-1,15,-1).flatten(1,2);z=z+pos+e.time_embed[:,:15].unsqueeze(1).expand(b,n,-1,-1).flatten(1,2);hascls=mode!="remove_cls"
  if hascls:z=torch.cat(((e.cls_token+e.pos_embed[:,0:1]).expand(b,-1,-1),z),1)
  z=e.pos_drop(z);bias=None
  if mode.startswith("cls_key_bias"):
   divisor=8 if mode.endswith("8") else 16;bias=torch.zeros((z.shape[1],z.shape[1]),device="cuda");bias[:,0]=-np.log(divisor)
  for i,blk in enumerate(e.blocks):
   att=blk(z,rel_pos_bias=bias,return_attention=True)
   if hascls:layers.append(dict(subject=sid,mode=mode,layer=i+1,patch_to_cls_mean=float(att[:,:,1:,0].mean()),patch_to_cls_max=float(att[:,:,1:,0].max()),patch_feature_std=float(z[:,1:].flatten(0,1).std(0).mean())))
   z=blk(z,rel_pos_bias=bias)
  z=e.norm(z);z=z[:,1:] if hascls else z
  if e.fc_norm is not None:z=e.fc_norm(z)
  z=e.head(z);z=nn.functional.normalize(t.encode_task_layer(z),dim=-1);c=t.quantize.embedding.weight;ids=(z.square().sum(-1,keepdim=True)+c.square().sum(1)-2*z@c.T).argmin(-1)
  return ids,z
 with torch.inference_mode():
  for sid in [5,15,25]:
   raw=np.load(ROOT/("data/processed" if sid<20 else "phase2/processed")/f"s{sid:02d}_x.npy",mmap_mode="r");x=preprocess(np.array(raw[np.linspace(0,len(raw)-1,32,dtype=int)])).cuda();allids={};allz={}
   for mode in ["official","cls_key_bias_minus_log8","cls_key_bias_minus_log16","remove_cls","duplicated16_same_positions"]:
    ids=[];zs=[]
    for b in x.split(8):
     xx=b.repeat(1,8,1,1) if mode=="duplicated16_same_positions" else b
     ii,zz=manual(xx,mode,sid);ids.append(ii.cpu().numpy());zs.append(zz.cpu().numpy())
     if mode=="official":
      expected=nn.functional.normalize(t.encode_task_layer(e(b,input_chans=[0,127,128],return_patch_tokens=True)),dim=-1);assert torch.allclose(zz,expected,atol=1e-6,rtol=1e-5)
    ids=np.concatenate(ids);zs=np.concatenate(zs);allids[mode]=ids;allz[mode]=zs;counts=np.bincount(ids.ravel(),minlength=8192);p=counts[counts>0]/counts.sum();row=dict(subject=sid,mode=mode,unique_codes=int((counts>0).sum()),dominant_fraction=float(counts.max()/counts.sum()),entropy_bits=float(-(p*np.log2(p)).sum()));rows.append(row);print(row,flush=True)
   equiv.append(dict(subject=sid,code_agreement=float(np.mean(allids["duplicated16_same_positions"][:,:30]==allids["cls_key_bias_minus_log8"])),max_latent_difference=float(np.max(np.abs(allz["duplicated16_same_positions"][:,:30]-allz["cls_key_bias_minus_log8"])))))
 assert all(torch.equal(v,t.state_dict()[k]) for k,v in original.items())
 pd.DataFrame(rows).to_csv(OUT26/"cls_diagnostic.csv",index=False);pd.DataFrame(layers).groupby(["subject","mode","layer"]).mean().reset_index().to_csv(OUT26/"cls_attention.csv",index=False);(OUT26/"cls_diagnostic_audit.json").write_text(json.dumps(dict(status="PASS",manual_forward_matches_official=True,weights_unchanged=True,labels_read=False,duplication_equivalence=equiv),indent=2))
