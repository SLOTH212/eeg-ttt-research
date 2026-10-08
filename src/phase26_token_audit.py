from phase21_full_encoder import *
import pandas as pd
OUT26=ROOT/"phase26"
if __name__=="__main__":
 tokenizer=vqnsp.vqnsp_encoder_base_decoder_3x200x12(code_dim=64).eval();tokenizer.load_state_dict(torch.load(ROOT/"phase14/tokenizer_weights_only.pt",map_location="cpu",weights_only=True),strict=True);tokenizer.requires_grad_(False);tokenizer.cuda();original={k:v.clone() for k,v in tokenizer.state_dict().items()};original_pos=tokenizer.encoder.pos_embed.clone();midpoint_positions(tokenizer.encoder);mid_pos=tokenizer.encoder.pos_embed.clone();raw=np.load(ROOT/"data/processed/s05_x.npy",mmap_mode="r");indices=np.linspace(0,len(raw)-1,64,dtype=int);x=preprocess(np.array(raw[indices])).cuda();tree=ast.parse((ROOT/"phase14/utils.py").read_text());names=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="standard_1020" for t in n.targets))
 def infer(xx,chans):
  ids=[];stds=[]
  with torch.inference_mode():
   for b in xx.split(8):
    z=nn.functional.normalize(tokenizer.encode_task_layer(tokenizer.encoder(b,input_chans=chans,return_patch_tokens=True)),dim=-1);code=tokenizer.quantize.embedding.weight;distance=z.square().sum(-1,keepdim=True)+code.square().sum(1)-2*(z@code.T);ids.append(distance.argmin(-1).cpu().numpy());stds.append(float(z.flatten(0,1).std(0).mean()))
  return np.concatenate(ids),float(np.mean(stds))
 manual,_=infer(x[:8],[0,127,128]);before={k:v.clone() for k,v in tokenizer.state_dict().items()}
 with torch.inference_mode():official=tokenizer.get_codebook_indices(x[:8],input_chans=[0,127,128]).cpu().numpy()
 changed=[k for k,v in tokenizer.state_dict().items() if not torch.equal(v,before[k])];assert np.array_equal(manual,official);assert set(changed)<={"quantize.cluster_size"};tokenizer.load_state_dict(before);rows=[]
 variants=[dict(name=f"midpoint_{length}s",length=length,scale=1.,layout="midpoint") for length in [1,4,5,8,15]]
 variants += [dict(name=f"midpoint_scale{scale}",length=15,scale=scale,layout="midpoint") for scale in [.1,10.,100.]]
 variants += [dict(name=name,length=15,scale=1.,layout=name) for name in ["standardized_half","fpz_pz","cz_oz","zero_spatial","zeros","gaussian"]]
 variants += [dict(name=f"synthetic_repeat{n}_{length}s",length=length,scale=1.,layout="repeat",channels=n) for n in [16,32] for length in [4,8,15]]
 for spec in variants:
  tokenizer.load_state_dict(original);length=spec["length"];xx=x[:,:,:length,:].clone()*spec["scale"];layout=spec["layout"];chans=[0,127,128]
  with torch.no_grad():
   if layout in ["midpoint","standardized_half","zeros","gaussian"]:tokenizer.encoder.pos_embed.copy_(mid_pos)
   if layout=="standardized_half":xx=(xx-xx.mean((1,2,3),keepdim=True))/xx.std((1,2,3),keepdim=True).clamp_min(1e-6)
   elif layout=="fpz_pz":chans=[0,names.index("FPZ")+1,names.index("PZ")+1]
   elif layout=="cz_oz":chans=[0,names.index("CZ")+1,names.index("OZ")+1]
   elif layout=="zero_spatial":tokenizer.encoder.pos_embed.zero_()
   elif layout=="zeros":xx.zero_()
   elif layout=="gaussian":torch.manual_seed(17);xx=torch.randn_like(xx)
   elif layout=="repeat":xx=xx.repeat(1,spec["channels"]//2,1,1);chans=list(range(spec["channels"]+1))
  ids,latent_std=infer(xx,chans);counts=np.bincount(ids.ravel(),minlength=8192);prob=counts[counts>0]/counts.sum();entropy=float(-(prob*np.log2(prob)).sum());row=dict(**spec,tokens=int(counts.sum()),unique_codes=int((counts>0).sum()),dominant_code=int(counts.argmax()),dominant_fraction=float(counts.max()/counts.sum()),entropy_bits=entropy,latent_std=latent_std);rows.append(row);print("TOKEN AUDIT",row,flush=True)
 assert torch.equal(tokenizer.quantize.embedding.weight,original["quantize.embedding.weight"])
 pd.DataFrame(rows).to_csv(OUT26/"token_input_sensitivity.csv",index=False);(OUT26/"token_path_audit.json").write_text(json.dumps(dict(status="PASS",manual_matches_official_exact=True,official_mutated_only_usage_statistic=changed,codebook_initialized=bool(original["quantize.embedding.initted"]),codebook_weight_unchanged=True,task_labels_read=False,subjects=[5],sampled_epochs=64,limitations="Single-source-person unlabelled sensitivity audit. Replicated channels, scaling, zero/noise and position variants are diagnostic perturbations, not valid physiological montage replacements or selected preprocessing."),indent=2))
