from phase21_full_encoder import *
OUT26=ROOT/"phase26"
if __name__=="__main__":
 tokenizer=vqnsp.vqnsp_encoder_base_decoder_3x200x12(code_dim=64).eval();tokenizer.load_state_dict(torch.load(ROOT/"phase14/tokenizer_weights_only.pt",map_location="cpu",weights_only=True),strict=True);tokenizer.requires_grad_(False);midpoint_positions(tokenizer.encoder);tokenizer.cuda();state={k:v.clone() for k,v in tokenizer.state_dict().items()};splits=json.loads((ROOT/"phase3/protocol.json").read_text())["folds"];subjects=sorted(set([7,17,27,37]+[sid for source in ["young","mixed"] for key in ["base_train","base_val"] for sid in splits[source][key]]))
 for sid in subjects:
  dest=OUT26/"tokens"/f"s{sid:02d}.npz"
  if dest.exists():continue
  root=ROOT/("data/processed" if sid<20 else "phase2/processed");raw=np.load(root/f"s{sid:02d}_x.npy",mmap_mode="r");parts=[];start=time.perf_counter()
  for i in range(0,len(raw),32):
   x=preprocess(np.array(raw[i:i+32])).cuda();parts.append(tokenizer_indices(tokenizer,x).cpu().numpy().reshape(-1,2,30).astype(np.uint16))
  tokens=np.concatenate(parts);assert tokens.max()<8192;old=ROOT/"phase17/cache"/f"s{sid:02d}_tokens.npy";agreement=float(np.mean(tokens==np.load(old))) if old.exists() else None
  if agreement is not None:assert agreement>.999
  assert all(torch.equal(v,tokenizer.state_dict()[k]) for k,v in state.items());counts=np.bincount(tokens.ravel(),minlength=8192);tmp=dest.with_suffix(".tmp.npz");np.savez_compressed(tmp,tokens=tokens);tmp.replace(dest);meta=dict(subject=sid,windows=len(raw),unique_codes=int((counts>0).sum()),dominant_code_fraction=float(counts.max()/counts.sum()),cpu_reference_agreement=agreement,teacher_state_unchanged=True,labels_read=False,seconds=time.perf_counter()-start);dest.with_suffix(".json").write_text(json.dumps(meta,indent=2));print("SOURCE AUX TOKENS",meta,flush=True)
