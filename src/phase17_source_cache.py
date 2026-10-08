from phase16_masked_pulses import *
OUT=ROOT/"phase17";cache=OUT/"cache";cache.mkdir(exist_ok=True)
tokenizer=vqnsp.vqnsp_encoder_base_decoder_3x200x12(code_dim=64).eval();tokenizer.load_state_dict(torch.load(ROOT/"phase14/tokenizer_weights_only.pt",map_location="cpu",weights_only=True),strict=True);tokenizer.requires_grad_(False);midpoint_positions(tokenizer.encoder);state={k:v.clone() for k,v in tokenizer.state_dict().items()}
for sid in [5,15,25,35]:
 done=cache/f"s{sid:02d}.json"
 if done.exists():continue
 root=ROOT/("data/processed" if sid<20 else "phase2/processed");raw=np.load(root/f"s{sid:02d}_x.npy",mmap_mode="r");dest=cache/f"s{sid:02d}_x.npy";x=np.lib.format.open_memmap(dest,mode="w+",dtype=np.float32,shape=(len(raw),2,2,15,200));tokens=np.empty((len(raw),2,30),np.uint16);start=time.perf_counter()
 for i in range(0,len(raw),32):
  batch=preprocess(np.array(raw[i:i+32]));x[i:i+32]=batch.numpy().reshape(-1,2,2,15,200);tokens[i:i+32]=tokenizer_indices(tokenizer,batch).numpy().reshape(-1,2,30)
 x.flush();np.save(cache/f"s{sid:02d}_tokens.npy",tokens);assert all(torch.equal(v,tokenizer.state_dict()[k]) for k,v in state.items());done.write_text(json.dumps(dict(subject=sid,windows=len(raw),labels_read=False,tokenizer_state_unchanged=True,seconds=time.perf_counter()-start),indent=2));print("SOURCE CACHE",sid,time.perf_counter()-start,flush=True)
