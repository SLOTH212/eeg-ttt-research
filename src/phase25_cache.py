from phase22_source_screen import *
OUT25=ROOT/"phase25"
def build_cache(source,sid):
 dest=OUT25/"cache"/f"{source}_s{sid:02d}.npz"
 if dest.exists():return
 model=FullCarrier(source,"pretrained").eval().cuda();model.load_state_dict(torch.load(source_checkpoint(source),map_location="cpu",weights_only=False)["state"]);x=np.load(ROOT/"phase17/cache"/f"s{sid:02d}_x.npy",mmap_mode="r");capture={};hook=model.head.register_forward_pre_hook(lambda module,args:capture.update(z=args[0].detach()));ps=[];zs=[]
 with torch.inference_mode():
  for i in range(0,len(x),128):
   ps.append(model(torch.tensor(np.array(x[i:i+128]),device="cuda")).softmax(1).cpu().numpy());zs.append(capture["z"].cpu().numpy())
 hook.remove();prob=np.concatenate(ps);embedding=np.concatenate(zs);assert np.isfinite(prob).all() and np.isfinite(embedding).all();tmp=dest.with_suffix(".tmp.npz");np.savez_compressed(tmp,probabilities=prob,embeddings=embedding);tmp.replace(dest);print("SOURCE OBSERVATION CACHE",source,sid,flush=True)
if __name__=="__main__":
 for source in ["young","mixed"]:
  for sid in [5,15,25,35]:build_cache(source,sid)
