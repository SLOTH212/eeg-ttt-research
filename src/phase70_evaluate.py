from phase66_train import *
OUT70=ROOT/"phase70"
def prepare_target(sid):
 folder=ROOT/("data/processed" if sid<20 else "phase2/processed" if sid<40 else "phase6/processed");raw=np.load(folder/f"s{sid:02d}_x.npy",mmap_mode="r");parts=[]
 for begin in range(0,len(raw),128):parts.append(prep(np.array(raw[begin:begin+128]),[.1,45]))
 return torch.from_numpy(np.concatenate(parts)),np.load(folder/f"s{sid:02d}_meta.npz")["y"]
def run():
 protocol=json.loads((OUT70/"protocol.json").read_text());assert hashlib.sha256((OUT66/"selection.json").read_bytes()).hexdigest()==protocol["selection_file_sha256"]
 models={};states={}
 for kind,choice in protocol["selection"].items():
  path=Path(choice["folder"])/"model.pt";assert hashlib.sha256(path.read_bytes()).hexdigest()==choice["checkpoint_sha256"]
  m=Carrier(kind).cuda().eval();m.load_state_dict(torch.load(path,map_location="cpu",weights_only=False)["state"]);m.requires_grad_(False);models[kind]=m;states[kind]=snapshot(m)
 for sid in protocol["subjects"]:
  done=OUT70/f"s{sid:02d}.json"
  if done.exists():continue
  x,y=prepare_target(sid);record=dict(subject=sid,cohort="young" if sid<20 else "middle" if sid<40 else "older",methods={},state_unchanged=True,no_online_updates=True);preds={}
  for kind,m in models.items():
   p=probabilities(m,x);preds[kind]=p
   with torch.no_grad():
    t=len(x)//2;single=m(x[t:t+1].cuda()).softmax(1).cpu().numpy()[0];err=float(abs(single-p[t]).max());assert err<2e-5 and single.argmax()==p[t].argmax()
   assert all(torch.equal(v.cpu(),states[kind][n]) for n,v in m.state_dict().items())
   record[kind+"_batch_one_error"]=err
   for smooth in [1.,.8,.5]:record["methods"][kind+f"_ema{smooth:g}"]=metrics(y,p if smooth==1 else ema(p,smooth))
  np.savez_compressed(OUT70/f"s{sid:02d}.npz",y=y,**preds);done.write_text(json.dumps(record,indent=2));print("CBRAMOD STATIC TRANSFER",sid,{k:v["macro_f1"] for k,v in record["methods"].items()},flush=True)
 (OUT70/"complete.json").write_text(json.dumps(dict(complete=True,subjects=len(protocol["subjects"]),classifiers=2,no_online_updates=True)))
if __name__=="__main__":run()
