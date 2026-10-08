from phase70_evaluate import *
OUT71=ROOT/"phase71"
def select_and_evaluate():
 protocol=json.loads((OUT71/"protocol.json").read_text());assert (OUT71/"training_complete.json").exists();selections={}
 for seed in [29,43]:
  candidates=[]
  for lr in [2e-5,.0001]:
   folder=OUT71/f"seed{seed}"/f"pretrained_{lr:g}";d=json.loads((folder/"complete.json").read_text());ck=torch.load(folder/"model.pt",map_location="cpu",weights_only=False);assert d["classifier_seed"]==seed==ck["classifier_seed"]
   best=max([dict(epoch=0,val_f1=d["initial_val_f1"])]+d["history"],key=lambda x:x["val_f1"]);assert best["epoch"]==d["best_epoch"] and best["val_f1"]==d["best_val_f1"]
   order=json.loads((folder/"training_orders.json").read_text());g=torch.Generator().manual_seed(seed);expected=[torch.randperm(d["training_valid_windows"],generator=g).tolist() for _ in range(10)];assert order["seed"]==seed and order["orders"]==expected
   candidates.append(dict(seed=seed,lr=lr,folder=str(folder),best_epoch=d["best_epoch"],source_f1=d["best_val_f1"],checkpoint_sha256=hashlib.sha256((folder/"model.pt").read_bytes()).hexdigest()))
  selections[str(seed)]=dict(selected=max(candidates,key=lambda x:x["source_f1"]),candidates=candidates)
 lock=OUT71/"selection_lock.json";payload=dict(selections=selections,subjects=json.loads((ROOT/"phase70/protocol.json").read_text())["subjects"],source_only_choice=True,locked_before_new_seed_target_evaluation=True)
 if lock.exists():assert json.loads(lock.read_text())==payload
 else:lock.write_text(json.dumps(payload,indent=2))
 for seed in [29,43]:
  out=OUT71/"evaluation"/f"seed{seed}";out.mkdir(parents=True,exist_ok=True);choice=selections[str(seed)]["selected"];m=Carrier("pretrained").cuda().eval();m.load_state_dict(torch.load(Path(choice["folder"])/"model.pt",map_location="cpu",weights_only=False)["state"]);m.requires_grad_(False);state=snapshot(m)
  for sid in payload["subjects"]:
   done=out/f"s{sid:02d}.json"
   if done.exists():continue
   x,y=prepare_target(sid);p=probabilities(m,x);assert all(torch.equal(v.cpu(),state[n]) for n,v in m.state_dict().items())
   with torch.no_grad():
    t=len(x)//2;single=m(x[t:t+1].cuda()).softmax(1).cpu().numpy()[0];error=float(abs(single-p[t]).max());assert error<2e-5 and single.argmax()==p[t].argmax()
   np.savez_compressed(out/f"s{sid:02d}.npz",y=y,p=p);rec=dict(seed=seed,subject=sid,state_unchanged=True,no_online_updates=True,batch_one_error=error,methods={str(a):metrics(y,p if a==1 else ema(p,a)) for a in [1.,.8,.5]});done.write_text(json.dumps(rec,indent=2));print("CBRAMOD SEED TRANSFER",seed,sid,rec["methods"]["1.0"]["macro_f1"],flush=True)
  del m,state
 (OUT71/"evaluation_complete.json").write_text(json.dumps(dict(complete=True,new_people_seed_cases=60,seeds=[29,43])))
if __name__=="__main__":select_and_evaluate()
