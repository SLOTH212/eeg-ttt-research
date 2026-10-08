from phase33_train import *
import pandas as pd,hashlib,warnings
warnings.filterwarnings("ignore",message="y_pred contains classes not in y_true")
def evaluate(source,weight,seed):
 folder=OUT26/f"{source}_joint_{weight:g}_seed{seed}"
 if not (folder/"complete.json").exists() or (folder/"validation_audit.json").exists():return
 meta=json.loads((folder/"complete.json").read_text());ck=torch.load(folder/"model.pt",map_location="cpu",weights_only=False);assert meta["best_val_f1"]==max([meta["initial_val_f1"]]+[h["val_f1"] for h in meta["history"]]);model=JointCarrier(source).cuda().eval();model.load_state_dict(ck["state"]);rows=[]
 for sid in [5,15,25,35]:
  x,y=prepared(sid);tokens=torch.tensor(np.load(ROOT/"phase28/tokens"/f"s{sid:02d}.npz")["tokens"].astype(np.int64));pr=probabilities(model,x);mm=metrics(y,pr);file=folder/f"s{sid:02d}.npz";np.savez_compressed(file,y=y,p=pr);idx=np.linspace(0,len(y)-1,min(256,len(y)),dtype=int);shuffled=np.roll(idx,len(idx)//2);losses=[];shuffle_losses=[];accs=[]
  with torch.inference_mode():
   for positions in np.array_split(np.arange(len(idx)),8):
    loss,acc=model.auxiliary(x[idx[positions]].cuda(),tokens[idx[positions]].cuda(),False);shuffled_loss,_=model.auxiliary(x[shuffled[positions]].cuda(),tokens[idx[positions]].cuda(),False);losses.append(float(loss));accs.append(float(acc));shuffle_losses.append(float(shuffled_loss))
  aux=float(np.mean(losses));assert abs(aux-meta["selected_auxiliary"][str(sid)]["masked_ce"])<1e-4
  rows.append(dict(source=source,weight=weight,training_seed=seed,subject=sid,f1=mm["macro_f1"],ce=mm["ce"],masked_ce=aux,masked_accuracy=float(np.mean(accs)),input_shuffled_masked_ce=float(np.mean(shuffle_losses))))
  print("JOINT VALIDATION",rows[-1],flush=True)
 score=float(np.mean([r["f1"] for r in rows]));assert abs(score-meta["best_val_f1"])<1e-10
 (folder/"validation_audit.json").write_text(json.dumps(dict(status="PASS",source=source,weight=weight,training_seed=seed,results=rows,classification_and_auxiliary_metrics_recomputed=True,selected_checkpoint_sha256=hashlib.sha256((folder/"model.pt").read_bytes()).hexdigest(),source_only=True),indent=2));del model
if __name__=="__main__":
 while True:
  for seed in [29,43]:
   for weight in [0.,.01]:evaluate("young",weight,seed)
  if len(list(OUT26.glob("*_joint_*/validation_audit.json")))==4:break
  time.sleep(20)
