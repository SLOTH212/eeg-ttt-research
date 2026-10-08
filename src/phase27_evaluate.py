from phase27_train import *
import hashlib

def evaluate(source,kind,seed):
 protocol=json.loads((OUT/"protocol.json").read_text());dirs=[OUT/f"{source}_{kind}_{lr:g}_seed{seed}" for lr in protocol["encoder_lrs"]]
 if not all((d/"complete.json").exists() for d in dirs):return
 out=OUT/"evaluation"/f"{source}_{kind}_seed{seed}";out.mkdir(parents=True,exist_ok=True)
 if (out/"evaluation.json").exists():return
 candidates=[json.loads((d/"complete.json").read_text()) for d in dirs];i=max(range(len(candidates)),key=lambda j:candidates[j]["best_val_f1"]);selected=dirs[i]
 # Persist source-only choice before reading target labels or producing target predictions.
 selection=dict(source=source,kind=kind,seed=seed,checkpoint_sha256=hashlib.sha256((selected/"model.pt").read_bytes()).hexdigest(),selected_directory=selected.name,selection=candidates[i],candidates=candidates,rule="Maximum mean source-validation F1 with epoch0 fallback; ties use earlier grid entry",exploratory=True)
 (out/"selection.json").write_text(json.dumps(selection,indent=2))
 model=FullCarrier(source,kind).eval().cuda();ck=torch.load(selected/"model.pt",map_location="cpu",weights_only=False);model.load_state_dict(ck["state"]);split=json.loads((ROOT/"phase3/protocol.json").read_text())["folds"][source];rows=[]
 for sid in split["evaluation"]+list(range(40,59)):
  file=out/f"s{sid:02d}.npz"
  if file.exists():
   with np.load(file) as saved:y=saved["y"];pr=saved["source_tuned"]
  else:
   x,y=prepared(sid);pr=probabilities(model,x);del x;tmp=file.with_suffix(".tmp.npz");np.savez_compressed(tmp,y=y,source_tuned=pr);tmp.replace(file)
  assert np.isfinite(pr).all() and pr.shape==(len(y),5)
  rows.append(dict(subject=sid,cohort="older" if sid>=40 else "middle_dev" if sid>=20 else "young_dev",method="full_source_tuned",**metrics(y,pr)))
  print("FULL EVAL",source,kind,sid,rows[-1]["macro_f1"],flush=True)
 (out/"evaluation.json").write_text(json.dumps(dict(source=source,kind=kind,results=rows,selection=selection,exploratory=True),indent=2))

if __name__=="__main__":
 while True:
  for seed in [29,43]:
   for source in ["young","mixed"]:evaluate(source,"pretrained",seed)
  if len(list((OUT/"evaluation").glob("*/evaluation.json")))==4:break
  time.sleep(20)
