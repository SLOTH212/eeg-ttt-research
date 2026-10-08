from phase22_source_screen import *
import hashlib
for source in ["young","mixed"]:
 try:checkpoint=source_checkpoint(source)
 except AssertionError:continue
 dest=OUT22/f"frozen_audit_{source}.json"
 if dest.exists():continue
 model=FullCarrier(source,"pretrained").eval().cuda();model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=False)["state"]);rows=[]
 for sid in [5,15,25,35]:
  x,y=prepared(sid);batch=probabilities(model,x);out=[];start=time.perf_counter()
  with torch.inference_mode():
   for i in range(len(x)):out.append(model(x[i:i+1].cuda()).softmax(1).cpu().numpy()[0])
   changed=x[:16].clone();changed[2:]*=1.7;pchanged=model(changed.cuda()).softmax(1).cpu().numpy()[:2];original=model(x[:16].cuda()).softmax(1).cpu().numpy()[:2]
  single=np.array(out);err=float(np.max(np.abs(single-batch)));disagreements=int((single.argmax(1)!=batch.argmax(1)).sum());assert err<.002;assert np.array_equal(pchanged,original)
  rows.append(dict(subject=sid,windows=len(y),max_probability_error=err,argmax_disagreements=disagreements,batch_f1=metrics(y,batch)["macro_f1"],single_f1=metrics(y,single)["macro_f1"],future_sample_perturbation_exact=True,seconds=time.perf_counter()-start));print(source,rows[-1],flush=True)
 dest.write_text(json.dumps(dict(source=source,checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),rows=rows),indent=2))
