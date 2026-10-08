from pathlib import Path
import os,sys,json,time
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
import numpy as np,pandas as pd,torch
from scipy.signal import butter,sosfiltfilt,resample_poly
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"phase62"
sys.path.insert(0,str(OUT/"upstream"))
from models.cbramod import CBraMod
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
assert os.sched_getaffinity(0)<=set(range(8,24))
def prep(raw,band):
 sos=butter(4,band,fs=100,btype="bandpass",output="sos")
 return resample_poly(sosfiltfilt(sos,raw,axis=-1),2,1,axis=-1).astype(np.float32).reshape(-1,2,30,200)
def run():
 assert json.loads((OUT/"checkpoint_audit.json").read_text())["status"]=="PASS"
 model=CBraMod().cuda().eval();state=torch.load(OUT/"pretrained_weights.pth",map_location="cpu",weights_only=True);model.load_state_dict(state,strict=True);model.requires_grad_(False)
 (OUT/"cases").mkdir(exist_ok=True);rows=[];audits=[]
 def predict(x,mask=None):
  out=[]
  with torch.no_grad():
   for start in range(0,len(x),8):
    xt=torch.tensor(x[start:start+8],device="cuda");mt=None if mask is None else torch.tensor(mask[start:start+8],device="cuda")
    out.append(model(xt,mt).cpu().numpy())
  return np.concatenate(out)
 for sid in [5,15,25,35]:
  folder=ROOT/("data/processed" if sid<20 else "phase2/processed");raw=np.load(folder/f"s{sid:02d}_x.npy",mmap_mode="r")
  times=np.linspace(0,len(raw)-1,96,dtype=int);sample=np.array(raw[times]);rng=torch.Generator().manual_seed(62000+sid)
  masks={"bernoulli":(torch.rand((96,2,30),generator=rng)<.5).numpy(),"checkerboard":np.broadcast_to((np.arange(2)[:,None]+np.arange(30)[None,:])%2==0,(96,2,30)).copy()}
  for prepname,band in [("existing",[.1,45]),("sleep_band",[.3,35])]:
   x=prep(sample,band);single=prep(sample[:1],band);assert np.array_equal(single,x[:1]);unmasked=predict(x);unmasked_mse=np.square(unmasked.astype(np.float64)-x).mean(axis=(1,2,3));np.savez_compressed(OUT/"cases"/f"s{sid:02d}_{prepname}_input.npz",times=times,x=x,unmasked_mse=unmasked_mse)
   for maskname,mask in masks.items():
    assert mask.reshape(96,-1).sum(1).min()>0 and (~mask).sum(2).min()>0
    pred=predict(x,mask);shuffle=predict(np.roll(x,48,axis=0),mask);visible=(x*(~mask)[...,None]).sum(2)/(~mask).sum(2)[...,None];template=np.broadcast_to(visible[:,:,None,:],x.shape)
    values=[]
    for i in range(len(x)):
     y=x[i][mask[i]].astype(np.float64);p=pred[i][mask[i]].astype(np.float64);s=shuffle[i][mask[i]].astype(np.float64);v=template[i][mask[i]].astype(np.float64)
     values.append([np.square(p-y).mean(),np.square(y).mean(),np.square(s-y).mean(),np.square(v-y).mean(),np.square(p).mean()])
    values=np.array(values);assert np.isfinite(values).all()
    # Inputs hidden by the explicit raw patch mask must never affect the output.
    xt=torch.tensor(x[:2],device="cuda");mt=torch.tensor(mask[:2],device="cuda")
    with torch.no_grad():
     baseline=model(xt,mt).clone();changed=xt.clone();changed[mt]+=3.17;leak=float(abs(model(changed,mt)-baseline).max())
     changed=xt.clone();changed[1]=changed[1]*1.7+.3;other=float(abs(model(changed,mt)[0]-baseline[0]).max())
    assert leak==0 and other<2e-5
    np.savez_compressed(OUT/"cases"/f"s{sid:02d}_{prepname}_{maskname}.npz",times=times,mask=mask,masked_target=x[mask],masked_prediction=pred[mask],masked_shuffle_prediction=shuffle[mask],window_metrics=values)
    mean=values.mean(0);rec=dict(subject=sid,preprocessing=prepname,mask=maskname,windows=len(x),masked_mse=mean[0],zero_mse=mean[1],shuffled_mse=mean[2],visible_template_mse=mean[3],predicted_energy=mean[4],ratio_to_zero=mean[0]/mean[1],ratio_to_shuffle=mean[0]/mean[2],unmasked_mse=float(unmasked_mse.mean()),validity_pass=bool(mean[0]<mean[1] and mean[0]<mean[2]))
    rows.append(rec);audits.append(dict(subject=sid,preprocessing=prepname,mask=maskname,masked_input_perturbation_error=leak,other_batch_sample_error=other,single_window_preprocess_exact=True));print("CBRAMOD RECON",rec,flush=True)
 assert all(torch.equal(v.cpu(),state[n]) for n,v in model.state_dict().items())
 pd.DataFrame(rows).to_csv(OUT/"reconstruction.csv",index=False)
 (OUT/"execution_audit.json").write_text(json.dumps(dict(status="PASS",cases=audits,state_unchanged_exact=True,classification_labels_not_read=True,native_channels=2,no_auxiliary_fitting=True),indent=2))
if __name__=="__main__":run()
