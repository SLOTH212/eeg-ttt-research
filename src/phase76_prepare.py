from pathlib import Path
import json,hashlib,numpy as np,pyedflib,pandas as pd
from scipy.signal import resample_poly,welch
ROOT=Path(__file__).resolve().parents[1];OUT76=ROOT/"phase76"
def header(path):
 with path.open("rb") as f:
  fixed=f.read(256);ns=int(fixed[252:256]);b=f.read(ns*256);offset=0;d={}
  for name,width in [("labels",16),("transducer",80),("units",8),("physical_min",8),("physical_max",8),("digital_min",8),("digital_max",8),("prefilter",80),("samples_per_record",8),("reserved",32)]:
   d[name]=[b[offset+i*width:offset+(i+1)*width].decode("latin1").strip() for i in range(ns)];offset+=ns*width
 d["header_bytes"]=int(fixed[184:192]);d["records"]=int(fixed[236:244]);d["record_seconds"]=float(fixed[244:252]);return d
def direct_samples(path,h,index,n=1000):
 ns=np.array(h["samples_per_record"],int);count=int(ns[index]);out=[]
 with path.open("rb") as f:
  for record in range(int(np.ceil(n/count))):
   f.seek(h["header_bytes"]+record*int(ns.sum())*2+int(ns[:index].sum())*2);out.extend(np.frombuffer(f.read(count*2),dtype="<i2"))
 digital=np.array(out[:n],float);lo=float(h["physical_min"][index]);hi=float(h["physical_max"][index]);dl=float(h["digital_min"][index]);dh=float(h["digital_max"][index]);return (digital-dl)*(hi-lo)/(dh-dl)+lo
def transform(windows):
 assert windows.shape[1:]==(6,6000)
 bipolar=np.stack([windows[:,0]-windows[:,1],windows[:,1]-windows[:,2],windows[:,3]-windows[:,4],windows[:,4]-windows[:,5]],axis=1)
 x=resample_poly(bipolar,1,2,axis=-1).astype(np.float32);x-=x.mean(-1,keepdims=True);return np.clip(x/100.,-20,20)
def features(x):
 f,power=welch(x,fs=100,nperseg=400,noverlap=200,axis=-1);edges=[.5,2,4,6,8,10,12,14,16,20,25,30,40];bands=np.stack([power[:,:,(f>=a)&(f<b)].sum(-1) for a,b in zip(edges[:-1],edges[1:])],axis=-1);logpower=np.log(bands+1e-10);rel=np.log((bands+1e-10)/(bands.sum(-1,keepdims=True)+1e-10));sd=x.std(-1);diff=np.diff(x,axis=-1).std(-1)/(sd+1e-6);return np.concatenate([logpower.reshape(len(x),-1),rel.reshape(len(x),-1),np.log(sd+1e-6),diff],axis=1).astype(np.float32)
def run():
 assert json.loads((OUT76/"download_audit.json").read_text())["status"]=="PASS";protocol=json.loads((OUT76/"protocol.json").read_text());channels=protocol["native_channels"];out=OUT76/"processed";out.mkdir(exist_ok=True);records=[]
 for sid in protocol["participants"]:
  folder=OUT76/"raw"/sid/"eeg";path=folder/(sid+"_task-sleep_eeg.edf");h=header(path);idx=[h["labels"].index(c) for c in channels];units=[h["units"][i] for i in idx];assert all(v in ["uV","µV"] for v in units)
  sf=[int(h["samples_per_record"][i])/h["record_seconds"] for i in idx];assert sf==[200.]*6
  with pyedflib.EdfReader(str(path),annotations_mode=0) as f:
   assert all(f.getSampleFrequency(i)==200 for i in idx);ns=f.getNSamples();assert len(set(int(ns[i]) for i in idx))==1;n=int(ns[idx[0]])//6000;leftover=int(ns[idx[0]])%6000;parts=[];directerr=0.
   for ii in idx:directerr=max(directerr,float(abs(f.readSignal(ii,start=0,n=1000)-direct_samples(path,h,ii)).max()))
   assert directerr<1e-8
   for first in range(0,n,64):
    count=min(64,n-first);raw=np.stack([f.readSignal(i,start=first*6000,n=count*6000) for i in idx]);windows=raw.reshape(6,count,6000).transpose(1,0,2);x=transform(windows);parts.append(x)
    if first==0:
     prefix=transform(windows[:1]);assert np.array_equal(prefix,x[:1]);changed=windows[:2].copy();changed[1]+=1000;assert np.array_equal(transform(changed)[:1],prefix);prefix_exact=True
   x=np.concatenate(parts);assert np.isfinite(x).all()
  for montage,sl in [("left",slice(0,2)),("right",slice(2,4))]:
   xx=x[:,sl].copy();ff=features(xx);np.save(out/(sid+"_"+montage+"_x.npy"),xx);np.save(out/(sid+"_"+montage+"_features.npy"),ff);assert ff.shape==(n,52) and np.isfinite(ff).all()
  rec=dict(subject=sid,windows=n,sampling_hz=200,units=units,discarded_partial_samples=leftover,source_samples=int(ns[idx[0]]),direct_edf_physical_calibration_error=directerr,current_window_prefix_exact=prefix_exact,normalized_channel_rms=np.sqrt(np.mean(x.astype(float)**2,axis=(0,2))).tolist(),clipped_fraction=float((abs(x)>=20).mean()),raw_sha256=protocol["raw_files_sha256"][str(path.relative_to(OUT76/"raw"))],target_events_read=False);records.append(rec);print("EXTERNAL INPUT AUDIT",rec,flush=True)
 (OUT76/"input_audit.json").write_text(json.dumps(dict(status="PASS",records=records,total_windows=sum(v["windows"] for v in records),montages=2,no_future_window_preprocessing=True,no_target_event_labels=True,no_exact_montage_equivalence=True),indent=2));pd.DataFrame(records).to_csv(OUT76/"input_summary.csv",index=False)
if __name__=="__main__":run()
