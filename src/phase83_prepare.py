from phase76_prepare import header,direct_samples,transform,features
from pathlib import Path
import json,hashlib,shutil,requests,numpy as np,pyedflib,re
ROOT=Path(__file__).resolve().parents[1];OUT83=ROOT/"phase83"
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def checked_download(v,p):
 assert v["bytes_url"].startswith("https://data.nemar.org/nm000111/v1.0.1/");p.parent.mkdir(parents=True,exist_ok=True)
 def correct(path):
  if not path.exists() or path.stat().st_size!=v["size"]:return False
  if v["checksum_algorithm"]=="sha256":return sha(path)==v["checksum"]
  assert v["checksum_algorithm"]=="git";b=path.read_bytes();return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\x00"+b).hexdigest()==v["checksum"]
 if correct(p):return
 for attempt in range(3):
  try:
   with requests.get(v["bytes_url"],stream=True,timeout=(20,40)) as r:
    r.raise_for_status()
    with p.with_suffix(p.suffix+".part").open("wb") as f:
     for b in r.iter_content(1<<20):f.write(b)
   tmp=p.with_suffix(p.suffix+".part");assert correct(tmp);tmp.replace(p);return
  except Exception as e:
   print("DOWNLOAD RETRY",v["path"],attempt,repr(e),flush=True)
   if attempt==2:raise
def resolve(labels,sites,references):
 norm=[v.upper().replace(" ","").replace("_","-").removeprefix("EEG") for v in labels]
 for ref in references:
  found=[[labels[i] for i,v in enumerate(norm) if re.fullmatch(site+"-?"+ref,v)] for site in sites]
  if all(len(v)==1 for v in found):return [v[0] for v in found],ref
 raise ValueError(("no unique shared-reference triplet",sites,labels))
def run():
 pr=json.loads((OUT83/"protocol.json").read_text());assert hashlib.sha256((OUT83/"protocol.json").read_bytes()).hexdigest()==(OUT83/"protocol.sha256").read_text();manifest=json.loads((OUT83/"manifest.json").read_text());out=OUT83/"processed";out.mkdir(exist_ok=True);scratch=OUT83/"download_tmp";scratch.mkdir(exist_ok=True);records=[]
 for sid in pr["subjects"]:
  done=out/(sid+"_input_audit.json")
  if done.exists():
   rec=json.loads(done.read_text())
   for name,digest in rec["processed_sha256"].items():assert sha(out/name)==digest
   records.append(rec);continue
  entries=[v for v in manifest if v["path"].split("/")[0]==sid];edf=next(v for v in entries if v["path"].endswith(".edf"));assert shutil.disk_usage(ROOT).free>edf["size"]+800_000_000
  for v in entries:
   if v["path"].endswith(("_eeg.json","_channels.tsv")):checked_download(v,OUT83/"metadata"/v["path"])
  path=scratch/(sid+".edf");checked_download(edf,path);h=header(path);left,lref=resolve(h["labels"],["F3","C3","O1"],["A2","M2"]);right,rref=resolve(h["labels"],["F4","C4","O2"],["A1","M1"]);channels=left+right;hi=[h["labels"].index(c) for c in channels];units=[h["units"][i] for i in hi];factor={"uv":1.,"mv":1000.,"v":1e6};factors=np.array([factor[u.replace("µ","u").replace("μ","u").lower()] for u in units]);assert [int(h["samples_per_record"][i])/h["record_seconds"] for i in hi]==[200.]*6
  with pyedflib.EdfReader(str(path),annotations_mode=0) as f:
   labels=f.getSignalLabels();fi=[labels.index(c) for c in channels];ns=f.getNSamples();assert len(set(int(ns[i]) for i in fi))==1;assert all(f.getSampleFrequency(i)==200 for i in fi);n=int(ns[fi[0]])//6000;leftover=int(ns[fi[0]])%6000;parts=[];err=0.
   for a,b,scale in zip(fi,hi,factors):err=max(err,float(abs((f.readSignal(a,start=0,n=1000)-direct_samples(path,h,b))*scale).max()))
   assert err<1e-8
   for first in range(0,n,64):
    count=min(64,n-first);native=np.stack([f.readSignal(i,start=first*6000,n=count*6000) for i in fi])*factors[:,None];windows=native.reshape(6,count,6000).transpose(1,0,2);x=transform(windows);parts.append(x)
    if first==0:
     prefix=transform(windows[:1]);assert np.array_equal(prefix,x[:1]);changed=windows[:2].copy();changed[1]+=1000;assert np.array_equal(transform(changed)[:1],prefix)
   xx=np.concatenate(parts);assert np.isfinite(xx).all()
  outputs={}
  for montage,sl in [("left",slice(0,2)),("right",slice(2,4))]:
   x=xx[:,sl].copy();ff=features(x);assert ff.shape==(n,52) and np.isfinite(ff).all()
   for name,value in [(sid+"_"+montage+"_x.npy",x),(sid+"_"+montage+"_features.npy",ff)]:np.save(out/name,value);outputs[name]=sha(out/name)
  rec=dict(subject=sid,windows=n,source_samples=int(ns[fi[0]]),trailing_partial_samples=leftover,channels=channels,shared_references=[lref,rref],header_units=units,units_to_microvolt=factors.tolist(),direct_physical_calibration_error=err,window_prefix_exact=True,future_perturbation_exact=True,source_sha256=edf["checksum"],source_bytes=edf["size"],processed_sha256=outputs,raw_retained=False,raw_removal_scope="onlynewverifiedtask-temporaryEDFafterderivedcacheaudit",target_event_labels_read=False)
  done.write_text(json.dumps(rec,indent=2));assert path.parent.resolve()==scratch.resolve();path.unlink();records.append(rec);print("PATIENT INPUT VERIFIED",sid,n,units,"temporary EDF released",flush=True)
 (OUT83/"input_audit.json").write_text(json.dumps(dict(status="PASS",people=len(records),records=records,total_windows=sum(v["windows"] for v in records),total_downloaded_edf_bytes=sum(v["source_bytes"] for v in records),no_target_eventlabels=True),indent=2))
if __name__=="__main__":run()
