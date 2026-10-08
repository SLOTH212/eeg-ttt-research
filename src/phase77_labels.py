from pathlib import Path
import requests,json,hashlib,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];o=ROOT/"phase77"
def run():
 assert (o/"predictions_complete.json").exists();pr=json.loads((o/"protocol.json").read_text());assert hashlib.sha256((o/"protocol.json").read_bytes()).hexdigest()==(o/"protocol.sha256").read_text();manifest=json.loads((ROOT/"phase76/cohort3_manifest.json").read_text());wanted=[v for v in manifest if v["path"].endswith(("_events.tsv","_events.json"))];assert len(wanted)==20;records=[]
 for v in wanted:
  p=o/"labels_raw"/v["path"];p.parent.mkdir(parents=True,exist_ok=True);b=p.read_bytes() if p.exists() else requests.get(v["bytes_url"],timeout=30).content;assert len(b)==v["size"]
  check=hashlib.sha256(b).hexdigest() if v["checksum_algorithm"]=="sha256" else hashlib.sha1(b"blob "+str(len(b)).encode()+b"\x00"+b).hexdigest();assert check==v["checksum"];p.write_bytes(b);records.append(dict(path=v["path"],sha256=hashlib.sha256(b).hexdigest(),manifest_algorithm=v["checksum_algorithm"]))
 audits=[];out=o/"labels";out.mkdir(exist_ok=True)
 for sid in pr["participants"]:
  p=o/"labels_raw"/sid/"eeg"/(sid+"_task-sleep_events.tsv");df=pd.read_csv(p,sep="\t");n=len(np.load(ROOT/"phase76/processed"/(sid+"_left_x.npy"),mmap_mode="r"));y=np.full(n,-1,dtype=np.int64);assigned=np.zeros(n,bool);unknown=set(df.trial_type)-set(pr["label_map"]);assert not unknown,unknown
  code_map=df[["trial_type","value"]].drop_duplicates();assert code_map.trial_type.is_unique and code_map.value.is_unique
  for row in df.itertuples(index=False):
   begin=float(row.onset);duration=float(row.duration);assert abs(duration-30)<1e-8 and abs(begin/30-round(begin/30))<1e-8;t=int(round(begin/30));assert t>=0
   if t>=n:raise ValueError((sid,"event beyond signal",t,n))
   assert not assigned[t];assigned[t]=True;y[t]=pr["label_map"][row.trial_type]
   assert int(row.sample)==int(round(begin*200))
  assert assigned.all(),(sid,int((~assigned).sum()));np.save(out/(sid+"_y.npy"),y);rec=dict(subject=sid,signal_windows=n,event_rows=len(df),assigned_windows=int(assigned.sum()),unscored=int((y<0).sum()),stage_counts=np.bincount(y[y>=0],minlength=5).tolist(),onset_grid_exact=True,no_label_based_cropping=True,event_code_mapping=code_map.to_dict("records"));audits.append(rec);print("EXTERNAL LABEL ALIGNMENT",rec,flush=True)
 (o/"label_audit.json").write_text(json.dumps(dict(status="PASS",files=records,participants=audits,protocol_locked_before_labels=True,predictions_completed_before_labels=True,scorer1_only=True),indent=2))
if __name__=="__main__":run()
