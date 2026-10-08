from pathlib import Path
import requests,json,hashlib,datetime,shutil,concurrent.futures,time
ROOT=Path(__file__).resolve().parents[1];o=ROOT/"phase76"
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def manifest_checksum(p,v):
 if v['checksum_algorithm']=='sha256':return sha(p)
 assert v['checksum_algorithm']=='git'
 b=p.read_bytes();return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\x00'+b).hexdigest()
def run():
 manifest=json.loads((o/"cohort3_manifest.json").read_text());chosen=[v for v in manifest if v["path"].endswith(("_eeg.edf","_eeg.json","_channels.tsv"))];assert sum(v["path"].endswith(".edf") for v in chosen)==10
 protocol=dict(locked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),purpose="external input compatibility; no target labels or model tuning",dataset="ISRUC-Sleep cohort III, all10 healthy participants in published subgroup",version="NEMAR nm000111 v1.0.1",source_url="https://sleeptight.isr.uc.pt/?page_id=48",mirror_url="https://www.nemar.org/dataset/nm000111",license="Official site states publicly available for research; mirror machine-readable License n/a; local research only, no raw redistribution",participants=[f"sub-III{i:03d}" for i in range(1,11)],raw_files_sha256={v["path"]:v["checksum"] for v in chosen},native_channels=["F3-A2","C3-A2","O1-A2","F4-A1","C4-A1","O2-A1"],primary_montage=["F3-C3","C3-O1"],secondary_montage=["F4-C4","C4-O2"],montage_limit="Lateral frontal-central and central-occipital bipolar proxies, not exact SleepEDF Fpz-Cz/Pz-Oz; no equivalence claim",preprocessing="Non-overlap full30s windows from record time0; intra-window resample_poly 200->100, perwindow center,/100uV,clip+-20; existing model preprocessing afterward; no future-window reads or whole-night stats",trailing_data="Keep every complete30s window, record partial remainder; no sleep-label cropping",events_not_downloaded_in_this_phase=True,metadata_exposure="participants.tsv downloaded includes stage-frequency summaries; preview exposed cohortI rows only; no stage summaries used for cohort or montage selection")
 if not (o/"protocol.json").exists():(o/"protocol.json").write_text(json.dumps(protocol,indent=2))
 assert shutil.disk_usage(ROOT).free>sum(v["size"] for v in chosen)+1_000_000_000
 def fetch(v):
  dest=o/"raw"/v["path"];dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists() and dest.stat().st_size==v["size"] and manifest_checksum(dest,v)==v["checksum"]:return dict(path=v["path"],bytes=v["size"],sha256=sha(dest),manifest_algorithm=v["checksum_algorithm"],cached=True)
  tmp=dest.with_suffix(dest.suffix+".part")
  for attempt in range(3):
   try:
    with requests.get(v["bytes_url"],stream=True,timeout=(20,40)) as response:
     response.raise_for_status()
     with tmp.open("wb") as f:
      for b in response.iter_content(1<<20):f.write(b)
    assert tmp.stat().st_size==v["size"];actual=sha(tmp);assert manifest_checksum(tmp,v)==v["checksum"];tmp.replace(dest);print("VERIFIED DOWNLOAD",v["path"],v["size"],flush=True);return dict(path=v["path"],bytes=v["size"],sha256=actual,manifest_algorithm=v["checksum_algorithm"],cached=False)
   except Exception as e:
    print("DOWNLOAD RETRY",v["path"],attempt,str(e),flush=True)
    if attempt==2:raise
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:records=list(ex.map(fetch,chosen))
 (o/"download_audit.json").write_text(json.dumps(dict(status="PASS",files=records,total_bytes=sum(v["bytes"] for v in records),target_event_labels_not_downloaded=True),indent=2))
if __name__=="__main__":run()
