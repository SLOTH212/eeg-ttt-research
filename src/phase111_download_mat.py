from pathlib import Path
import requests,json,subprocess,hashlib,shutil,datetime
from urllib.parse import urlsplit
R=Path(__file__).resolve().parents[1];O=R/"phase111";(O/"scratch").mkdir(exist_ok=True)
node=next(n for n in json.loads((O/"decoded_public_listing.json").read_text()) if n["name"]=="subject1.mat");size=node["size"];key=bytes.fromhex(node["key_hex"]);aeskey=bytes(a^b for a,b in zip(key[:16],key[16:]));nonce=key[16:24];expected=key[24:32]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
protocol=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),purpose="Cross-format signal provenance check for prespecified ISRUC healthy subject1, no fitting or unit reinterpretation.",official_page="https://sleeptight.isr.uc.pt/?page_id=76",public_folder="https://mega.nz/folder/NUAwhSjC#jux7JQDkXgUOgMVquWEVzQ",file="subject1.mat",bytes=size,comparison="Original referenced nativechannels versus previouslyverifiedEDF sub-III001; investigate samplealignment, filtering and scale, not modelscoreselection.",official_caveat="ExtractedMAT files omit last30epochs per officialpage; do not replace fullEDF evaluation timeline.",storage="Only one newlydownloaded decryptedMAT in task scratch; streamciphertextwithoutdiskcopy; retainSHA256/MACandcomparisonartifacts, remove only this tasktemporaryMAT after auditedcomparison.",integrity="AES-CTR decryption and canonical128KiB..1MiB chunkCBCMAC then metaMAC, crosschecked against savedofficialMEGASDKsource. Publiclinkkey only, noaccountaccess.",sdk_sources={str(p.name):sha(p) for p in (O/"sdk_reference").iterdir()})
if (O/"protocol.json").exists():
 old=json.loads((O/"protocol.json").read_text());protocol["created_utc"]=old["created_utc"];assert old==protocol
else:(O/"protocol.json").write_text(json.dumps(protocol,indent=2));(O/"protocol.sha256").write_text(sha(O/"protocol.json"))
out=O/"scratch/subject1.mat"
if out.exists() and (O/"download_audit.json").exists():
 audit=json.loads((O/"download_audit.json").read_text());assert audit["status"]=="PASS" and sha(out)==audit["plaintext_sha256"];print("EXISTING VERIFIED");raise SystemExit
assert shutil.disk_usage(R).free>size+200_000_000
response=requests.post("https://g.api.mega.co.nz/cs",params={"id":111002,"n":"NUAwhSjC"},json=[{"a":"g","g":1,"n":node["handle"],"ssl":1}],timeout=40);response.raise_for_status();data=response.json()[0];assert isinstance(data,dict) and data["s"]==size;url=data["g"];assert isinstance(url,str) and urlsplit(url).scheme=="https" and urlsplit(url).hostname.endswith((".mega.co.nz",".mega.nz"));(O/"download_response.json").write_text(json.dumps(data,indent=2))
def crypt(block,mode,iv=None,decrypt=False):
 cmd=["openssl","enc","-aes-128-"+mode,"-K",aeskey.hex(),"-nopad"]
 if decrypt:cmd.append("-d")
 if iv is not None:cmd+=["-iv",iv.hex()]
 return subprocess.run(cmd,input=block,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True).stdout
mac=bytes(16);plainhash=hashlib.sha256();cipherhash=hashlib.sha256();offset=0;index=1;chunks=0;part=out.with_suffix(".mat.part")
with requests.get(url,stream=True,timeout=(20,60)) as r:
 r.raise_for_status()
 with part.open("wb") as f:
  while offset<size:
   length=min(min(index,8)*131072,size-offset);cipher=r.raw.read(length);assert len(cipher)==length
   plain=crypt(cipher,"ctr",nonce+(offset//16).to_bytes(8,"big"),True);assert len(plain)==length
   if offset==0:assert plain.startswith(b"MATLAB"),plain[:32]
   f.write(plain);plainhash.update(plain);cipherhash.update(cipher);padded=plain+bytes((-len(plain))%16);cm=crypt(padded,"cbc",nonce+nonce)[-16:];mac=crypt(bytes(a^b for a,b in zip(mac,cm)),"ecb");offset+=length;index+=1;chunks+=1
   if chunks%32==0:print("DOWNLOADED",offset,size,flush=True)
  assert r.raw.read(1)==b""
actual=bytes(a^b for a,b in zip(mac[:4],mac[4:8]))+bytes(a^b for a,b in zip(mac[8:12],mac[12:]))
assert actual==expected,(actual.hex(),expected.hex())
assert part.stat().st_size==size;part.replace(out)
audit=dict(status="PASS",file="subject1.mat",bytes=size,chunks=chunks,plaintext_sha256=plainhash.hexdigest(),ciphertext_sha256=cipherhash.hexdigest(),expected_meta_mac=expected.hex(),verified_meta_mac=actual.hex(),matlab_header_verified=True,downloaded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),official_public_link=True,labels_not_inspected=True)
(O/"download_audit.json").write_text(json.dumps(audit,indent=2));print(audit,flush=True)
