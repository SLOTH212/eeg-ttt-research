from phase86_meta_waveform import *
from phase119_models import ResidualLogitHead,transform as history_features
from phase99_covariance import transform as shape_transform
import datetime
O=ROOT/"phase120";O.mkdir(exist_ok=True);P=O/"predictions";P.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def legacy(c,mode):
 d,s,m=c["domain"],c["subject"],c["montage"]
 if mode=="full":return ROOT/"phase100/predictions"/f"{d}_{s}_{m}_full.npz"
 if d in ["source","healthy"]:return ROOT/"phase79/predictions"/f'{"source" if d=="source" else "external"}_{s}_{m}_{mode}.npz'
 if d=="sleepedf":return ROOT/"phase81/predictions"/f"{s}_{mode}.npz"
 return ROOT/"phase83/predictions"/f"{s}_{m}_{mode}.npz"
def run():
 decision=json.loads((ROOT/"phase119/decision.json").read_text());assert decision["advance_existing_target_exploratory"];assert json.loads((ROOT/"phase119/source_replay_audit.json").read_text())["status"]=="PASS"
 cases=json.loads((ROOT/"phase85/protocol.json").read_text())["cases"];parent=[s for s in json.loads((ROOT/"phase79/protocol.json").read_text())["models"] if s["model"]=="cbramod"];heads=[]
 for seed in [17,29,43]:
  for variant in ["current","repeat11","history11"]:
   path=ROOT/"phase119"/f"{variant}_seed{seed}"/"model.pt";heads.append(dict(seed=seed,variant=variant,path=str(path),sha256=sha(path)))
 pr=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=cases,parents=parent,heads=heads,modes="Source4 and oldSleepEDF30: rawonly. Healthy10and oldpatient19 bothleft/right: raw and alreadyfixedfullshapeguard. 150case-inputconditions.",primary="OldSleepEDF30 native rawinput, sameEMA.5, selectedhistory11 minus samecapacity selectedrepeat11; average3seed metricswithinperson, pairedperson bootstrap10000 rng120017 95percentileCI. No primaryreplacement by EMA1 or externalcohort.",secondary="Currenthead,frozenparent, originalcontext11teacher andfixedhalfparentteacherblend; EMA1/.8; healthy andoldpatient separate,leftprimarywithincohort/rightsecondary; existingfullshapeinputvsraw.",interpretation="Alltargets previouslyobserved. Newpredictions evaluated exploratorily, notblindreplication. Staticheads selectedsourceonly; no targetparameterupdates or thresholds. NewI041-I060 excludedbecause raw/featurecachesnotretained; no redownload/cohortexpansion.",shape_reference_sha256=sha(ROOT/"phase99/source_reference.npz"),scope="All30selectedheadstates fixedand hashed beforetargetinference (9heads actual); failed/fallbackseeds retained. Classificationlabels notread during predictionphase; no newblindclaim.",cpu_ids=list(range(8,24)),gpu=True)
 pr["scope"]="All9selectedheadstates fixedand hashed beforetargetinference; failed/fallbackseeds retained. Classificationlabels notread during predictionphase; no newblindclaim."
 pp=O/"protocol.json"
 if pp.exists():
  old=json.loads(pp.read_text());pr["created_utc"]=old["created_utc"];assert old==pr
 else:pp.write_text(json.dumps(pr,indent=2));(O/"protocol.sha256").write_text(sha(pp))
 parents={};hm={};states={}
 for sp in parent:
  path=Path(sp["path"]);assert sha(path)==sp["sha256"];m=meta_model();m.load_state_dict(torch.load(path,map_location="cpu",weights_only=False)["state"]);m.cuda().eval().requires_grad_(False);parents[sp["seed"]]=m;states[f"parent{sp['seed']}"]=snapshot(m)
 for sp in heads:
  path=Path(sp["path"]);assert sha(path)==sp["sha256"];ck=torch.load(path,map_location="cpu",weights_only=False);m=ResidualLogitHead(ck["dim"]);m.load_state_dict(ck["state"]);m.cuda().eval().requires_grad_(False);key=f"{sp['variant']}_seed{sp['seed']}";hm[key]=m;states[key]=snapshot(m)
 reference=np.load(ROOT/"phase99/source_reference.npz")["reference"];records=[]
 for ci,c in enumerate(cases):
  name="_".join(c[k] for k in ["domain","subject","montage"]);modes=["raw"] if c["domain"] in ["source","sleepedf"] else ["raw","full"];raw=np.load(c["x"],mmap_mode="r")
  for mode in modes:
   dest=P/f"{name}_{mode}.npz";ap=P/f"{name}_{mode}_audit.json"
   if dest.exists() and ap.exists():
    rec=json.loads(ap.read_text());assert sha(dest)==rec["prediction_sha256"];records.append(rec);continue
   if mode=="full":
    scalar=legacy(c,"guarded_median");gp=scalar.with_name(scalar.stem+"_normalizer.npz");gains=np.load(gp)["gains"];xx,aux,stats=shape_transform(raw,gains,reference,"full");prefix=shape_transform(raw[:65],gains[:65],reference,"full")[0];assert np.array_equal(prefix,xx[:65]);gainhash=sha(gp)
   else:xx=raw;gainhash=None
   parts=[prep(np.array(xx[a:a+128]),[.1,45]) for a in range(0,len(xx),128)];x=torch.from_numpy(np.concatenate(parts));del parts
   oldpath=legacy(c,mode);old=np.load(oldpath);result={};checks=[]
   for seed,m in parents.items():
    with torch.inference_mode():
     logits=torch.cat([m(b.cuda()).cpu() for b in x.split(128)]);centered=(logits-logits.mean(1,keepdim=True)).numpy();base=torch.from_numpy(centered).softmax(1).numpy()
    ref=old[f"cbramod_seed{seed}"];err=float(abs(base-ref).max());assert err<2e-5 and np.array_equal(base.argmax(1),ref.argmax(1));result[f"logits_seed{seed}"]=centered;result[f"frozen_seed{seed}"]=base
    for variant in ["current","repeat11","history11"]:
     key=f"{variant}_seed{seed}";head=hm[key];ff=history_features(centered,variant);assert np.array_equal(ff[:65],history_features(centered[:65],variant));tensor=torch.from_numpy(ff).cuda()
     with torch.inference_mode():
      pred=torch.cat([head(tensor[a:a+1024]).softmax(1).cpu() for a in range(0,len(tensor),1024)]).numpy();prefix=head(tensor[:65]).softmax(1).cpu().numpy()
     pe=float(abs(prefix-pred[:65]).max());assert pe<2e-5 and np.array_equal(prefix.argmax(1),pred[:65].argmax(1));assert all(torch.equal(v.cpu(),states[key][k]) for k,v in head.state_dict().items());result[key]=pred;checks.append(dict(seed=seed,variant=variant,head_prefix_error=pe,parent_legacy_error=err))
     if c["domain"]=="source":
      saved=np.load(ROOT/"phase119"/f"{variant}_seed{seed}"/f"{c['subject']}.npz")["p"];assert abs(pred-saved).max()<2e-5 and np.array_equal(pred.argmax(1),saved.argmax(1))
    assert all(torch.equal(v.cpu(),states[f"parent{seed}"][k]) for k,v in m.state_dict().items())
   np.savez_compressed(dest,**result);rec=dict(case=c,mode=mode,windows=len(raw),prediction_sha256=sha(dest),legacy_path=str(oldpath),legacy_sha256=sha(oldpath),gain_sha256=gainhash,checks=checks,frozen_states_exact=True,history_prefix_exact=True);ap.write_text(json.dumps(rec,indent=2));records.append(rec)
  print("CASE",ci+1,len(cases),name,flush=True)
 assert len(records)==150
 (O/"audit.json").write_text(json.dumps(dict(status="PASS",case_input_conditions=len(records),heads=9,parents=3,labels_not_read=True,no_adaptation=True,records=records),indent=2));print("COMPLETE",flush=True)
if __name__=="__main__":
 with sdpa_kernel(SDPBackend.MATH):run()
