from phase119_models import *
import hashlib,datetime
os.environ["CUDA_VISIBLE_DEVICES"]="0"
O=ROOT/"phase131";O.mkdir(exist_ok=True);P=O/"predictions";P.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run():
 assert json.loads((ROOT/"phase130/source_replay_audit.json").read_text())["status"]=="PASS"
 sources={"cbramod":"phase120","labram":"phase123"};heads=[]
 for backbone in sources:
  for seed in [17,29,43]:
   for variant in ["current","repeat11","history11"]:
    path=ROOT/"phase130"/f"{backbone}_{variant}_seed{seed}"/"model.pt";heads.append(dict(backbone=backbone,seed=seed,variant=variant,path=str(path),sha256=sha(path)))
 pr=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),purpose="Fixed checkpoint-selector sensitivity on samecachedtargetinputs, evaluatingallalternate-selector heads andfallbacks.",sources=sources,heads=heads,source_selector=[7,17,27,37],original_selector=[5,15,25,35],source_audit_sha256=sha(ROOT/"phase130/source_replay_audit.json"),original_metric_hashes={k:sha(ROOT/v/"participants.csv") for k,v in sources.items()},primary="For eachprespecifiedbackbone, oldSleepEDF30 native rawEMA.5: alternate-selectedhistory minusalternate-selectedrepeat. Reportboth; onlydescribe sharedrobustness ifbothpairedperson95percentileCIs lower>0. Samepeopleacrossbackbones, notindependentcohorts.",sensitivity="Paired change inhistory-minus-repeat effect betweenalternateandoriginalselectors; samepersonbootstrap10000 RNG131017. Also absoluteheadperformance vsoriginalselectedhistory/frozenparent/teacher, andfixedexternal/EMAsecondaryresults.",scope="Originalresults retained. Do notselectbettersourcevalidationgroup bytargetscore. Alltargets previouslyobserved; no newblindclaim, trainingrecipe or targetparameterupdates.",cpu_ids=list(range(8,24)),gpu=True)
 p=O/"protocol.json"
 if p.exists():
  old=json.loads(p.read_text());pr["created_utc"]=old["created_utc"];assert old==pr
 else:p.write_text(json.dumps(pr,indent=2));(O/"protocol.sha256").write_text(sha(p))
 records=[]
 for backbone,phase in sources.items():
  oldaudit=json.loads((ROOT/phase/"audit.json").read_text());assert oldaudit["status"]=="PASS";oldpr=json.loads((ROOT/phase/"protocol.json").read_text());models={};states={}
  for sp in [h for h in heads if h["backbone"]==backbone]:
   path=Path(sp["path"]);assert sha(path)==sp["sha256"];ck=torch.load(path,map_location="cpu",weights_only=False);parent=next(s for s in oldpr["parents"] if s["seed"]==sp["seed"]);assert ck["parent_sha256"]==parent["sha256"] and ck["source_selector"]==[7,17,27,37];m=ResidualLogitHead(ck["dim"]);m.load_state_dict(ck["state"]);m.cuda().eval().requires_grad_(False);key=f"{sp['variant']}_seed{sp['seed']}";models[key]=m;states[key]={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
  for ci,oldrec in enumerate(oldaudit["records"]):
   c=oldrec["case"];mode=oldrec["mode"];name="_".join(c[k] for k in ["domain","subject","montage"]);dest=P/f"{backbone}_{name}_{mode}.npz";ap=dest.with_name(dest.stem+"_audit.json")
   if dest.exists() and ap.exists():
    rec=json.loads(ap.read_text());assert sha(dest)==rec["prediction_sha256"];records.append(rec);continue
   inputpath=ROOT/phase/"predictions"/f"{name}_{mode}.npz";assert sha(inputpath)==oldrec["prediction_sha256"];data=np.load(inputpath);result={};checks=[]
   for key,m in models.items():
    variant,seedstr=key.rsplit("_seed",1);seed=int(seedstr);logits=data[f"logits_seed{seed}"];ff=transform(logits,variant);assert np.array_equal(ff[:65],transform(logits[:65],variant));x=torch.from_numpy(ff).cuda()
    with torch.inference_mode():
     p=torch.cat([m(x[a:a+1024]).softmax(1).cpu() for a in range(0,len(x),1024)]).numpy();prefix=m(x[:65]).softmax(1).cpu().numpy()
    err=float(abs(p[:65]-prefix).max());assert err<2e-5 and np.array_equal(p[:65].argmax(1),prefix.argmax(1));sourceerr=None
    if c["domain"]=="source":
     gold=np.load(ROOT/"phase130"/f"{backbone}_{variant}_seed{seed}"/f"{c['subject']}.npz")["p"];sourceerr=float(abs(p-gold).max());assert sourceerr<2e-5 and np.array_equal(p.argmax(1),gold.argmax(1))
    assert all(torch.equal(v.cpu(),states[key][k]) for k,v in m.state_dict().items());result[key]=p;checks.append(dict(model=key,prefix_probability_error=err,source_probability_error=sourceerr))
   np.savez_compressed(dest,**result);rec=dict(backbone=backbone,case=c,mode=mode,prediction_sha256=sha(dest),parent_cache_path=str(inputpath),parent_cache_sha256=sha(inputpath),checks=checks,states_unchanged_exact=True);ap.write_text(json.dumps(rec,indent=2));records.append(rec)
   print("SELECTOR TARGET",backbone,ci+1,len(oldaudit["records"]),name,mode,flush=True)
  del models;torch.cuda.empty_cache()
 assert len(records)==300
 (O/"audit.json").write_text(json.dumps(dict(status="PASS",conditions=len(records),heads=18,labels_not_read=True,parent_caches_exact_original=True,no_selection_between_selectors=True,no_target_updates=True,records=records),indent=2));print("COMPLETE",flush=True)
if __name__=="__main__":run()
