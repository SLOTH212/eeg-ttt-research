from common import *
from phase2_temporal import ema
import pandas as pd,hashlib,datetime
O=ROOT/"phase128";O.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
parents={"cbramod":"phase120","labram":"phase123"}
pr=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),purpose="Fixed .5temporalhead/teacher mixtures, matchedheadcapacity/training/control; cachedpredictions only.",backbones=parents,new_methods=["half_history_teacher","half_repeat_teacher"],weights=".5same-seed headprobability +.5same-seed originalcontext11teacher, then existingEMA. Weightinheritedfromphase104/105, no grid or selection.",primary="CBraMod oldSleepEDF30 native rawinput, EMA.5: half_history_teacher minus half_repeat_teacher. Three-seed metricmeanwithinperson; pairedperson10000bootstrap RNG128017.",secondary="ExactLaBraMbackbone replication on samepeople (notindependentcohort); parent/teacherhalfblend, teacher andhistoryalone controls; separateexternalcohorts/montages/inputmodes/EMA1,.8.",scope="Allsource andtargets alreadyobserved; exploratory, no newblindclaim. Bothheadbranches have equalcapacity/30sourceepochs; mixture includesextra teacherhistory/training/compute. No newtraining or parameterupdates.",prediction_protocol_hashes={k:sha(ROOT/v/"protocol.json") for k,v in parents.items()},cpu_ids=list(range(8,24)),gpu=False)
pp=O/"protocol.json"
if pp.exists():
 old=json.loads(pp.read_text());pr["created_utc"]=old["created_utc"];assert old==pr
else:pp.write_text(json.dumps(pr,indent=2));(O/"protocol.sha256").write_text(sha(pp))
rows=[];hashes=[];prefixerr=0.;oldframes=[]
for backbone,phase in parents.items():
 audit=json.loads((ROOT/phase/"audit.json").read_text());assert audit["status"]=="PASS";old=pd.read_csv(ROOT/phase/"participants.csv");old["backbone"]=backbone;oldframes.append(old)
 for rec in audit["records"]:
  c=rec["case"];mode=rec["mode"];name="_".join(c[k] for k in ["domain","subject","montage"]);path=ROOT/phase/"predictions"/f"{name}_{mode}.npz";assert sha(path)==rec["prediction_sha256"];legacy=Path(rec["legacy_path"]);assert sha(legacy)==rec["legacy_sha256"];z=np.load(path);teacher=np.load(legacy);loaded=np.load(c["labels"]);y=loaded["y"] if c["labels"].endswith(".npz") else loaded
  for seed in [17,29,43]:
   tp=teacher[f"context11_seed{seed}"]
   for variant,head in [("half_history_teacher","history11"),("half_repeat_teacher","repeat11")]:
    p=.5*z[f"{head}_seed{seed}"]+.5*tp;assert np.isfinite(p).all() and abs(p.sum(1)-1).max()<3e-6
    hashes.append(dict(backbone=backbone,case=name,mode=mode,seed=seed,variant=variant,probability_sha256=hashlib.sha256(p.tobytes()).hexdigest()))
    for alpha in [1.,.8,.5]:
     q=p if alpha==1 else ema(p,alpha);prefix=p[:65] if alpha==1 else ema(p[:65],alpha);err=float(abs(q[:65]-prefix).max());assert err==0;prefixerr=max(prefixerr,err);met=metrics(y,q);rows.append(dict(backbone=backbone,domain=c["domain"],subject=c["subject"],montage=c["montage"],mode=mode,seed=seed,variant=variant,ema=alpha,macro_f1=100*met["macro_f1"],balanced_accuracy=100*met["balanced_accuracy"],ce=met["ce"]))
df=pd.DataFrame(rows);assert len(df)==5400;df.to_csv(O/"new_streams.csv",index=False);(O/"prediction_hashes.json").write_text(json.dumps(hashes,indent=2));allrows=pd.concat([df]+oldframes,ignore_index=True);keys=["backbone","domain","subject","montage","mode","variant","ema"];assert (allrows.groupby(keys).seed.nunique()==3).all();person=allrows.groupby(keys)[["macro_f1","balanced_accuracy","ce"]].mean().reset_index();person.to_csv(O/"people.csv",index=False);summary=person.groupby(["backbone","domain","montage","mode","variant","ema"])[["macro_f1","balanced_accuracy","ce"]].mean().reset_index();summary.to_csv(O/"summary.csv",index=False)
comparisons=[]
for (backbone,domain,montage,mode,alpha),g in person.groupby(["backbone","domain","montage","mode","ema"]):
 tab=g.pivot(index="subject",columns="variant",values="macro_f1");rng=np.random.default_rng(128017);ix=rng.integers(0,len(tab),(10000,len(tab)))
 for ref in ["half_repeat_teacher","static_half_blend","teacher","history11"]:
  d=tab.half_history_teacher-tab[ref];ci=np.quantile(d.to_numpy()[ix].mean(1),[.025,.975]);comparisons.append(dict(backbone=backbone,domain=domain,montage=montage,mode=mode,ema=alpha,reference=ref,people=len(d),history_blend_f1=float(tab.half_history_teacher.mean()),reference_f1=float(tab[ref].mean()),difference_pp=float(d.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]),positive_people=int((d>0).sum()),harm_over1pp=int((d< -1).sum()),worst_pp=float(d.min())))
comp=pd.DataFrame(comparisons);comp.to_csv(O/"comparisons.csv",index=False);primary=comp[(comp.backbone=="cbramod")&(comp.domain=="sleepedf")&(comp.montage=="native")&(comp["mode"]=="raw")&(comp.ema==.5)&(comp.reference=="half_repeat_teacher")];assert len(primary)==1;primary.to_csv(O/"primary.csv",index=False)
audit=dict(status="PASS",new_streams=len(df),no_new_training=True,no_weight_search=True,prefix_exact=True,three_seed_metrics_averaged_within_person=True,all_people_previously_observed=True,paired_capacity_and_training_control=True);(O/"audit.json").write_text(json.dumps(audit,indent=2))
show=comp[(comp.ema==.5)&comp.montage.isin(["native","left"])&((comp.domain.isin(["source","sleepedf"]))|(comp["mode"]=="full"))]
(O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1500px;margin:35px auto}td,th{padding:5px;border:1px solid #ddd}</style><h1>固定半混合中，历史信息还有多少额外收益？</h1><p>仅复用已冻结预测，先将同种子历史头或同容量重复头与同一个原context11教师各取0.5，再应用既有EMA。没有搜索混合权重、重新训练或参数更新。主比较两侧的额外源30轮头训练、参数量及教师信息相同；与原父模型/教师混合比较则存在额外头训练预算差异。</p><h2>固定探索性主比较</h2>'+primary.to_html(index=False,float_format=lambda v:f"{v:.6f}")+'<h2>EMA0.5、原生/左侧的主要对照</h2>'+show.to_html(index=False,float_format=lambda v:f"{v:.6f}")+'<p>主比较为CBraMod的SleepEDF30人原始输入、EMA0.5。LaBraM在同一人群上复现方法，不是新的独立人群。每人先平均三个种子指标，再按人计算配对区间；没有种子概率集成。所有目标已观察，外部及其他EMA是次要结果，不能替代主比较。</p><p><a href="protocol.json">固定权重与比较</a> · <a href="audit.json">审计</a> · <a href="comparisons.csv">全部结果</a></p>')
print(primary.to_string(index=False));print(show.to_string(index=False));print(audit)
