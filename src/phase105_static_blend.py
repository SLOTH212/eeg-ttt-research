from common import *
from phase2_temporal import ema
import pandas as pd,hashlib,datetime
O=ROOT/"phase105";O.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
cases=json.loads((ROOT/"phase85/protocol.json").read_text())["cases"]
for sid in json.loads((ROOT/"phase101/cohort_lock.json").read_text())["included"]:
 for montage in ["left","right"]:cases.append(dict(domain="patient_new",subject=sid,montage=montage,labels=str(ROOT/"phase101/labels"/(sid+"_y.npy"))))
def path_for(c,mode):
 d,s,m=c["domain"],c["subject"],c["montage"]
 if d=="patient_new":return ROOT/"phase101/predictions"/f"{s}_{m}_{mode}.npz"
 if mode=="full":return ROOT/"phase100/predictions"/f"{d}_{s}_{m}_full.npz"
 name="guarded_median" if mode=="scalar" else "raw"
 if d in ["source","healthy"]:return ROOT/"phase79/predictions"/f'{"source" if d=="source" else "external"}_{s}_{m}_{name}.npz'
 if d=="sleepedf":return ROOT/"phase81/predictions"/f"{s}_{name}.npz"
 return ROOT/"phase83/predictions"/f"{s}_{m}_{name}.npz"
protocol=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),purpose="Full-night three-seed static teacher/student controls after phase104 failed staticcompetitiveness. No new model, training, or targetadaptation.",cases=cases,models=["cbramod","context11","half_blend"],seeds=[17,29,43],modes=["raw","scalar","full"],ema=[1.,.8,.5],blend="Same-seed .5CBraMod+.5context11 probabilities at eachwindow, then samecausal probabilityEMA. Metrics averagedacrossseeds withinperson; no crossseedprobabilityensemble.",primary="Exploratory oldSleepEDF30 native rawinput, EMA.5, halfblend versusstrongercontext11 teacher. Othercohorts/modes/EMA separatelyreported, no tuning or selection.",limitations="All cohorts now previously observed, including phase101. Context11 has extra past10windows and differenttrainingbudget; directteacher and blend explicit to prevent attributingteacherstrength toTTT.",input_hashes={str(path_for(c,m)):sha(path_for(c,m)) for c in cases for m in ["raw","scalar","full"]})
(O/"protocol.json").write_text(json.dumps(protocol,indent=2));(O/"protocol.sha256").write_text(sha(O/"protocol.json"));rows=[]
for c in cases:
 yy=np.load(c["labels"]);y=yy["y"] if c["labels"].endswith(".npz") else yy
 for mode in protocol["modes"]:
  p=path_for(c,mode);z=np.load(p)
  for seed in protocol["seeds"]:
   b=z[f"cbramod_seed{seed}"];t=z[f"context11_seed{seed}"]
   for method,q in [("cbramod",b),("context11",t),("half_blend",.5*b+.5*t)]:
    for alpha in protocol["ema"]:
     pp=q if alpha==1 else ema(q,alpha);mm=metrics(y,pp);cm=np.array(mm["confusion"]);rows.append(dict(domain=c["domain"],subject=c["subject"],montage=c["montage"],mode=mode,method=method,seed=seed,ema=alpha,macro_f1=100*mm["macro_f1"],balanced_accuracy=100*mm["balanced_accuracy"],ce=mm["ce"],accuracy=100*cm.trace()/cm.sum()))
df=pd.DataFrame(rows);assert len(df)==10692;df.to_csv(O/"participants.csv",index=False);keys=["domain","subject","montage","mode","method","ema"];assert (df.groupby(keys).seed.nunique()==3).all();person=df.groupby(keys)[["macro_f1","balanced_accuracy","ce","accuracy"]].mean().reset_index();person.to_csv(O/"seedmean_people.csv",index=False);summary=person.groupby(["domain","montage","mode","method","ema"])[["macro_f1","balanced_accuracy","ce","accuracy"]].mean().reset_index();summary.to_csv(O/"summary.csv",index=False)
comparisons=[]
for (domain,montage,mode,alpha),g in person.groupby(["domain","montage","mode","ema"]):
 pp=g.pivot(index="subject",columns="method",values="macro_f1");pc=g.pivot(index="subject",columns="method",values="ce");rng=np.random.default_rng(105017);idx=rng.integers(0,len(pp),(10000,len(pp)))
 for new,old in [("half_blend","context11"),("half_blend","cbramod"),("context11","cbramod")]:
  d=pp[new]-pp[old];ci=np.quantile(d.to_numpy()[idx].mean(1),[.025,.975]);comparisons.append(dict(domain=domain,montage=montage,mode=mode,ema=alpha,new_method=new,old_method=old,people=len(d),new_f1=pp[new].mean(),old_f1=pp[old].mean(),difference_pp=d.mean(),ci_low=ci[0],ci_high=ci[1],ce_gain=(pc[old]-pc[new]).mean(),positive_people=int((d>0).sum()),harm_over_1pp=int((d< -1).sum()),worst_pp=d.min()))
comp=pd.DataFrame(comparisons);comp.to_csv(O/"paired_comparisons.csv",index=False);primary=comp[(comp.domain=="sleepedf")&(comp.montage=="native")&(comp["mode"]=="raw")&(comp.ema==.5)&(comp.new_method=="half_blend")&(comp.old_method=="context11")];assert len(primary)==1;primary.to_csv(O/"primary_results.csv",index=False)
audit=dict(status="PASS",cases=len(cases),streams=len(df),fixed_blend_weight=.5,no_new_training=True,no_online_parameterupdates=True,three_seed_metrics_within_person=True,no_new_blind_claim=True)
(O/"audit.json").write_text(json.dumps(audit,indent=2))
(O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1450px;margin:35px auto}td,th{padding:6px;border:1px solid #ddd}</style><h1>整夜三种子静态教师与固定概率平均</h1><p>phase104中教师梯度改善学生但落后直接教师与静态平均，故不继续在线更新。本阶段对全部既有整夜记录作固定静态对照：每个种子分别将CBraMod与context11概率各取一半，再使用相同EMA；先在人内平均三个种子的指标，不合并种子概率。没有新训练、权重搜索或参数更新。</p><h2>固定探索性主比较：SleepEDF原输入，相同EMA0.5，平均对教师</h2>'+primary.to_html(index=False,float_format=lambda v:f"{v:.6f}")+'<h2>EMA0.5的原生/左侧全部对照</h2>'+comp[(comp.ema==.5)&comp.montage.isin(["native","left"])].to_html(index=False,float_format=lambda v:f"{v:.6f}")+'<p>context11使用当前及过去10窗频谱特征，训练预算与CBraMod不同；它是可用强静态参照，不是严格同架构归因实验。所有人群现已观察，包括新患者组；本次不再具有独立验证地位，也不改变phase101原定失败主结果。CE改善与Macro-F1改善分别报告。</p><p><a href="audit.json">汇总审计</a> · <a href="protocol.json">固定定义和输入哈希</a> · <a href="../phase104/report.html">脉冲WRITE与教师</a> · <a href="../phase101/report.html">保留独立主结果</a></p>')
print(primary.to_string(index=False));print(comp[(comp.ema==.5)&comp.montage.isin(["native","left"])&(comp.new_method=="half_blend")&(comp.old_method=="context11")].to_string(index=False));print(audit)
