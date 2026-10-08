from common import *
from phase2_temporal import ema
import pandas as pd,hashlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
O=ROOT/"phase120";pr=json.loads((O/"protocol.json").read_text());assert hashlib.sha256((O/"protocol.json").read_bytes()).hexdigest()==(O/"protocol.sha256").read_text().strip();audit=json.loads((O/"audit.json").read_text());assert audit["status"]=="PASS"
rows=[];stages=[];errors=[]
for rec in audit["records"]:
 c=rec["case"];mode=rec["mode"];name="_".join(c[k] for k in ["domain","subject","montage"]);path=O/"predictions"/f"{name}_{mode}.npz";assert hashlib.sha256(path.read_bytes()).hexdigest()==rec["prediction_sha256"];lp=Path(rec["legacy_path"]);assert hashlib.sha256(lp.read_bytes()).hexdigest()==rec["legacy_sha256"];z=np.load(path);legacy=np.load(lp);loaded=np.load(c["labels"]);y=loaded["y"] if c["labels"].endswith(".npz") else loaded
 for seed in [17,29,43]:
  parent=z[f"frozen_seed{seed}"];teacher=legacy[f"context11_seed{seed}"];candidates={variant:z[f"{variant}_seed{seed}"] for variant in ["frozen","current","repeat11","history11"]};candidates.update(teacher=teacher,static_half_blend=.5*parent+.5*teacher)
  for variant,p in candidates.items():
   for alpha in [1.,.8,.5]:
    q=p if alpha==1 else ema(p,alpha);met=metrics(y,q);cm=np.asarray(met["confusion"]);den=cm.sum(1)+cm.sum(0);f1=np.divide(2*np.diag(cm),den,out=np.zeros(5),where=den>0);err=abs(f1.mean()-met["macro_f1"]);assert err<1e-12;errors.append(err);base=dict(domain=c["domain"],subject=c["subject"],montage=c["montage"],mode=mode,seed=seed,variant=variant,ema=alpha)
    rows.append(dict(**base,macro_f1=100*met["macro_f1"],balanced_accuracy=100*met["balanced_accuracy"],accuracy=100*np.trace(cm)/cm.sum(),ce=met["ce"]))
    if alpha==.5:
     for k,label in enumerate(["W","N1","N2","N3","REM"]):stages.append(dict(**base,stage=label,f1=100*f1[k],recall=100*cm[k,k]/max(cm[k].sum(),1),true_windows=int(cm[k].sum()),predicted_windows=int(cm[:,k].sum())))
df=pd.DataFrame(rows);assert len(df)==8100;df.to_csv(O/"participants.csv",index=False);keys=["domain","subject","montage","mode","variant","ema"];assert (df.groupby(keys).seed.nunique()==3).all();person=df.groupby(keys)[["macro_f1","balanced_accuracy","accuracy","ce"]].mean().reset_index();person.to_csv(O/"seedmean_people.csv",index=False);summary=person.groupby(["domain","montage","mode","variant","ema"])[["macro_f1","balanced_accuracy","accuracy","ce"]].mean().reset_index();summary.to_csv(O/"summary.csv",index=False)
comp=[]
for (domain,montage,mode,alpha),g in person.groupby(["domain","montage","mode","ema"]):
 table=g.pivot(index="subject",columns="variant",values="macro_f1");rng=np.random.default_rng(120017);ix=rng.integers(0,len(table),(10000,len(table)))
 for reference in ["repeat11","current","frozen","teacher","static_half_blend"]:
  delta=table.history11-table[reference];ci=np.quantile(delta.to_numpy()[ix].mean(1),[.025,.975]);comp.append(dict(domain=domain,montage=montage,mode=mode,ema=alpha,reference=reference,people=len(delta),history_f1=float(table.history11.mean()),reference_f1=float(table[reference].mean()),difference_pp=float(delta.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]),positive_people=int((delta>0).sum()),harm_over_1pp=int((delta < -1).sum()),worst_pp=float(delta.min())))
comp=pd.DataFrame(comp);comp.to_csv(O/"paired_comparisons.csv",index=False);primary=comp[(comp.domain=="sleepedf")&(comp.montage=="native")&(comp["mode"]=="raw")&(comp.ema==.5)&(comp.reference=="repeat11")];assert len(primary)==1;primary.to_csv(O/"primary_results.csv",index=False)
stage=pd.DataFrame(stages);stage.to_csv(O/"stage_participants.csv",index=False);sp=stage.groupby(keys+["stage"])[["f1","recall","true_windows","predicted_windows"]].mean().reset_index();sp.groupby(["domain","montage","mode","variant","ema","stage"]).agg(f1=("f1","mean"),recall=("recall","mean"),true_windows=("true_windows","sum"),predicted_windows=("predicted_windows","sum")).to_csv(O/"stage_summary.csv")
sa=dict(status="PASS",streams=len(df),input_conditions=len(audit["records"]),independent_macro_f1_error=max(errors),three_seed_metric_means_within_person=True,no_probability_ensemble=True,no_target_selection=True,no_online_parameter_updates=True,all_targets_previously_observed=True)
(O/"summary_audit.json").write_text(json.dumps(sa,indent=2))
show=comp[(comp.ema==.5)&comp.montage.isin(["native","left"])]
plot=show[(show.domain=="sleepedf")|(show.domain.isin(["healthy","patient"])&(show["mode"]=="full"))]
fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
for ax,domain,title in zip(axes,["sleepedf","healthy","patient"],["SleepEDF30 / raw","Healthy10 / full shape","Patients19 / full shape"]):
 dd=plot[plot.domain==domain].set_index("reference").loc[["repeat11","frozen","teacher","static_half_blend"]]
 yy=np.arange(4);ax.errorbar(dd.difference_pp,yy,xerr=np.vstack([dd.difference_pp-dd.ci_low,dd.ci_high-dd.difference_pp]),fmt="o",capsize=3);ax.axvline(0,color="gray",lw=1);ax.set_yticks(yy,["Repeat current","Frozen parent","Teacher","Half blend"]);ax.set(title=title,xlabel="History minus control Macro-F1 (pp)");ax.invert_yaxis();ax.grid(axis="x",alpha=.2)
fig.suptitle("Fixed EMA .5; three-seed metrics averaged within each person; all cohorts previously observed")
fig.savefig(O/"history_effect.png",dpi=160);plt.close(fig)
(O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1500px;margin:35px auto}td,th{padding:5px;border:1px solid #ddd}img{max-width:100%}</style><h1>静态因果历史头：既有目标数据探索性复核</h1><p>九个源选定残差头和三个冻结CBraMod固定后，评价150个记录-输入条件、8100条模型/种子/EMA指标流。当前头、重复当前11次、过去10窗加当前的历史头，均保留所有种子的回退状态。原context11教师和固定半混合为额外静态对照。没有目标参数更新。</p><h2>预定主比较</h2>'+primary.to_html(index=False,float_format=lambda v:f"{v:.6f}")+'<p>主比较是原SleepEDF30人、原始输入、相同EMA=0.5下历史头减同容量重复头。原始概率/EMA0.8及外部人群为次要结果，不能替换主结果。</p><img src="history_effect.png"><h2>EMA0.5、原生/左侧蒙太奇</h2>'+show.to_html(index=False,float_format=lambda v:f"{v:.6f}")+'<p>人内先平均三种子的指标，再按人配对bootstrap估计区间；不对概率作三种子集成。外部健康10人和既有患者19人分别汇总，raw与既有fullshape输入并列。左右蒙太奇不是独立个体。所有目标人群此前已反复观察，所以不宣称新盲验证。I041–I060瞬态原始输入未保留，本阶段不重新下载或扩展人群。</p><p>额外源30轮监督训练与模型容量已计入，静态因果历史不是TTT。<a href="../phase119/report.html">源训练和推进门槛</a> · <a href="protocol.json">固定目标协议</a> · <a href="summary_audit.json">独立统计审计</a> · <a href="paired_comparisons.csv">全部次要比较</a></p>')
print(primary.to_string(index=False));print(show.to_string(index=False));print(sa)
