from common import *
from phase2_temporal import ema
import pandas as pd,hashlib
O=ROOT/"phase107";pr=json.loads((O/"protocol.json").read_text());assert (O/"complete.json").exists();assert json.loads((O/"audit.json").read_text())["status"]=="PASS";assert hashlib.sha256((O/"protocol.json").read_bytes()).hexdigest()==(O/"protocol.sha256").read_text();rows=[];stages=[]
for c in pr["cases"]:
 name="_".join(c[k] for k in ["domain","subject","montage"]);z=np.load(O/"predictions"/(name+".npz"));yy=np.load(c["labels"]);y=yy["y"] if c["labels"].endswith(".npz") else yy
 for key in z.files:
  variant,seed=key.rsplit("_seed",1);p=z[key]
  for alpha in pr["ema"]:
   q=p if alpha==1 else ema(p,alpha);mm=metrics(y,q);cm=np.array(mm["confusion"]);base=dict(domain=c["domain"],subject=c["subject"],montage=c["montage"],variant=variant,seed=int(seed),ema=alpha);rows.append(dict(**base,macro_f1=100*mm["macro_f1"],balanced_accuracy=100*mm["balanced_accuracy"],accuracy=100*cm.trace()/cm.sum(),ce=mm["ce"]))
   if alpha==.5:
    for k,stage in enumerate(["W","N1","N2","N3","REM"]):stages.append(dict(**base,stage=stage,f1=200*cm[k,k]/max(cm[k].sum()+cm[:,k].sum(),1),recall=100*cm[k,k]/max(cm[k].sum(),1),true_windows=int(cm[k].sum()),predicted_windows=int(cm[:,k].sum())))
df=pd.DataFrame(rows);assert len(df)==3312;df.to_csv(O/"participants.csv",index=False);keys=["domain","subject","montage","variant","ema"];assert (df.groupby(keys).seed.nunique()==3).all();person=df.groupby(keys)[["macro_f1","balanced_accuracy","accuracy","ce"]].mean().reset_index();person.to_csv(O/"seedmean_people.csv",index=False);summary=person.groupby(["domain","montage","variant","ema"])[["macro_f1","balanced_accuracy","accuracy","ce"]].mean().reset_index();summary.to_csv(O/"summary.csv",index=False)
comp=[]
for (domain,montage,alpha),g in person.groupby(["domain","montage","ema"]):
 p=g.pivot(index="subject",columns="variant",values="macro_f1");rng=np.random.default_rng(107017);idx=rng.integers(0,len(p),(10000,len(p)))
 for old in ["relative_repeat11","relative","full_context11"]:
  d=p.relative_context11-p[old];ci=np.quantile(d.to_numpy()[idx].mean(1),[.025,.975]);comp.append(dict(domain=domain,montage=montage,ema=alpha,comparison="relative_context11-minus-"+old,people=len(d),new_f1=p.relative_context11.mean(),old_f1=p[old].mean(),difference_pp=d.mean(),ci_low=ci[0],ci_high=ci[1],positive_people=int((d>0).sum()),harm_over_1pp=int((d< -1).sum()),worst_pp=d.min()))
comp=pd.DataFrame(comp);comp.to_csv(O/"paired_comparisons.csv",index=False);primary=comp[(comp.domain=="sleepedf")&(comp.montage=="native")&(comp.ema==.5)&(comp.comparison=="relative_context11-minus-relative_repeat11")];assert len(primary)==1;primary.to_csv(O/"primary_results.csv",index=False)
st=pd.DataFrame(stages);st.to_csv(O/"stage_participants.csv",index=False);sp=st.groupby(keys+["stage"])[["f1","recall","true_windows","predicted_windows"]].mean().reset_index();sp.groupby(["domain","montage","variant","ema","stage"]).agg(f1=("f1","mean"),recall=("recall","mean"),true_windows=("true_windows","sum"),predicted_windows=("predicted_windows","sum")).to_csv(O/"stage_summary.csv")
audit=dict(status="PASS",streams=3312,three_seed_metric_averaging_within_person=True,no_probabilityensemble=True,no_target_checkpoint_selection=True,raw_inputs_only=True,no_parameter_adaptation=True,all_targets_previouslyobserved=True)
(O/"summary_audit.json").write_text(json.dumps(audit,indent=2))
(O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1450px;margin:35px auto}td,th{padding:6px;border:1px solid #ddd}</style><h1>相对频谱与因果历史的固定静态对照</h1><p>四种源训练模型：当前26维相对频谱、重复当前11次的286维对照、真实当前及过去10窗的286维相对频谱、572维完整频谱历史。三种子、相同30轮/批次/标签呈现和训练顺序；286维两组初始参数相同，仅历史信息与对应源特征缩放不同。四位源验证者的原始概率Macro-F1选择各检查点。完整频谱模型也重新按同一顺序协议训练，并非直接复用旧教师。</p><h2>固定探索性主比较：历史对同维度重复当前窗</h2>'+primary.to_html(index=False,float_format=lambda v:f"{v:.6f}")+'<h2>EMA0.5全部对照</h2>'+comp[comp.ema==.5].to_html(index=False,float_format=lambda v:f"{v:.6f}")+'<h2>绝对结果</h2>'+summary[summary.ema==.5].to_html(index=False,float_format=lambda v:f"{v:.6f}")+'<p>全部目标此前已观察，方案由既有幅值与时间信息结果启发；不作为新盲验证。相对特征减少绝对幅值依赖，但数值稳定常数意味着它不是严格的任意幅值不变表示。每人先平均种子指标，再按人配对估计区间。没有测试时参数更新，也未从目标结果选择窗口长度、轮数或EMA。新患者20人的瞬态特征未保留，本阶段仅评价既有缓存的92条记录。</p><p><a href="../phase106/report.html">源训练与预算审计</a> · <a href="audit.json">冻结推理与前缀审计</a> · <a href="summary_audit.json">统计审计</a></p>')
print(primary.to_string(index=False));print(comp[(comp.ema==.5)&comp.montage.isin(["native","left"])].to_string(index=False));print(audit)
