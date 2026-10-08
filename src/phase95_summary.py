from common import *
from phase2_temporal import ema
import pandas as pd,hashlib
O=ROOT/"phase95";assert (O/"complete.json").exists();pr=json.loads((O/"protocol.json").read_text());assert hashlib.sha256((O/"protocol.json").read_bytes()).hexdigest()==(O/"protocol.sha256").read_text();assert json.loads((O/"prediction_audit.json").read_text())["status"]=="PASS"
for spec in pr["models"]:assert hashlib.sha256(Path(spec["path"]).read_bytes()).hexdigest()==spec["sha256"]
assert hashlib.sha256((ROOT/"phase79/protocol.json").read_bytes()).hexdigest()==pr["normalizer_protocol_sha256"]
rows=[];stages=[]
for c in pr["cases"]:
 name="_".join(c[k] for k in ["domain","subject","montage"]);data=np.load(O/"predictions"/(name+".npz"));assert len(data.files)==6;yy=np.load(c["labels"]);y=yy["y"] if c["labels"].endswith(".npz") else yy
 for method in data.files:
  p=data[method];assert p.shape==(len(y),5) and np.isfinite(p).all()
  for alpha in pr["ema"]:
   q=p if alpha==1 else ema(p,alpha);mm=metrics(y,q);cm=np.array(mm["confusion"]);rows.append(dict(domain=c["domain"],subject=c["subject"],montage=c["montage"],method=method,ema=alpha,macro_f1=100*mm["macro_f1"],accuracy=100*cm.trace()/cm.sum(),ce=mm["ce"],balanced_accuracy=100*mm["balanced_accuracy"]))
   for k,stage in enumerate(["W","N1","N2","N3","REM"]):stages.append(dict(domain=c["domain"],subject=c["subject"],montage=c["montage"],method=method,ema=alpha,stage=stage,f1=200*cm[k,k]/max(cm[k].sum()+cm[:,k].sum(),1),recall=100*cm[k,k]/max(cm[k].sum(),1),true_windows=int(cm[k].sum()),predicted_windows=int(cm[:,k].sum())))
df=pd.DataFrame(rows);assert len(df)==1584;df.to_csv(O/"participants.csv",index=False);pd.DataFrame(stages).to_csv(O/"stage_participants.csv",index=False);summary=df.groupby(["domain","montage","method","ema"])[["macro_f1","accuracy","ce","balanced_accuracy"]].mean();summary.to_csv(O/"summary.csv")
comparisons=[];rng=np.random.default_rng(95017);pairs=[("amplitude_augmented_selected","repeat_raw_selected","primary_source_selected"),("amplitude_augmented_final5","repeat_raw_final5","secondary_fixed5epochs"),("amplitude_augmented_selected","initial_guarded_median","secondary_vs_input_guard"),("amplitude_augmented_selected","initial_floor","secondary_vs_input_floor")]
for (domain,montage,alpha),g in df.groupby(["domain","montage","ema"]):
 p=g.pivot(index="subject",columns="method",values="macro_f1");idx=rng.integers(0,len(p),(10000,len(p)))
 for new,old,label in pairs:
  v=p[new]-p[old];ci=np.quantile(v.to_numpy()[idx].mean(1),[.025,.975]);comparisons.append(dict(domain=domain,montage=montage,ema=alpha,comparison=label,people=len(v),new_f1=p[new].mean(),old_f1=p[old].mean(),difference_pp=v.mean(),ci_low=ci[0],ci_high=ci[1],positive_people=int((v>0).sum()),harm_over_1pp=int((v< -1).sum()),worst_pp=v.min()))
comp=pd.DataFrame(comparisons);comp.to_csv(O/"paired_comparisons.csv",index=False);primary=comp[(comp.comparison=="primary_source_selected")&(comp.ema==.5)&comp.montage.isin(["native","left"])];primary.to_csv(O/"primary_results.csv",index=False)
stage=pd.DataFrame(stages).groupby(["domain","montage","method","ema","stage"]).agg(f1=("f1","mean"),recall=("recall","mean"),people_with_stage=("true_windows",lambda v:int((v>0).sum())),true_windows=("true_windows","sum"),predicted_windows=("predicted_windows","sum"));stage.to_csv(O/"stage_summary.csv")
audit=dict(status="PASS",method_streams_recomputed=1584,people_by_domain={"sleepedf":30,"healthy":10,"patient":19},classifier_seed=17,model_hashes_verified=True,no_testtime_updates=True,no_target_checkpoint_selection=True,all_target_people_previously_observed=True,independent_patient_phase83_primary_unchanged=True,not_new_blind_replication=True);(O/"audit.json").write_text(json.dumps(audit,indent=2))
(O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1450px;margin:35px auto}td,th{padding:6px;border:1px solid #ddd}</style><h1>源幅值增强模型：既有目标队列的静态探索性复核</h1><p>使用源选择锁定的seed17模型，推理时无参数更新。原SleepEDF30人、健康ISRUC10人和患者ISRUC19人均已在项目中观察；幅值问题曾影响方法设计，因此这是探索性外部复核，不是新的盲测，不替换phase83的预定患者主结果。</p><p>主比较为源选定增强模型减源选定原输入继续训练对照，同EMA0.5、外部左侧近似导联。源规则选择增强第3轮、原输入第0轮，因此有效训练轮数不同；另列固定第5轮配对作为训练更新次数相同的诊断比较。禁止从目标得分中选择这些模型。</p>'+primary.to_html(index=False,float_format=lambda v:f"{v:.6f}")+'<h2>同EMA0.5的全部比较</h2>'+comp[comp.ema==.5].to_html(index=False,float_format=lambda v:f"{v:.6f}")+'<h2>模型与简单输入处理的绝对结果</h2>'+summary.reset_index().query("ema==.5").to_html(index=False,float_format=lambda v:f"{v:.6f}")+'<p>单个分类器种子尚不足以确认稳定收益；区间条件于该模型与现有已观察受试者，不含训练种子和方法选择不确定性。另两个分类器种子的源复现已在本阶段评分前锁定。</p><p><a href="audit.json">指标复算</a> · <a href="../phase94/report.html">源选择与实际检查点审计</a> · <a href="../phase96/protocol.json">提前锁定的种子复现</a> · <a href="../phase83/report.html">保留原患者主比较</a> · <a href="stage_summary.csv">分期级结果</a></p>')
print(primary.to_string(index=False));print(comp[(comp.ema==.5)&(comp.comparison=="secondary_fixed5epochs")&comp.montage.isin(["native","left"])].to_string(index=False));print(audit)
