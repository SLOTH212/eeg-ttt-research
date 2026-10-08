from common import *
from phase2_temporal import ema
import pandas as pd,hashlib
o=ROOT/"phase77";pr=json.loads((o/"protocol.json").read_text());assert hashlib.sha256((o/"protocol.json").read_bytes()).hexdigest()==(o/"protocol.sha256").read_text()
assert json.loads((o/"prediction_audit.json").read_text())["status"]=="PASS";assert json.loads((o/"label_audit.json").read_text())["status"]=="PASS";records=[];stages=[]
for spec in pr["models"]:assert hashlib.sha256(Path(spec["path"]).read_bytes()).hexdigest()==spec["sha256"]
for sid in pr["participants"]:
 y=np.load(o/"labels"/(sid+"_y.npy"));valid=y>=0
 for montage in pr["montages"]:
  data=np.load(o/"predictions"/(sid+"_"+montage+".npz"));assert len(data.files)==9
  for spec in pr["models"]:
   key=f'{spec["model"]}_seed{spec["seed"]}';p=data[key];assert p.shape==(len(y),5) and np.isfinite(p).all() and np.max(abs(p.sum(1)-1))<2e-6
   for smooth in pr["ema"]:
    q=p if smooth==1 else ema(p,smooth);m=metrics(y,q);cm=np.array(m["confusion"]);records.append(dict(subject=sid,montage=montage,model=spec["model"],seed=spec["seed"],ema=smooth,macro_f1=100*m["macro_f1"],accuracy=100*cm.trace()/cm.sum(),balanced_accuracy=100*m["balanced_accuracy"],kappa=m["kappa"],ce=m["ce"],windows=m["n"]))
    precision=cm.diagonal()/np.maximum(cm.sum(0),1);recall=cm.diagonal()/np.maximum(cm.sum(1),1);f1=2*cm.diagonal()/np.maximum(cm.sum(0)+cm.sum(1),1)
    for j,name in enumerate(["W","N1","N2","N3","REM"]):stages.append(dict(subject=sid,montage=montage,model=spec["model"],seed=spec["seed"],ema=smooth,stage=name,f1=100*f1[j],recall=100*recall[j],precision=100*precision[j],true_windows=int(cm.sum(1)[j]),predicted_windows=int(cm.sum(0)[j])))
df=pd.DataFrame(records);assert len(df)==540;df.to_csv(o/"participants.csv",index=False);stage=pd.DataFrame(stages);stage.to_csv(o/"stage_participants.csv",index=False);person=df.groupby(["subject","montage","model","ema"])[["macro_f1","accuracy","balanced_accuracy","kappa","ce"]].mean().reset_index();person.to_csv(o/"seedmean_people.csv",index=False);summary=person.groupby(["montage","model","ema"])[["macro_f1","accuracy","balanced_accuracy","kappa","ce"]].mean();summary.to_csv(o/"summary.csv")
rng=np.random.default_rng(77017);idx=rng.integers(0,10,size=(10000,10));comparisons=[]
for montage in pr["montages"]:
 for smooth in pr["ema"]:
  pivot=person[(person.montage==montage)&(person.ema==smooth)].pivot(index="subject",columns="model",values="macro_f1");assert len(pivot)==10
  for ref in ["labram","context11"]:
   d=(pivot.cbramod-pivot[ref]).to_numpy();ci=np.quantile(d[idx].mean(1),[.025,.975]);comparisons.append(dict(montage=montage,ema=smooth,comparison="cbramod_vs_"+ref,difference_pp=float(d.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]),positive_people=int((d>0).sum()),harm_over_1pp=int((d< -1).sum()),worst_person_pp=float(d.min()),primary=montage=="left" and smooth==.5 and ref=="labram"))
comparisons=pd.DataFrame(comparisons);comparisons.to_csv(o/"paired_comparisons.csv",index=False);primary=comparisons[comparisons.primary].iloc[0].to_dict();(o/"primary_result.json").write_text(json.dumps(primary,indent=2))
ss=stage.groupby(["subject","montage","model","ema","stage"])[["f1","recall","precision","true_windows","predicted_windows"]].mean().reset_index();stage_summary=ss.groupby(["montage","model","ema","stage"]).agg(f1=("f1","mean"),recall=("recall","mean"),precision=("precision","mean"),people_with_stage=("true_windows",lambda v:int((v>0).sum())),true_windows=("true_windows","sum"),predicted_windows=("predicted_windows","sum"));stage_summary.to_csv(o/"stage_summary.csv")
audit=dict(status="PASS",prediction_method_streams_recomputed=540,individual_people=10,classifier_seeds=[17,29,43],fixed_source_models=9,montages=2,no_target_updates=True,no_seed_pseudoreplication=True,no_probability_ensemble=True,protocol_hash_verified=True,checkpoint_hashes_verified=True,uncertainty="Paired person bootstrap conditional on observed seeds, source selection and proxy montages; not full protocol uncertainty");(o/"audit.json").write_text(json.dumps(audit,indent=2))
html='<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1300px;margin:35px auto}td,th{padding:6px;border:1px solid #ddd}</style><h1>ISRUC 健康队列固定模型静态迁移</h1><p>完整10人、8,889个30秒窗口。源检查点、两套近似双极导联、三个EMA系数在读取事件标签前固定；全部预测先完成，之后才读取评分标签。每人先平均三个种子的指标，再平均受试者；不做概率集成，不把种子当独立受试者。</p><p>主比较：左侧近似导联、同EMA0.5，CBraMod相对同预算全量微调LaBraM。历史context11为更强的因果频谱历史基线，含前10窗，训练轮数也不同，不作架构因果归因。</p>'+pd.DataFrame([primary]).to_html(index=False)+'<h2>全部预先列出的静态条件</h2>'+summary.to_html()+'<h2>配对人级差值</h2>'+comparisons.to_html(index=False)+'<p>重要边界：F3−C3/C3−O1及右侧对称导联不是SleepEDF训练导联的等价替代。波形幅值按原始EDF微伏单位处理，不依目标标签重标定；LaBraM源训练位置嵌入保持不变。该结果混合了人群、导联、采集和幅值差异，不能把失败单独归于适应策略。本轮没有执行TTT，也不证明任何保留控制器有效。置信区间只反映这10人及固定种子条件下的人级不确定性。</p><p><a href="audit.json">独立评分审计</a> · <a href="protocol.json">标签前固定协议</a> · <a href="label_audit.json">事件对齐</a> · <a href="../phase76/report.html">输入与单位审计</a> · <a href="stage_summary.csv">分期指标</a></p>'
(o/"report.html").write_text(html);print(summary.to_string());print(comparisons.to_string(index=False));print(audit)
