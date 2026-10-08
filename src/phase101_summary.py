from common import *
from phase2_temporal import ema
import pandas as pd,hashlib
O=ROOT/"phase101";pr=json.loads((O/"protocol.json").read_text());assert hashlib.sha256((O/"protocol.json").read_bytes()).hexdigest()==(O/"protocol.sha256").read_text();assert json.loads((O/"label_audit.json").read_text())["status"]=="PASS";cohort=json.loads((O/"cohort_lock.json").read_text());ids=cohort["included"];assert ids
seal=json.loads((O/"prediction_hashes.json").read_text());assert all(hashlib.sha256((O/"predictions"/name).read_bytes()).hexdigest()==digest for name,digest in seal.items())
rows=[];stages=[]
for sid in ids:
 y=np.load(O/"labels"/(sid+"_y.npy"))
 for montage in pr["montages"]:
  gain=np.load(O/"predictions"/f"{sid}_{montage}_scalar_normalizer.npz")["gains"];trigger=bool((gain>1).any())
  for mode in pr["modes"]:
   z=np.load(O/"predictions"/f"{sid}_{montage}_{mode}.npz")
   for key in z.files:
    model,seed=key.rsplit("_seed",1);p=z[key];assert p.shape==(len(y),5)
    for alpha in pr["ema"]:
     q=p if alpha==1 else ema(p,alpha);mm=metrics(y,q);cm=np.array(mm["confusion"]);base=dict(subject=sid,montage=montage,mode=mode,model=model,seed=int(seed),ema=alpha,triggered=trigger,triggered_fraction=float((gain>1).mean()))
     rows.append(dict(**base,macro_f1=100*mm["macro_f1"],balanced_accuracy=100*mm["balanced_accuracy"],accuracy=100*cm.trace()/cm.sum(),ce=mm["ce"]))
     if alpha==.5:
      for k,stage in enumerate(["W","N1","N2","N3","REM"]):stages.append(dict(**base,stage=stage,f1=200*cm[k,k]/max(cm[k].sum()+cm[:,k].sum(),1),recall=100*cm[k,k]/max(cm[k].sum(),1),true_windows=int(cm[k].sum()),predicted_windows=int(cm[:,k].sum())))
df=pd.DataFrame(rows);assert len(df)==len(ids)*270;df.to_csv(O/"participants.csv",index=False);keys=["subject","montage","mode","model","ema","triggered","triggered_fraction"];assert (df.groupby(keys).seed.nunique()==3).all();person=df.groupby(keys)[["macro_f1","balanced_accuracy","accuracy","ce"]].mean().reset_index();person.to_csv(O/"seedmean_people.csv",index=False)
summary=person.groupby(["montage","mode","model","ema"])[["macro_f1","balanced_accuracy","accuracy","ce"]].mean().reset_index();summary.to_csv(O/"summary.csv",index=False)
pairs=[("full","scalar"),("diagonal","scalar"),("full","raw"),("full","floor"),("scalar","raw"),("floor","raw")]
comp=[]
for subset in ["all","triggered_only_secondary"]:
 d=person if subset=="all" else person[person.triggered]
 for (montage,model,alpha),g in d.groupby(["montage","model","ema"]):
  p=g.pivot(index="subject",columns="mode",values="macro_f1");rng=np.random.default_rng(101017);idx=rng.integers(0,len(p),(10000,len(p)))
  for new,old in pairs:
   v=p[new]-p[old];ci=np.quantile(v.to_numpy()[idx].mean(1),[.025,.975]);comp.append(dict(subset=subset,montage=montage,model=model,ema=alpha,new_mode=new,old_mode=old,people=len(v),new_f1=p[new].mean(),old_f1=p[old].mean(),difference_pp=v.mean(),ci_low=ci[0],ci_high=ci[1],positive_people=int((v>0).sum()),harm_over_1pp=int((v< -1).sum()),worst_pp=v.min()))
comp=pd.DataFrame(comp);comp.to_csv(O/"paired_comparisons.csv",index=False);primary=comp[(comp.subset=="all")&(comp.montage=="left")&(comp.model=="cbramod")&(comp.ema==.5)&(comp.new_mode=="full")&(comp.old_mode=="scalar")];assert len(primary)==1;primary.to_csv(O/"primary_results.csv",index=False)
st=pd.DataFrame(stages);st.to_csv(O/"stage_participants.csv",index=False);sp=st.groupby(["subject","montage","mode","model","ema","stage"])[["f1","recall","true_windows","predicted_windows"]].mean().reset_index();sp.groupby(["montage","mode","model","ema","stage"]).agg(f1=("f1","mean"),recall=("recall","mean"),true_windows=("true_windows","sum"),predicted_windows=("predicted_windows","sum")).to_csv(O/"stage_summary.csv")
r=primary.iloc[0];decision=dict(primary_mean_supported=bool(r.difference_pp>0 and r.ci_low>0),primary=r.to_dict(),scope="Fixed-method independent-person cohort from same dataset; all developmentevidence retained; not newdataset or gradientTTTsuccess.",safety="Individual harms reported, no universal safety or clinicalclaim.")
(O/"decision.json").write_text(json.dumps(decision,indent=2))
audit=dict(status="PASS",people=len(ids),planned=20,excluded=cohort["excluded"],streams=len(df),predictions_hashes_verified_before_scoring=True,person_seedmetric_averaging=True,no_probability_ensemble=True,no_pseudoreplication=True,no_target_tuning=True)
(O/"summary_audit.json").write_text(json.dumps(audit,indent=2))
(O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1450px;margin:35px auto}td,th{padding:6px;border:1px solid #ddd}</style><h1>固定通道形状校正：后续独立患者验证</h1><p>连续ISRUC I041–I060于逐人标签读取前固定；只按预先规定的输入参考、物理单位和采样兼容性排除，不补人。所有合格记录先完成五种输入、三个模型各三个种子、两侧导联的预测并封存哈希，再下载逐窗标签。方法已在先前人群开发；同一数据集的新受试者验证不等于跨新数据集验证。公开全体人员统计文件曾下载，但未使用本组的统计行进行选择。</p><h2>原定主比较</h2>'+primary.to_html(index=False,float_format=lambda v:f"{v:.6f}")+'<p>支持均值改善：'+str(decision["primary_mean_supported"])+'。完整通道校正与标量校正使用相同幅值触发和增益；每窗保持裁剪前联合RMS。每个人先平均三个种子的指标，再按人配对重采样，未合并概率或将种子视为独立受试者。</p><h2>同EMA0.5全部受试者比较</h2>'+comp[(comp.subset=="all")&(comp.ema==.5)].to_html(index=False,float_format=lambda v:f"{v:.6f}")+'<h2>绝对分数</h2>'+summary[summary.ema==.5].to_html(index=False,float_format=lambda v:f"{v:.6f}")+'<h2>输入排除记录</h2><pre>'+json.dumps(cohort["excluded"],indent=2)+'</pre><p>其余比较与触发子组均为预定次要描述，不能替换主结果。既有SleepEDF和健康组受损结果仍保留。这是无梯度输入归一化，不能宣布TTT有效，也不构成临床验证。</p><p><a href="protocol.json">预定协议</a> · <a href="prediction_hashes.json">预测封存</a> · <a href="input_prediction_audit.json">输入与推理审计</a> · <a href="label_audit.json">标签审计</a> · <a href="summary_audit.json">统计审计</a> · <a href="../phase100/report.html">先前开发人群结果</a></p>')
print(primary.to_string(index=False));print(decision);print(audit)
