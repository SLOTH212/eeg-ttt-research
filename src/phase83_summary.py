from common import *
from phase2_temporal import ema
from scipy.signal import lfilter
import pandas as pd,hashlib
o=ROOT/"phase83";pr=json.loads((o/"protocol.json").read_text());assert hashlib.sha256((o/"protocol.json").read_bytes()).hexdigest()==(o/"protocol.sha256").read_text();assert json.loads((o/"prediction_audit.json").read_text())["status"]=="PASS";la=json.loads((o/"label_audit.json").read_text());assert la["status"]=="PASS";base=json.loads((ROOT/"phase79/protocol.json").read_text());assert hashlib.sha256((ROOT/"phase79/protocol.json").read_bytes()).hexdigest()==pr["normalizer_protocol_sha256"]
cohort=json.loads((o/"cohort_lock_after_input_audit.json").read_text());eligible=cohort["included"];N=len(eligible);rows=[];stages=[];normrows=[];maxerr=0.
for spec in pr["models"]:assert hashlib.sha256(Path(spec["path"]).read_bytes()).hexdigest()==spec["sha256"]
for sid in eligible:
 y=np.load(o/"labels"/(sid+"_y.npy"))
 for montage in pr["montages"]:
  raw=np.load(o/"processed"/(sid+"_"+montage+"_x.npy"),mmap_mode="r");assert len(raw)==len(y);rms=np.sqrt(np.mean(raw.astype(float)**2,axis=(1,2)))
  for mode in pr["input_modes"]:
   if mode!="raw":
    a=np.load(o/"predictions"/f"{sid}_{montage}_{mode}_normalizer.npz");valid=rms>=1e-6;logs=np.log(rms[valid]);z,_=lfilter([.01],[1,-.99],logs,zi=[.99*logs[0]]);estimate=np.exp(z);g=np.ones(len(raw))
    if mode=="logema01":gg=np.clip(base["source_reference_rms"]/estimate,.1,100)
    elif mode=="guarded_median":gg=np.clip(np.where(estimate<base["lower_source_rms_quantile"],base["source_reference_rms"]/estimate,1.),1,100)
    else:gg=np.clip(np.maximum(1,base["lower_source_rms_quantile"]/estimate),1,100)
    g[valid]=gg;err=float(abs(g-a["gains"]).max());maxerr=max(maxerr,err);assert err<1e-10 and np.array_equal(rms,a["current_rms"]);normrows.append(dict(subject=sid,montage=montage,mode=mode,modified_windows=int((g!=1).sum()),windows=len(g),gain_median=float(np.median(g)),gain_max=float(g.max())))
   d=np.load(o/"predictions"/f"{sid}_{montage}_{mode}.npz");assert len(d.files)==(12 if mode=="raw" else 9)
   for key in d.files:
    model,seed=key.rsplit("_seed",1);p=d[key];assert p.shape==(len(y),5) and np.isfinite(p).all() and abs(p.sum(1)-1).max()<2e-6
    for smooth in pr["ema"]:
     q=p if smooth==1 else ema(p,smooth);m=metrics(y,q);cm=np.array(m["confusion"]);rows.append(dict(subject=sid,montage=montage,mode=mode,model=model,seed=int(seed),ema=smooth,macro_f1=100*m["macro_f1"],accuracy=100*cm.trace()/cm.sum(),balanced_accuracy=100*m["balanced_accuracy"],kappa=m["kappa"],ce=m["ce"],scored_windows=m["n"]))
     for j,name in enumerate(["W","N1","N2","N3","REM"]):stages.append(dict(subject=sid,montage=montage,mode=mode,model=model,seed=int(seed),ema=smooth,stage=name,f1=200*cm[j,j]/max(cm[j].sum()+cm[:,j].sum(),1),recall=100*cm[j,j]/max(cm[j].sum(),1),true_windows=int(cm[j].sum()),predicted_windows=int(cm[:,j].sum())))
df=pd.DataFrame(rows);assert len(df)==N*234;df.to_csv(o/"participants.csv",index=False);pd.DataFrame(stages).to_csv(o/"stage_participants.csv",index=False);pd.DataFrame(normrows).to_csv(o/"normalizer_people.csv",index=False);person=df.groupby(["subject","montage","mode","model","ema"])[["macro_f1","accuracy","balanced_accuracy","kappa","ce"]].mean().reset_index();person.to_csv(o/"seedmean_people.csv",index=False);summary=person.groupby(["montage","mode","model","ema"])[["macro_f1","accuracy","balanced_accuracy","ce"]].mean();summary.to_csv(o/"summary.csv");rng=np.random.default_rng(83017);comp=[]
for montage in pr["montages"]:
 for model in ["cbramod","labram","context11"]:
  for smooth in pr["ema"]:
   pv=person[(person.montage==montage)&(person.model==model)&(person.ema==smooth)].pivot(index="subject",columns="mode",values="macro_f1");assert len(pv)==N;idx=rng.integers(0,N,size=(10000,N))
   for mode in ["logema01","guarded_median","floor"]:
    delta=pv[mode]-pv.raw;ci=np.quantile(delta.to_numpy()[idx].mean(1),[.025,.975]);comp.append(dict(montage=montage,model=model,ema=smooth,mode=mode,raw_f1=float(pv.raw.mean()),adjusted_f1=float(pv[mode].mean()),difference_pp=float(delta.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]),positive_people=int((delta>0).sum()),harm_over_1pp=int((delta< -1).sum()),worst_person_pp=float(delta.min()),primary=montage=="left" and model=="cbramod" and smooth==.5 and mode=="guarded_median"))
comp=pd.DataFrame(comp);comp.to_csv(o/"paired_comparisons.csv",index=False);primary=comp[comp.primary].iloc[0].to_dict();(o/"primary_result.json").write_text(json.dumps(primary,indent=2));stage=pd.DataFrame(stages).groupby(["subject","montage","mode","model","ema","stage"])[["f1","recall","true_windows","predicted_windows"]].mean().reset_index();stage.groupby(["montage","mode","model","ema","stage"]).agg(f1=("f1","mean"),recall=("recall","mean"),people_with_stage=("true_windows",lambda v:int((v>0).sum())),true_windows=("true_windows","sum"),predicted_windows=("predicted_windows","sum")).to_csv(o/"stage_summary.csv")
audit=dict(status="PASS",planned_people=20,scored_people=N,excluded_input=cohort["excluded"],method_streams_recomputed=N*234,normalizer_cases=N*6,maximum_gain_error=maxerr,protocol_and_source_hashes_verified=True,all_predictions_before_event_labels=True,within_person_seed_averaging=True,no_targettuning=True,no_gradientupdates=True,healthy10_usedfordevelopment_disclosed=True,windows=sum(v["full_signal_windows"] for v in la["records"]));(o/"audit.json").write_text(json.dumps(audit,indent=2))
tab=summary.reset_index();tab=tab[tab.ema==.5]
(o/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1400px;margin:35px auto}td,th{padding:6px;border:1px solid #ddd}</style><h1>ISRUC患者：固定20人计划的输入合格子集复核</h1><p>队列I021–I040的20人计划在读取事件标签前固定。I040因未能核实命名参考导联而不符合预定输入条件，在未读标签、未生成其预测前排除；不换人补数。主要分析为其余19人，完整排除理由单独锁定。全套预测完成后才下载评分标签；未按结果换人、换导联、换模型或调阈值。健康队列III的10人/8,889标签曾用于前期方法分析，开发暴露明确披露。此处是独立受试者的固定方案复核，不是梯度TTT或保留控制实验。</p>'+pd.DataFrame([primary]).to_html(index=False)+'<h2>同EMA0.5的全部条件</h2>'+tab.to_html(index=False)+'<h2>受试者配对变化</h2>'+comp[comp.ema==.5].to_html(index=False)+'<p>先按人平均三个种子指标，再跨人平均；区间为19位输入合格患者在固定种子和协议下的bootstrap，不涵盖全部选择不确定性。导联仍是侧方近似映射，不能与SleepEDF中线导联等同。既有SleepEDF中最差−12.34pp的幅值匹配伤害依然成立，不能因本队列改善而宣称通用安全。</p><p><a href="audit.json">独立指标审计</a> · <a href="protocol.json">标签前固定协议</a> · <a href="input_audit.json">单位与完整时间轴</a> · <a href="cohort_lock_after_input_audit.json">标签前技术排除记录</a> · <a href="label_audit.json">标签对齐</a> · <a href="../phase81/report.html">原人群风险</a></p>')
print(pd.DataFrame([primary]).to_string(index=False));print(tab.to_string(index=False));print(audit)
