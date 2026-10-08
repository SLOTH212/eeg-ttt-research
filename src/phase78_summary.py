from common import *
from phase2_temporal import ema
from scipy.signal import lfilter
import pandas as pd,hashlib
o=ROOT/"phase78";pr=json.loads((o/"protocol.json").read_text());assert (o/"complete.json").exists();audit=json.loads((o/"execution_audit.json").read_text());assert audit["status"]=="PASS";assert hashlib.sha256((o/"protocol.json").read_bytes()).hexdigest()==(o/"protocol.sha256").read_text()
ref=[]
for s in pr["source_reference_subjects"]:
 x=np.load(ROOT/("data/processed" if s<20 else "phase2/processed")/f"s{s:02d}_x.npy",mmap_mode="r");ref.append(np.median(np.sqrt(np.mean(x.astype(float)**2,axis=(1,2)))))
assert float(np.median(ref))==pr["source_reference_rms"]
cases=[("source",f"s{s:02d}","native") for s in pr["source_validation"]]+[("external",s,m) for s in pr["external_participants"] for m in pr["montages"]];rows=[];maxgainerr=0.;normalizer_cases=0
for domain,sid,montage in cases:
 if domain=="source":
  number=int(sid[1:]);folder=ROOT/("data/processed" if number<20 else "phase2/processed");y=np.load(folder/(sid+"_meta.npz"))["y"];raw=np.load(folder/(sid+"_x.npy"),mmap_mode="r")
 else:
  y=np.load(ROOT/"phase77/labels"/(sid+"_y.npy"));raw=np.load(ROOT/"phase76/processed"/(sid+"_"+montage+"_x.npy"),mmap_mode="r")
 for mode in ["raw"]+list(pr["normalizers"]):
  if mode!="raw":
   a=np.load(o/"predictions"/f"{domain}_{sid}_{montage}_{mode}_normalizer.npz");rms=np.sqrt(np.mean(raw.astype(float)**2,axis=(1,2)));assert np.array_equal(rms,a["current_rms"]);valid=rms>=1e-6;logs=np.log(rms[valid]);alpha=pr["normalizers"][mode];z,_=lfilter([alpha],[1,-(1-alpha)],logs,zi=[(1-alpha)*logs[0]]);gain=np.ones(len(raw));gain[valid]=np.clip(pr["source_reference_rms"]/np.exp(z),.1,100);err=float(abs(gain-a["gains"]).max());maxgainerr=max(maxgainerr,err);assert err<1e-10;assert abs(z-a["log_rms_state"][valid]).max()<1e-12;normalizer_cases+=1
  d=np.load(o/"predictions"/f"{domain}_{sid}_{montage}_{mode}.npz");assert len(d.files)==(12 if mode=="raw" else 9)
  for key in d.files:
   model,seed=key.rsplit("_seed",1);p=d[key];assert p.shape==(len(y),5) and np.isfinite(p).all() and abs(p.sum(1)-1).max()<2e-6
   if domain=="external" and mode=="raw" and model!="relative":assert np.array_equal(p,np.load(ROOT/"phase77/predictions"/f"{sid}_{montage}.npz")[key])
   for smooth in pr["ema"]:
    q=p if smooth==1 else ema(p,smooth);m=metrics(y,q);cm=np.array(m["confusion"]);rows.append(dict(domain=domain,subject=sid,montage=montage,mode=mode,model=model,seed=int(seed),ema=smooth,macro_f1=100*m["macro_f1"],accuracy=100*cm.trace()/cm.sum(),ce=m["ce"],balanced_accuracy=100*m["balanced_accuracy"],kappa=m["kappa"]))
df=pd.DataFrame(rows);assert len(df)==2160;df.to_csv(o/"participants.csv",index=False);person=df.groupby(["domain","subject","montage","mode","model","ema"])[["macro_f1","accuracy","ce","balanced_accuracy","kappa"]].mean().reset_index();person.to_csv(o/"seedmean_people.csv",index=False);summary=person.groupby(["domain","montage","mode","model","ema"])[["macro_f1","accuracy","balanced_accuracy","ce"]].mean();summary.to_csv(o/"summary.csv")
rng=np.random.default_rng(78017);comp=[]
for domain,montage in [("source","native"),("external","left"),("external","right")]:
 for model in ["cbramod","labram","context11"]:
  for smooth in pr["ema"]:
   p=person[(person.domain==domain)&(person.montage==montage)&(person.model==model)&(person.ema==smooth)].pivot(index="subject",columns="mode",values="macro_f1");idx=rng.integers(0,len(p),size=(10000,len(p)))
   for mode in pr["normalizers"]:
    delta=(p[mode]-p.raw).to_numpy();ci=np.quantile(delta[idx].mean(1),[.025,.975]);comp.append(dict(domain=domain,montage=montage,model=model,ema=smooth,mode=mode,raw_f1=float(p.raw.mean()),normalized_f1=float(p[mode].mean()),difference_pp=float(delta.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]),positive_people=int((delta>0).sum()),harm_over_1pp=int((delta< -1).sum()),people=len(delta)))
comp=pd.DataFrame(comp);comp.to_csv(o/"paired_comparisons.csv",index=False);primary=comp[(comp.ema==.5)&(comp["mode"]=="logema01")];primary.to_csv(o/"primary_diagnostics.csv",index=False);result=dict(status="PASS",method_streams_recomputed=2160,normalizer_cases_independently_recomputed=normalizer_cases,maximum_gain_error_vs_scipy_filter=maxgainerr,source_reference_exact=True,raw_external_predictions_reused_exact=True,no_model_updates=True,posthoc_external_diagnostic=True);(o/"audit.json").write_text(json.dumps(result,indent=2))
tab=summary.reset_index();tab=tab[tab.ema==.5]
(o/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1300px;margin:35px auto}td,th{padding:6px;border:1px solid #ddd}</style><h1>无梯度幅值校准的机制检查</h1><p>这是在看到Phase77外部结果后开展的探索性诊断。原始静态迁移失败结果不变。所有模型、分类器与归一化层参数保持冻结；额外状态仅一个因果log RMS估计值，每条记录重置。</p><p>源10人波形的逐人窗口RMS中位数再取中位数，得到18.847微伏参考，不读取源类别标签。共同增益保留通道幅值比例，范围[0.1,100]。对照当前窗口估计和因果EMA0.01估计；二者都只使用已经到达的窗口。原有相对频谱26维模型作为幅值不敏感参考。无新增训练。</p>'+primary.to_html(index=False)+'<h2>同EMA0.5的所有条件</h2>'+tab.to_html(index=False)+'<p>每位受试者先平均三个种子指标；区间为条件于已观察种子与源选择的按人bootstrap。源验证仅4人，外部10人；这些结果不能改称盲测，不能推断输入校准等于参数学习或保留控制成功。导联近似映射和跨人群差异仍然存在。</p><p><a href="audit.json">独立审计</a> · <a href="protocol.json">固定事后诊断方案</a> · <a href="paired_comparisons.csv">全部配对结果</a> · <a href="../phase77/report.html">原始静态结果</a></p>')
print(primary.to_string(index=False));print(tab.to_string(index=False));print(result)
