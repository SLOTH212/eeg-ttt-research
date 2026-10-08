from common import *
from phase2_temporal import ema
import pandas as pd,hashlib,warnings
warnings.filterwarnings("ignore",message="y_pred contains classes not in y_true")
out=ROOT/"phase27";rows=[];checks=[]
files=[(17,f) for f in (ROOT/"phase21/evaluation").glob("*_pretrained/evaluation.json")]
files += [(int(f.parent.name.rsplit("seed",1)[1]),f) for f in (out/"evaluation").glob("*/evaluation.json")]
for seed,f in files:
 d=json.loads(f.read_text());sel=d["selection"];source=d["source"];root=ROOT/("phase21" if seed==17 else "phase27");path=root/sel["selected_directory"]/"model.pt"
 assert sel["selection"]["best_val_f1"]==max(c["best_val_f1"] for c in sel["candidates"])
 if seed!=17:assert hashlib.sha256(path.read_bytes()).hexdigest()==sel["checkpoint_sha256"]
 ck=torch.load(path,map_location="cpu",weights_only=False);assert ck["best_val_f1"]==max([ck["initial_val_f1"]]+[h["val_f1"] for h in ck["history"]])
 for rr in d["results"]:
  sid=rr["subject"];saved=np.load(f.parent/f"s{sid:02d}.npz");reference=np.load(ROOT/"phase18"/f"{source}_pretrained"/f"s{sid:02d}.npz");y=saved["y"];p=saved["source_tuned"];assert np.array_equal(y,reference["y"]);assert np.isfinite(p).all() and np.max(abs(p.sum(1)-1))<1e-5;assert abs(metrics(y,p)["macro_f1"]-rr["macro_f1"])<1e-12
  transition=np.zeros(len(y),bool)
  for b in np.flatnonzero((y[1:]!=y[:-1])&(y[1:]>=0)&(y[:-1]>=0))+1:transition[max(0,b-2):min(len(y),b+3)]=True
  for method,pr in [("full",p),("full_ema08",ema(p,.8)),("frozen",reference["frozen"]),("frozen_ema08",ema(reference["frozen"],.8))]:
   rows.append(dict(source=source,seed=seed,subject=sid,cohort=rr["cohort"],method=method,f1=metrics(y,pr)["macro_f1"],transition=metrics(y[transition],pr[transition])["macro_f1"]))
  checks.append(dict(source=source,seed=seed,subject=sid))
df=pd.DataFrame(rows);df.to_csv(out/"participant_metrics.csv",index=False);seedmeans=df.groupby(["source","seed","cohort","method"]).f1.mean().mul(100).reset_index();seedmeans.to_csv(out/"seed_means.csv",index=False)
summary=seedmeans.groupby(["source","cohort","method"]).f1.agg(["mean","std","min","max","count"]).reset_index();summary.to_csv(out/"summary.csv",index=False)
a=pd.read_csv(ROOT/"phase4/all_metrics.csv");a["cohort"]=a.cohort.map({"young":"young_dev","middle":"middle_dev"});b=pd.read_csv(ROOT/"phase6/all_metrics.csv");b["cohort"]="older";old=pd.concat([a,b],ignore_index=True);contrasts=[]
def contrast(source,cohort,method,name,delta,base,ref,nseeds):
 rng=np.random.default_rng(20261006);ci=np.quantile(rng.choice(delta,(20000,len(delta)),replace=True).mean(1),[.025,.975]);return dict(source=source,cohort=cohort,method=method,reference=name,n_people=len(delta),n_training_seeds=nseeds,f1=100*base.mean(),reference_f1=100*ref.mean(),delta_pp=delta.mean(),ci_low=ci[0],ci_high=ci[1],harm_over_1pp=int((delta< -1).sum()),worst_delta_pp=delta.min())
for (source,cohort,method),g in df[df.method.isin(["full","full_ema08"])].groupby(["source","cohort","method"]):
 seeds=sorted(g.seed.unique());carrier=g.groupby("subject").f1.mean()
 frozen=df[(df.source==source)&(df.cohort==cohort)&(df.method==("frozen" if method=="full" else "frozen_ema08"))].groupby("subject").f1.mean().reindex(carrier.index)
 contrasts.append(contrast(source,cohort,method,"same_initial_frozen",100*(carrier-frozen).to_numpy(),carrier,frozen,len(seeds)))
 for variant in ["full52","wide","context11"]:
  for control in ["frozen","ema_08"]:
   ref=old[(old.source==source)&(old.cohort==cohort)&(old.variant==variant)&(old.method==control)&old.seed.isin(seeds)].groupby("subject").f1.mean().reindex(carrier.index);assert not ref.isna().any()
   contrasts.append(contrast(source,cohort,method,variant+"_"+control,100*(carrier-ref).to_numpy(),carrier,ref,len(seeds)))
comp=pd.DataFrame(contrasts);comp.to_csv(out/"paired_comparisons.csv",index=False)
complete=len(files)==6;audit=dict(status="PASS",complete=complete,completed_carriers=len(files),audited_prediction_streams=len(checks),new_prediction_streams=sum(x["seed"]!=17 for x in checks),checks="saved labels, normalized finite probabilities, exactF1 recomputation, source LR/epochchoice, selected checkpoint hash",bootstrap="Average seed-specific F1 per person before resampling people; no seed pseudoreplication",details=checks);(out/"audit.json").write_text(json.dumps(audit,indent=2))
selected=comp[(comp.method=="full_ema08")&(comp.reference=="context11_ema_08")]
html="""<!doctype html><html lang="zh"><meta charset="utf-8"><title>完整编码器随机种子复现</title><style>body{font:16px/1.6 system-ui;max-width:1250px;margin:30px auto}td,th{padding:5px;border-bottom:1px solid #ccc}</style><h1>完整编码器：三种子复现</h1><p>相同预训练权重、源拟合分类头与数据划分，只改变训练随机数17/29/43。每个种子独立用源验证选择学习率和epoch，含epoch0回退。当前人群均已观察过，结果为探索性；不是在线TTT有效性证明，也不是三种子的预训练对随机初始化比较。</p><p>每种子先算个人Macro-F1，再求个人种子均值，bootstrap按人重采样；表中标准差仅反映三个训练种子的均值波动，不是总体置信区间。</p>"""+f"<p>完成模型数：{len(files)}/6；审计预测流：{len(checks)}。</p>"+summary.to_html(index=False,float_format=lambda x:f"{x:.3f}")+"""<h2>相同来源、人员与种子的历史强基线</h2><p>完整编码器加alpha=.8因果EMA，对照11窗口频谱模型加同样EMA。后者有5分钟历史，比较实际可用表现，不能隔离架构或预训练贡献。两者历史调参预算不同。</p>"""+selected.to_html(index=False,float_format=lambda x:f"{x:.3f}")+"""<p><a href="paired_comparisons.csv">全部预定义比较</a> · <a href="seed_means.csv">各随机种子结果</a> · <a href="audit.json">审计</a></p></html>""";(out/"report.html").write_text(html)
print(seedmeans[seedmeans.method=="full"].to_string(index=False));print(selected.to_string(index=False));print("AUDIT",len(checks),"COMPLETE",complete)
