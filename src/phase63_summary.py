from common import *
from phase2_temporal import ema
import pickle,pandas as pd,sys,hashlib
OUT=ROOT/"phase63";pr=json.loads((OUT/"protocol.json").read_text());assert (OUT/"probe_complete.json").exists()
rows=[];maximum=0.;files=0
for kind in ["pretrained","random"]:
 with (OUT/("probe_"+kind)/"model.pkl").open("rb") as f:carrier=pickle.load(f)
 selection=json.loads((OUT/("probe_"+kind)/"selection.json").read_text())
 best=max(selection["history"],key=lambda a:a["val_macro_f1"])
 assert best["C"]==selection["C"]==carrier["model"].C and best["val_macro_f1"]==selection["val_f1"]
 assert selection["training_subjects"]==pr["training_subjects"] and selection["validation_subjects"]==pr["validation_subjects"]
 train=[]
 for sid in pr["training_subjects"]+pr["validation_subjects"]:
  info=json.loads((OUT/"features"/f"s{sid:02d}_{kind}.json").read_text());z=np.load(OUT/"features"/f"s{sid:02d}_{kind}.npz")["features"]
  folder=ROOT/("data/processed" if sid<20 else "phase2/processed");y=np.load(folder/f"s{sid:02d}_meta.npz")["y"]
  assert z.shape==(len(y),400) and np.isfinite(z).all() and info["encoder_state_unchanged"]
  assert info["audit"]["other_sample_error"]<2e-5 and info["audit"]["perwindow_preprocessing_exact"];files+=1
  if sid in pr["training_subjects"]:train.append(z[y>=0])
  else:
   expected=carrier["model"].predict_proba(carrier["scaler"].transform(z));saved=np.load(OUT/("probe_"+kind)/f"s{sid:02d}.npz")
   assert np.array_equal(y,saved["y"]);err=float(abs(expected-saved["p"]).max());maximum=max(maximum,err);assert err==0
   for smooth in [1.,.8,.5]:
    m=metrics(y,expected if smooth==1 else ema(expected,smooth));rows.append(dict(subject=sid,model="cbramod_"+kind,ema=smooth,macro_f1=100*m["macro_f1"]))
 x=np.concatenate(train).astype(np.float64);assert np.allclose(x.mean(0),carrier["scaler"].mean_,atol=1e-10,rtol=1e-10);assert np.allclose(x.var(0),carrier["scaler"].var_,atol=1e-10,rtol=1e-10)
 pretrained=torch.load(OUT/(kind+"_encoder_state.pt"),map_location="cpu",weights_only=True)
 if kind=="pretrained":
  official=torch.load(ROOT/"phase62/pretrained_weights.pth",map_location="cpu",weights_only=True);assert all(torch.equal(v,official[n]) for n,v in pretrained.items())
df=pd.read_csv(OUT/"participants.csv");assert len(df)==60
for r in rows:
 saved=df[(df.subject==r["subject"])&(df.model==r["model"])&(df.ema==r["ema"])].iloc[0];assert abs(saved.macro_f1-r["macro_f1"])<1e-10
summary=df.groupby(["model","ema"])[["macro_f1","balanced_accuracy","cross_entropy"]].mean();comparisons=[]
for smooth in [1.,.8,.5]:
 primary=summary.loc[("cbramod_pretrained",smooth),"macro_f1"]
 for reference in ["cbramod_random","labram_frozen_pretrained","labram_source_finetuned_reference"]:
  rr=summary.loc[(reference,smooth),"macro_f1"];a=df[(df.model=="cbramod_pretrained")&(df.ema==smooth)].set_index("subject").macro_f1;b=df[(df.model==reference)&(df.ema==smooth)].set_index("subject").macro_f1
  comparisons.append(dict(ema=smooth,reference=reference,difference_pp=primary-rr,positive_people=int((a>b).sum())))
cmp=pd.DataFrame(comparisons);cmp.to_csv(OUT/"comparisons.csv",index=False)
audit=dict(status="PASS",feature_files=files,source_people=14,validation_people=4,method_streams=60,prediction_replay_error=maximum,scalers_train_only_recomputed=True,source_selection_argmax_verified=True,pretrained_state_exactly_official=True,no_target_features=True)
(OUT/"audit.json").write_text(json.dumps(audit,indent=2))
html='<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{font:15px/1.7 system-ui;max-width:1200px;margin:30px auto}td,th{padding:5px;border-bottom:1px solid #ddd}</style><h1>CBraMod 冻结分类表征源侧检查</h1><p>使用原生两导联，每个30秒窗口内对各导联30个patch平均，形成400维特征。官方预训练模型与seed17随机初始化模型均冻结；相同十名混合来源训练者、四名验证者，线性头正则化网格和按人/类别加权沿用阶段14。输入采用既有0.1–45Hz每窗预处理。全部只在源侧，没有目标人群提取或自动晋级。</p><p>LaBraM冻结400维探针沿用相同来源和正则化选择规则；两种模型的预训练资料、架构及池化含义不同，不能把差异单独归因于架构。来源微调LaBraM另列参考，其训练量与冻结探针不同。</p>'+summary.to_html(float_format=lambda x:f"{x:.5f}")+'<h2>相同输出EMA比较</h2>'+cmp.to_html(index=False,float_format=lambda x:f"{x:.5f}")+'<p>阶段62重建主检查失败仍保留。本阶段仅检验分类表征，不代表辅助损失或测试时更新有效。所有源验证者已多轮使用，结果属探索性。</p><p><a href="protocol.json">协议</a> · <a href="audit.json">审计</a> · <a href="probe_pretrained/selection.json">预训练探针选择</a> · <a href="probe_random/selection.json">随机探针选择</a></p></html>'
(OUT/"report.html").write_text(html);print(summary.to_string());print(cmp.to_string(index=False));print(audit)
