from common import *
import pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from html import escape
rows=[];diagnostics=[];cv=[];stress=[];selections=[]
for directory in sorted((ROOT/'results').glob('*_*_*')):
 if not (directory/'summary.json').exists():continue
 model,fold,seed=directory.name.split('_');meta={'model':model,'fold':fold,'seed':int(seed)}
 rows.extend([{**meta,**r} for r in json.loads((directory/'evaluation.json').read_text()) if r['method']!='causal_bn'])
 if (directory/'supplement.json').exists():rows.extend([{**meta,**r} for r in json.loads((directory/'supplement.json').read_text())['results']])
 if (directory/'lr_sensitivity.json').exists():rows.extend([{**meta,**r} for r in json.loads((directory/'lr_sensitivity.json').read_text())['results']])
 if (directory/'identity_control.json').exists():rows.extend([{**meta,**r} for r in json.loads((directory/'identity_control.json').read_text())['results']])
 diagnostics.extend([{**meta,**r} for r in json.loads((directory/'evaluation_diagnostics.json').read_text())])
 cv.extend([{**meta,**r} for r in json.loads((directory/'controller_cv.json').read_text())])
 selections.append({**meta,**json.loads((directory/'selection.json').read_text())})
 if (directory/'stress.json').exists():stress.extend([{**meta,**r} for r in json.loads((directory/'stress.json').read_text())['results']])
df=pd.DataFrame(rows);df.to_csv(ROOT/'results/all_subject_metrics.csv',index=False)
base=df[df.method=='frozen'][['model','fold','seed','subject','macro_f1']].rename(columns={'macro_f1':'frozen_f1'})
df=df.merge(base,on=['model','fold','seed','subject']);df['delta_pp']=100*(df.macro_f1-df.frozen_f1)
# Seeds first averaged within participant; never pretend 3 seeds are 3 participants.
sub=df.groupby(['model','fold','subject','method'],as_index=False)[['macro_f1','delta_pp']].mean();sub.to_csv(ROOT/'results/seed_averaged_subject_metrics.csv',index=False)
rng=np.random.default_rng(81023);summary=[]
for (model,method),g in sub.groupby(['model','method']):
 values=g.delta_pp.to_numpy();boots=np.array([rng.choice(values,len(values),replace=True).mean() for _ in range(4000)])
 summary.append({'model':model,'method':method,'subjects':len(g),'macro_f1_pct':g.macro_f1.mean()*100,'delta_pp':values.mean(),'descriptive_ci_low':np.quantile(boots,.025),'descriptive_ci_high':np.quantile(boots,.975),'negative_subject_rate_1pp':float((values<-1).mean()),'worst_subject_delta_pp':float(values.min())})
summ=pd.DataFrame(summary);summ.to_csv(ROOT/'results/summary.csv',index=False)
diag=pd.DataFrame(diagnostics);diags=[]
for (model,fold,seed,subject,h),g in diag.groupby(['model','fold','seed','subject','h']):
 gain=g.write_gain.to_numpy();reset=g.reset_gain.to_numpy()
 diags.append({'model':model,'fold':fold,'seed':seed,'subject':subject,'horizon':h,'n':len(g),'mean_write_gain':gain.mean(),'positive_write_fraction_1e6':(gain>1e-6).mean(),'positive_write_fraction_1e4':(gain>1e-4).mean(),'positive_write_fraction_1e3':(gain>1e-3).mean(),'keep_write_oracle_gain':np.maximum(0,gain).mean(),'keep_write_reset_oracle_gain':np.maximum.reduce([np.zeros(len(gain)),gain,reset]).mean(),'mean_reset_gain':reset.mean(),'write_harm_at_transition':g.loc[g.transition,'write_gain'].mean(),'write_harm_interior':g.loc[~g.transition,'write_gain'].mean()})
diagnostic=pd.DataFrame(diags);diagnostic.to_csv(ROOT/'results/diagnostic_summary.csv',index=False);pd.DataFrame(cv).to_csv(ROOT/'results/controller_loso.csv',index=False)
if stress:pd.DataFrame(stress).to_csv(ROOT/'results/stress_summary.csv',index=False)
# Offline stage-transition scoring uses labels only for analysis, never online actions.
stage_rows=[]
for directory in sorted((ROOT/'results').glob('*_*_*')):
 if not (directory/'summary.json').exists():continue
 model,fold,seed=directory.name.split('_')
 for p in directory.glob('predictions_s*.npz'):
  sid=int(p.stem.split('_s')[1]);d=dict(np.load(p));y=d.pop('y');transition=np.zeros(len(y),dtype=bool)
  boundaries=np.where((y[1:]!=y[:-1])&(y[1:]>=0)&(y[:-1]>=0))[0]+1
  for b in boundaries:transition[max(0,b-2):min(len(y),b+3)]=True
  supplement=directory/f'supplement_predictions_s{sid:02d}.npz'
  if supplement.exists():d.update({k:v for k,v in np.load(supplement).items() if k!='y'})
  lrpred=directory/f'lr_predictions_s{sid:02d}.npz'
  if lrpred.exists():d.update({k:v for k,v in np.load(lrpred).items() if k!='y'})
  identitypred=directory/f'identity_predictions_s{sid:02d}.npz'
  if identitypred.exists():d.update({k:v for k,v in np.load(identitypred).items() if k!='y'})
  for method,probs in d.items():
   if method=='causal_bn':continue
   for label,mask in [('near_transition',transition),('interior',~transition)]:
    stage_rows.append({'model':model,'fold':fold,'seed':int(seed),'subject':sid,'method':method,'region':label,**metrics(y[mask],probs[mask])})
pd.DataFrame(stage_rows).to_csv(ROOT/'results/stage_transition_metrics.csv',index=False)
methods=['frozen','smooth_tuned','extended_lr_continuous','retuned_future_controller','identity_future','identity_confidence','tiny_lr','periodic','reset_periodic','continuous','low_entropy','low_gradient','matched_random','confidence_controller','current_controller','future_controller','refined_controller','no_reset_controller','t3a','causal_bn_corrected']
models=sorted(summ.model.unique());fig,axs=plt.subplots((len(models)+1)//2,2,figsize=(16,8*((len(models)+1)//2)),squeeze=False)
for ax,model in zip(axs.ravel(),models):
 g=summ[summ.model==model].set_index('method').reindex([v for v in methods if v not in ['t3a','causal_bn_corrected']]).dropna(subset=['delta_pp']);yy=np.arange(len(g));ax.barh(yy,g.delta_pp,color=['#268a80' if v>=0 else '#c65b52' for v in g.delta_pp]);ax.set_yticks(yy,g.index);ax.invert_yaxis();ax.axvline(0,color='black',lw=.8);ax.set_xlabel('Macro-F1 change vs frozen (percentage points)');ax.set_title(model+' | mean over participants and seeds');ax.grid(axis='x',alpha=.2)
for ax in axs.ravel()[len(models):]:ax.set_visible(False)
fig.tight_layout();fig.savefig(ROOT/'results/method_comparison.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12,4))
g=diagnostic[diagnostic.horizon==4].groupby('model')[['keep_write_oracle_gain','keep_write_reset_oracle_gain']].mean();g.plot.bar(ax=axs[0]);axs[0].set_ylabel('Future cross-entropy decrease (H=4)');axs[0].set_title('Local hindsight gain: isolate RESET contribution');axs[0].tick_params(axis='x',rotation=0)
g=diagnostic.groupby(['model','horizon']).positive_write_fraction_1e4.mean().unstack(0);g.plot(marker='o',ax=axs[1]);axs[1].set_ylabel('Fraction of writes with future CE gain > 0.0001');axs[1].set_title('Horizon sensitivity');fig.tight_layout();fig.savefig(ROOT/'results/diagnostics.png',dpi=160);plt.close(fig)
policy=[]
for directory in sorted((ROOT/'results').glob('*_*_*')):
 if (directory/'policy_diagnostics.json').exists():
  model,fold,seed=directory.name.split('_');policy.extend([{'model':model,'fold':fold,'seed':int(seed),**r} for r in json.loads((directory/'policy_diagnostics.json').read_text())['summary']])
if policy:pd.DataFrame(policy).to_csv(ROOT/'results/policy_diagnostic_summary.csv',index=False)
# Compact Chinese HTML report, updated as runs finish; counts make incompleteness explicit.
complete=len(list((ROOT/'results').glob('*/summary.json')));suppcount=len(list((ROOT/'results').glob('*/supplement.json')))
table=summ.round(4).to_html(index=False,classes='metrics')
html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>EEG TTT CPU 探索报告</title><style>body{font:16px/1.65 system-ui;max-width:1250px;margin:35px auto;padding:0 25px;color:#1c2b36}h1,h2{line-height:1.25}table{border-collapse:collapse;font-size:13px}td,th{padding:6px;border:1px solid #ddd}img{max-width:100%}.note{background:#f0f4f5;padding:16px}code{background:#eee}</style>'''
html+=f'<h1>EEG TTT：CPU 可行性探索</h1><p>核心与敏感性实验完成 {complete}/24 组；补充实验完成 {suppcount}/24 组。每组为一个载体 × 一个划分 × 一个种子。CPU 限定 8–23，GPU 禁用。</p>'
html+='<p class="note">这是小样本方法可行性实验。共 20 名受试者，每人一晚；每折 10 人训练、2 人验证、4 人训练控制器、4 人评估。两折评估受试者合计 8 人。三个种子先在同一受试者内平均，不能当作 24 个独立受试者。描述性区间不构成正式显著性或泛化保证。</p>'
html+='<h2>比较结果</h2><p>delta_pp 是相对冻结模型的 Macro-F1 百分点变化。固定五类，缺失类别 F1=0；不按真实睡眠位置裁剪。旧版因果 BN 实现被排除，表中仅保留修正版本。</p>'+table+'<p>图中聚焦更新策略与平滑；T3A 和因果 BN 的大幅退化完整保留在上表。</p><img src="method_comparison.png">'
html+='<h2>机制诊断</h2><p>KEEP、WRITE、RESET 从同一状态分叉；随后暂停更新，评估未来 1、4、10 个窗口。写入正收益与重置收益必须分开；后见 oracle 使用未来标签，仅能用于诊断。</p><img src="diagnostics.png">'
html+='<h2>实施与边界</h2><ul><li>所有在线动作只用当前及历史信号，当前预测先发布、随后才更新；标签仅用于源域训练和离线评分。</li><li>EEGNet 风格 CNN 是明确修改的轻量载体：51 点卷积、末端自适应平均池化，并非原论文睡眠性能复现。频谱 BN-MLP 是额外低成本探针；spectralhead / eegnethead 为相同源模型仅更新末端线性层的 V2 敏感性分析，单独报告。</li><li>控制器使用标准化线性 Ridge 回归；额外加入当前收益与置信度控制器、去掉 RESET、匹配频率的随机动作以及一轮源域策略回放。</li><li>T3A 改成先预测后加入支持集；因果 BN 是简单统计刷新基线，不能冒充完整成熟 TTA 方法。</li><li>合成干扰按固定时间位置施加；不是临床分布。外部数据集、基础模型及完整成熟方法对照仍未验证。</li><li>主机重启中断记录已保存；完成标记以前的中间模型不作为最终结果。</li></ul>'
html+='<p>数据来源：<a href="https://physionet.org/content/sleep-edfx/1.0.0/">PhysioNet Sleep-EDF Expanded</a>。架构参考：<a href="https://github.com/vlawhern/arl-eegmodels">EEGNet 官方实现</a>。全部配置、逐受试者指标、预测与运行日志保存在本地项目。</p></html>'
(ROOT/'results/report.html').write_text(html);print(summ.to_string(index=False));print('Report',ROOT/'results/report.html')
