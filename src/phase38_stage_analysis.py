from common import *
from phase2_temporal import ema
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
o=ROOT/"phase38";rows=[];timing=[]
for seed in [17,29,43]:
 parent=ROOT/"phase28/young_joint_0" if seed==17 else ROOT/"phase33"/f"young_joint_0_seed{seed}";case=ROOT/"phase37/young_fitted_0.01" if seed==17 else o/"evaluation"/f"young_fitted_0.01_seed{seed}"
 for sid in [5,15,25,35]:
  z=np.load(case/f"s{sid:02d}.npz");base=np.load(parent/f"s{sid:02d}.npz");y=z["y"];valid=y>=0;assert np.array_equal(y,base["y"]);p=base["p"];q=z["p"]
  for smoothing in ["raw","EMA0.8"]:
   a=p if smoothing=="raw" else ema(p,.8);b=q if smoothing=="raw" else ema(q,.8);aa=a.argmax(1);bb=b.argmax(1);ap,ar,af,support=precision_recall_fscore_support(y[valid],aa[valid],labels=range(5),zero_division=0);bp,br,bf,_=precision_recall_fscore_support(y[valid],bb[valid],labels=range(5),zero_division=0)
   for c,name in enumerate(["W","N1","N2","N3","REM"]):
    mask=y==c;rows.append(dict(classifier_seed=seed,subject=sid,smoothing=smoothing,stage=name,n=int(mask.sum()),f1_delta_pp=100*(bf[c]-af[c]),recall_delta_pp=100*(br[c]-ar[c]),precision_delta_pp=100*(bp[c]-ap[c]),changed=int((aa[mask]!=bb[mask]).sum()),corrected=int(((aa[mask]!=y[mask])&(bb[mask]==y[mask])).sum()),harmed=int(((aa[mask]==y[mask])&(bb[mask]!=y[mask])).sum())))
   for quarter,ii in enumerate(np.array_split(np.arange(len(y)),4),1):
    vv=ii[y[ii]>=0];timing.append(dict(classifier_seed=seed,subject=sid,smoothing=smoothing,quarter=quarter,n=len(vv),accuracy_gain_pp=100*np.mean((bb[vv]==y[vv]).astype(float)-(aa[vv]==y[vv])),baseline_predicted_wake_fraction=np.mean(aa[vv]==0),auxiliary_ce=float(z["loss"][ii].mean()),gradient_norm=float(z["gradient_norm"][ii].mean())))
d=pd.DataFrame(rows);d.to_csv(o/"stage_participants.csv",index=False);summary=d.groupby(["classifier_seed","smoothing","stage"])[["f1_delta_pp","recall_delta_pp","precision_delta_pp"]].mean().reset_index();summary.to_csv(o/"stage_summary.csv",index=False);counts=d.groupby(["classifier_seed","smoothing","stage"])[["n","changed","corrected","harmed"]].sum().reset_index();counts.to_csv(o/"stage_counts.csv",index=False);pd.DataFrame(timing).to_csv(o/"time_quarters.csv",index=False)
style='<style>body{font:15px/1.65 system-ui;max-width:1200px;margin:30px auto}td,th{padding:5px;border-bottom:1px solid #ddd}</style>';(o/"stage_analysis.html").write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'+style+'<h1>三种子源复现：事后分睡眠阶段错误分析</h1><p>阶段38同一冻结分类器与拟合辅助头连续更新LR0.01，原始预测及同EMA0.8分别比较。以下是全部实验完成后的解释性分析，没有用于改选步长或模型。每种子包含同四人；各阶段F1、召回及精确率按人等权，完整改对/改错计数另存CSV。</p>'+summary.to_html(index=False,float_format=lambda x:f"{x:.4f}")+'<p>按记录四等分的时间统计仅描述变化，不能证明更新次数或某一阶段造成损害。</p><a href="stage_counts.csv">分类改变计数</a> · <a href="stage_participants.csv">逐人各阶段</a> · <a href="time_quarters.csv">时间分布</a></html>');print(summary[summary.smoothing=="EMA0.8"].to_string(index=False))
