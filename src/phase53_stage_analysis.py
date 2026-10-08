from common import *
import pandas as pd
o=ROOT/"phase53";rows=[]
methods={"frozen":ROOT/"phase28/mixed_joint_0"}
for v in ["initial","future","current","reconstruction"]:
 for t in ["continuous","reset64"]:methods[v+"_"+t]=o/(v+"_"+t)
methods.update(static_current=o/"static_current",static_future=o/"static_future")
for name,folder in methods.items():
 for sid in [5,15,25,35]:
  z=np.load(folder/f"s{sid:02d}.npz");y=z["y"];p=z["predictions"] if "predictions" in z else z["p"];valid=y>=0;truth=y[valid];pred=p[valid].argmax(1);cm=confusion_matrix(truth,pred,labels=range(5));tp=cm.diagonal();precision=tp/np.maximum(cm.sum(0),1);recall=tp/np.maximum(cm.sum(1),1);f1=2*precision*recall/np.maximum(precision+recall,1e-12)
  for c,stage in enumerate(["W","N1","N2","N3","REM"]):rows.append(dict(method=name,subject=sid,stage=stage,windows=int(cm[c].sum()),precision=precision[c]*100,recall=recall[c]*100,f1=f1[c]*100))
d=pd.DataFrame(rows);d.to_csv(o/"stage_people.csv",index=False);summary=d.groupby(["method","stage"])[["precision","recall","f1"]].mean();summary.to_csv(o/"stage_summary.csv");base=d[d.method=="frozen"].set_index(["subject","stage"])[["precision","recall","f1"]];delta=[]
for method,g in d.groupby("method"):
 a=g.set_index(["subject","stage"])[["precision","recall","f1"]]-base;a["method"]=method;delta.append(a.reset_index())
dd=pd.concat(delta);ds=dd.groupby(["method","stage"])[["precision","recall","f1"]].mean();ds.to_csv(o/"stage_delta_vs_frozen.csv");html='<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{font:15px/1.7 system-ui;max-width:1200px;margin:30px auto}td,th{padding:5px;border-bottom:1px solid #ddd}</style><h1>元辅助头连续更新的逐阶段变化</h1><p>阶段53完成后的离线诊断，四名源验证人等权，五类固定；不用于重新选择阈值、重置位置或学习率。下表为精确完整时间轴的各阶段精确率/召回/F1相对冻结之差，单位百分点。CE改善和MacroF1改善可能分离；不能将此表作为独立新验证集。</p>'+ds.to_html(float_format=lambda v:f"{v:.4f}")+'</html>';(o/"stage_analysis.html").write_text(html);print(ds.loc[["current_continuous","current_reset64","future_reset64","static_current"]].to_string())
