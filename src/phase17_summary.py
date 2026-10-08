from pathlib import Path
import json,numpy as np,pandas as pd
r=Path(__file__).resolve().parents[1];out=r/"phase17";rows=[];people=[]
for f in out.glob("*/complete.json"):
 d=json.loads(f.read_text());assert len(d["results"])==4
 for v in d["results"]:people.append(dict(source=d["source"],objective=d["objective"],lr=d["lr"],subject=v["subject"],frozen=v["frozen"]["macro_f1"],adapted=v["adapted"]["macro_f1"],weak_ema=v["weak_ema"]["macro_f1"]))
 delta=np.array([v["adapted"]["macro_f1"]-v["frozen"]["macro_f1"] for v in d["results"]]);rows.append(dict(source=d["source"],objective=d["objective"],lr=d["lr"],n=4,frozen=100*d["mean_frozen"],adapted=100*d["mean_adapted"],weak_ema=100*np.mean([v["weak_ema"]["macro_f1"] for v in d["results"]]),delta_pp=100*delta.mean(),harm_n=int((delta<-.01).sum()),worst_delta_pp=100*delta.min()))
s=pd.DataFrame(rows).sort_values(["source","objective","lr"]);s.to_csv(out/"summary.csv",index=False);pd.DataFrame(people).to_csv(out/"participant_metrics.csv",index=False);selections=[]
if len(s)==20:
 for (source,obj),g in s.groupby(["source","objective"]):
  best=g.loc[g.adapted.idxmax()];frozen=float(best.frozen);selections.append(dict(source=source,objective=obj,lr=0. if best.adapted<=frozen else float(best.lr),best_positive_lr=float(best.lr),frozen=frozen,best_positive_f1=float(best.adapted),selected_f1=max(frozen,float(best.adapted)),weak_ema=float(best.weak_ema)))
 (out/"source_selection.json").write_text(json.dumps(dict(source_validation_only=True,selection_rule="MeanfourpersonF1withfrozenfallback;notatestestimate",selections=selections),indent=2))
style="<style>body{max-width:1100px;margin:40px auto;font:16px/1.7 system-ui;padding:0 20px;color:#234}table{border-collapse:collapse;font-size:14px}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:right}.notice{background:#fff1df;padding:18px}</style>"
html="<!doctype html><html lang=zh><meta charset=utf-8><title>源域连续更新筛选</title>"+style+"<body><h1>源域连续更新筛选</h1><p class=notice>已完成 "+str(len(s))+" / 20 组。仅为源验证选择表现，不是目标测试成绩，也不作为泛化效应估计。</p><p>四位源验证人员：5、15、25、35。LaBraM pretrained400 的既有 LayerNorm 仿射参数可写，其他编码器权重与分类头固定。每个当前30秒窗口先预测再更新；每人从源参数开始。比较熵及固定分词器的掩码辅助目标，学习率含后续源验证驱动扩展1e−6、1e−5，最终选择包含冻结回退。</p>"+s.to_html(index=False,float_format=lambda x:f"{x:.6f}",border=0)+"<p>大步长熵更新可出现几乎全部预测为清醒的塌缩，同时自身熵继续下降。源域单步有益写入不保证累积运行有效。小步长微弱收益仍需对照轻度平滑。</p><p>完整逐窗口冻结审计：2种来源×4位人员，分类结果与批量缓存一致，0处argmax差异。</p><p><a href=frozen_audit.json>冻结流审计</a> · <a href=protocol.json>协议</a> · <a href=../phase16/report.html>单步机制诊断</a> · <a href=../research_overview.html>总览</a></p></body></html>"
(out/"report.html").write_text(html);print("Complete source-only cases",len(s));print(json.dumps(selections,indent=2))
