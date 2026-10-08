from pathlib import Path
import pandas as pd,json,html
r=Path(__file__).resolve().parents[1]
style="<style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.7 system-ui;color:#203047}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:right}th{background:#eef3f8}.notice{background:#fff0df;padding:20px;border-left:5px solid #b65e19}a{color:#1463a0}</style>"
def table(d):return d.to_html(index=False,float_format=lambda x:f"{x:.3f}",border=0)
def save(path,title,body):path.write_text("<!doctype html><html lang=zh><meta charset=utf-8><title>"+title+"</title>"+style+"<body><h1>"+title+"</h1>"+body+"</body></html>")
s=pd.read_csv(r/"phase6/summary.csv");p=pd.read_csv(r/"phase6/paired.csv");c=p[(p.variant=="context11")&(p.method=="frozen")&(p.comparator_variant=="wide")&(p.comparator_method=="frozen")]
body="""<div class=notice><b>主要比较未支持预期优势。</b>年轻来源 context11 相对于参数量匹配的 wide，在 19 位新受试者上平均 −0.081 个百分点，95% 受试者配对 bootstrap 区间 [−1.638, +1.476]。不能据此宣称等效，也不能用次要比较取代主要比较。</div>
<p>受试者 40–58，66–74 岁；每人首个可用夜晚。模型、算法、主要比较及代码摘要在解析目标标签前锁定。每人先平均三个随机种子，再进行受试者级比较。所有指标为固定五类 Macro-F1；完整连续记录无基于标签的睡眠裁剪。</p>
<h2>冻结模型</h2>"""+table(s[s.method=="frozen"][["source","variant","n","f1","transition"]])+"<h2>历史上下文与匹配参数量对照</h2>"+table(c[["source","delta_pp","ci_low","ci_high","wins","harm_rate","worst_delta_pp","transition_delta_pp"]])+"""<p>mixed 的 +2.436pp 是次要发现；存在 3/19 人下降超过 1pp，最差约 −8.09pp，转换期指标下降约 1.21pp。多重比较未经校正，不能据此宣称稳定泛化优势。</p><h2>保留控制器</h2>"""+table(s[(s.variant=="context11")&s.method.isin(["frozen","identity_future","identity_confidence","identity_current","continuous","ema_08"])])+"""<p>future 控制器相对冻结模型仅约 +0.011pp（young）与 +0.002pp（mixed），没有出现有实际量级的提升。confidence 控制器也使用源域未来收益标签；它检验观测特征的作用，不检验是否需要未来监督。</p>
<h2>独立审计</h2><p>969 个结果文件、3192 条预测流通过审计；模型及锁定文件摘要一致，重算指标误差为零。18 次控制器前缀审计完全一致。171 条 batch=1 冻结流与批量预测的分类结果无差异，最大概率差 9.54e−7。</p>
<p><a href=evaluation_lock.json>锁定协议</a> · <a href=final_audit.json>完整审计</a> · <a href=paired.csv>全部配对比较</a> · <a href=../research_overview.html>研究总览</a></p>
<p>范围限制：同一公开睡眠数据集、少量源域训练样本与小型模型；尚不能外推为对所有 EEG 基础模型或所有测试时训练方法的否定。</p>"""
save(r/"phase6/report.html","第二次新受试者验证：主要比较未获支持",body)
s12=pd.read_csv(r/"phase12/summary.csv");s13=pd.read_csv(r/"phase13/summary.csv")
body="""<div class=notice>以下均为已观察开发数据上的探索性比较，不是新增盲测。每种模型三个种子；原始 EEGNet/频谱模型合并两个人群不重叠的折，共 8 人，上下文模型共 11 人。区间为条件于这些训练种子的受试者 bootstrap，未经多重比较校正。</div>
<h2>时序教师更新 BN 参数</h2><p>固定源模型的因果 EMA 概率作为软目标；每个当前片段先预测，再更新。只写 BN 仿射参数，运行统计固定，逐张量改动半径为源参数 RMS（下限 1）的 10%。源验证集选择 EMA 系数和学习率；保留策略仅由源控制器人员的反事实未来收益训练。</p>"""+table(s12[s12.method.isin(["frozen","future","current","teacher_confidence","continuous","weak_ema","source_selected_ema"])])+"""<p>EEGNet 的 future 增益约 +0.399pp，但直接轻度 EMA +1.036pp；频谱模型 +0.200pp，也弱于轻度 EMA +1.025pp。上下文模型没有稳定增益。teacher_confidence 同样由未来收益监督训练，是输入特征消融。</p>
<h2>同范围参数的有监督静态校准</h2><p>只使用控制器已经可以访问的 H4 未来源标签，固定编码器其余参数。分别调整 BN 仿射或既有分类头，使用相同改动半径；由源验证集选择 epoch，包含 epoch=0 的冻结回退。此项是有监督容量对照，不是无监督测试时训练，也不是性能上限。</p>"""+table(s13)+"""<p>改善不一致，所有非零平均效应的区间均跨零。不能把它概括为“参数不能学”，也不能单独断言瓶颈必然是无监督目标；源人群覆盖、参数范围、目标信号与选择噪声仍共同影响结果。</p><p><a href=../phase12/participant_metrics.csv>教师实验受试者指标</a> · <a href=summary.csv>静态校准汇总</a> · <a href=../research_overview.html>研究总览</a></p>"""
save(r/"phase13/report.html","更新信号与可写参数的补充对照",body)
p4=r/"phase4/report.html";t=p4.read_text();banner="<div class=notice><b>后续验证：</b>本页为开发数据探索。第二次新受试者验证的主要比较未支持 context11 优于匹配参数量的 wide：−0.081pp，95%CI [−1.638,+1.476]。<a href=../phase6/report.html>查看验证报告</a></div>"
if "后续验证：</b>" not in t:t=t.replace("<body>","<body>"+banner);p4.write_text(t)
print("Reports written")
