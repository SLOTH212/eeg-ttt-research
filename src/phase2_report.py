from pathlib import Path
import json,numpy as np,pandas as pd,matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parents[1];out=r/"phase2";s=pd.read_csv(out/"fresh_summary.csv");paired=pd.read_csv(out/"fresh_paired.csv");complete=len(s)==24
methods=["ema_08","ema_05","source_selected_filter","head_entropy","head_pseudo","head_temporal","source_prototype","online_prototype","identity_future","identity_confidence","identity_current"]
fig,axs=plt.subplots(1,2,figsize=(13,6),sharey=True)
for ax,kind in zip(axs,["eegnet","spectral"]):
 d=s[s.kind==kind].set_index("method");available=[m for m in methods if m in d.index];a=d.loc[available];ix=[methods.index(m) for m in available]
 ax.errorbar(a.delta_pp,ix,xerr=np.vstack([a.delta_pp-a.ci_low,a.ci_high-a.delta_pp]),fmt="o",capsize=3,color="#166c91");ax.axvline(0,color="gray",lw=1);ax.set_yticks(range(len(methods)),methods);ax.invert_yaxis();ax.set_xlabel("Macro-F1 change vs frozen (percentage points)");ax.set_title(kind+" | 19 fresh participants");ax.grid(axis="x",alpha=.2)
fig.tight_layout();fig.savefig(out/"fresh_comparison.png",dpi=180);plt.close(fig)
show=s[["kind","method","n","mean_f1","delta_pp","ci_low","ci_high","harm_rate","worst_delta_pp","transition_delta_pp"]].copy();show.columns=["模型","方法","人数","平均 F1 (%)","变化(pp)","区间下界","区间上界","下降>1pp比例","最差变化(pp)","转换段变化(pp)"]
primary=paired[(paired.kind=="eegnet")&(paired.method=="source_selected_filter")&(paired.comparator=="ema_08")]
primary_text=primary.to_html(index=False,float_format=lambda x:f"{x:.3f}")
html="""<!doctype html><html lang="zh"><meta charset="utf-8"><title>EEG CPU 探索 · 新受试者验证</title><style>body{font:16px/1.65 system-ui,sans-serif;max-width:1200px;margin:35px auto;padding:0 25px;color:#20303d}table{border-collapse:collapse;font-size:13px;width:100%}td,th{border:1px solid #ccd5df;padding:6px;text-align:right}th{background:#eef3f7}img{max-width:100%}.box{background:#eef5f7;padding:16px;border-left:4px solid #166c91}a{color:#166c91}</style><h1>CPU 探索第二轮：19 名新受试者</h1>"""
status="全部完成" if complete else "部分方法仍在运行，表格只收录完整方法"
html+=f"<p>状态：{status}。所有实验限制在 CPU 8–23 核，禁用 GPU。</p>"
html+="""<div class="box">目前证据仍不支持把未来收益保留控制器作为已验证的核心创新。开发集约 2pp 的时序过滤收益在新队列明显缩水；参数写入收益很小或产生负迁移。较强的平均平滑收益伴随个体风险，不能只看总平均。</div><h2>预先锁定的验证</h2><p>以 SC20–38 首个可用夜晚为新队列，共19人。38个公开EDF文件均通过官方SHA256校验；方法、参数、源模型及控制器训练数据哈希在解码新标签前锁定。源模型仍为上一轮固定模型；每名新受试者平均2个源域划分×3个种子的结果，再在19人上平均。</p><p>这是同一数据集上的新受试者检验，不是跨数据集复现。原队列年龄25–34岁，新队列50–60岁；年龄与其他队列差异混在一起，不能据此作年龄的因果归因。源域A为10名女性、B为10名男性，后续加入性别平衡训练对照。</p><h2>结果</h2>"""
html+=show.to_html(index=False,float_format=lambda x:f"{x:.3f}")+"<p>区间为受试者配对重采样的描述性95%区间；不是把种子当独立样本。除预定主比较外，多方法结果均按探索性解释。固定五类 Macro-F1，某类缺失记0；不按睡眠标签裁切时间线。</p><img src=\"fresh_comparison.png\" alt=\"Fresh participant comparison\"><h2>预定主比较：EEGNet 状态过滤 vs 弱平滑</h2>"+primary_text
html+="""<h2>实现与可审查证据</h2><p>当前窗口先预测，再更新；无目标标签进入在线策略。状态过滤只使用过去状态与当前冻结模型概率。分类头的时序软目标也只用当前和历史冻结预测。原型基线额外使用源域类别标注构建模板，应与固定源域模板比较。</p><p>前缀因果检查通过。NumPy更新与独立Torch自动微分逐步对照，最大概率差异小于2×10⁻⁸；锁定文件哈希通过。运行中审计线程均限制在指定核内。</p><p>仅计算实际出现的类别后，冻结模型仍表现偏弱：EEGNet约53.46%、谱特征模型58.15%，说明下降不只是缺失类别口径。N1和REM是主要薄弱阶段。</p><h2>继续进行的工作</h2><p>第三轮控制训练人数与性别组成，比较年轻／中年／混合训练来源；第四轮比较相对频谱和严格因果历史输入。两轮复用已经观察过的受试者，明确属于开发探索，不能冒称新的盲测。</p><ul><li><a href="evaluation_lock.json">方法锁定及哈希</a></li><li><a href="fresh_summary.csv">完整汇总 CSV</a></li><li><a href="fresh_participant.csv">受试者级结果</a></li><li><a href="fresh_paired.csv">方法间配对比较</a></li><li><a href="shift_analysis.json">队列与转移统计</a></li><li><a href="results/phase2_audit.json">实现和CPU审计</a></li></ul></html>"""
(out/"report.html").write_text(html);print(out/"report.html","complete",complete)
