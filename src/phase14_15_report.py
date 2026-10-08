from pathlib import Path
import pandas as pd,json
r=Path(__file__).resolve().parents[1]
style="<style>body{max-width:1200px;margin:40px auto;padding:0 24px;font:16px/1.7 system-ui;color:#203047}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:right}th{background:#eef3f8}.notice{background:#fff0df;padding:20px;border-left:5px solid #b65e19}a{color:#1463a0}</style>"
def table(d):return d.to_html(index=False,float_format=lambda x:f"{x:.3f}",border=0)
s=pd.read_csv(r/"phase14/summary.csv");p=pd.read_csv(r/"phase14/paired.csv");h=pd.read_csv(r/"phase15/paired.csv")
body="""<div class=notice><b>预训练信息与测试时更新，都尚未形成稳定优势。</b>预训练特征单独优于单个随机编码器，但并未稳定超过频谱特征。预训练＋频谱相对于随机＋频谱的同维度对照，其开发队列与年长队列配对区间均跨零。分类头未来收益控制器也未稳定超过轻度概率平滑。</div>
<p>全部为已经观察过的人群上的探索；没有新增盲测。所有提取与训练仅使用 CPU 8–23 核。现有两次锁定验证的主要结论保持不变。</p>
<h2>预训练载体及限制</h2><p>使用 LaBraM 官方 base 编码器（约 582 万参数），权重严格匹配；冻结编码器后训练源域线性探针。官方论文附录 D 未列出 Sleep-EDF，但未独立审计作者私有数据的受试者身份。使用当前两条双极 EEG 导联，位置嵌入取端点电极嵌入的平均，这是本项目的探索性适配，不是官方双极导联方案。</p>
<p>每个 30 秒片段内部执行 0.1–45Hz 滤波和 100→200Hz 重采样，拆成两个 15 秒片段，各取最终归一化非 CLS token 的均值，拼接为 400 维。原始采样不能恢复 50Hz 以上的信息；不读取后续 epoch。未进行全编码器有监督微调，结论仅适用于该冻结表征和线性探针方案。</p>
<p>所有 58 人同时提取预训练和 seed17 随机编码器特征，不读取任务标签。源训练人群拟合标准化及逻辑回归，由四位源验证人员选择 C；弱正则化候选需要增加迭代次数，所有纳入的模型均达到收敛标准。单个预训练权重和单个随机编码器不构成种子重复实验。</p>
<h2>冻结线性探针</h2><p>all_dev 为 15 位开发受试者，older 为 19 位年长受试者。分数为受试者平均固定五类 Macro-F1（%）；transition 是真实阶段边界前后各两窗的离线评估。</p>"""+table(s[(s.method=="frozen")&s.cohort.isin(["all_dev","older"])][["source","cohort","representation","f1","transition"]])+"""<h2>同维度联合特征对照</h2><p>下表比较 pretrained＋spectral 与 random＋spectral，均为 452 维。该对照是在观察到联合特征增益后追加的，明确属于事后探索。区间采用受试者配对 bootstrap 10000 次，未经多重比较校正。</p>"""+table(p[(p.representation=="pretrained_plus_spectral452")&(p.comparator_representation=="random_plus_spectral452")&(p.method=="frozen")&p.cohort.isin(["all_dev","older"])][["source","cohort","n","delta_pp","ci_low","ci_high","harm_n","worst_delta_pp"]])+"""<p>相对纯频谱的提升，不能单独证明预训练贡献。控制特征维度与随机非线性特征后，没有稳定的独立预训练优势；区间跨零也不证明等效。</p>
<h2>分类头测试时更新及保留</h2><p>只更新原有分类头，编码器冻结；熵目标或固定源预测的因果 EMA 教师目标。当前先预测，随后 KEEP/COMMIT/RESET；参数投影及梯度裁剪与 phase2 分类头实验一致。源验证选学习率和锚定系数，包含冻结回退。控制器使用四位源人员的 H4 反事实未来 CE 收益，目标标签仅离线评分。</p><p>all_dev 在排除四位控制器人员后为 11 人；older 仍为同一 19 人。下表为 future 相对轻度 EMA 的配对差异。</p>"""+table(h[(h.method=="future")&(h.comparator_method=="weak_ema")&h.cohort.isin(["all_dev","older"])][["source","representation","objective","cohort","n","delta_pp","ci_low","ci_high","harm_n"]])+"""<p>12 组结果均未显示 future 稳定超过轻度 EMA；某些相对冻结的微小正效应，不足以建立控制器相对强简单基线的价值。confidence 控制器同样由未来收益监督训练，是观测特征对照。</p>
<h2>后续源域诊断</h2><p>正在检验 LaBraM 已有 LayerNorm 参数的单步无标签更新：当前熵与固定官方神经分词器的掩码预测辅助目标。每次从源参数开始，未来四窗仅用于离线效应与梯度诊断。这不是目标在线性能试验。</p>
<p><a href="protocol.json">预训练特征协议</a> · <a href="paired.csv">特征完整配对比较</a> · <a href="../phase15/protocol.json">保留实验协议</a> · <a href="../phase15/paired.csv">保留完整配对比较</a> · <a href="../phase16/protocol.json">源域辅助目标诊断协议</a> · <a href="../research_overview.html">研究总览</a></p>
<p>官方来源：<a href="https://github.com/935963004/LaBraM">代码与公开权重</a>，<a href="https://proceedings.iclr.cc/paper_files/paper/2024/file/47393e8594c82ce8fd83adc672cf9872-Paper-Conference.pdf">论文与预训练数据列表</a>。</p>"""
(r/"phase14/report.html").write_text("<!doctype html><html lang=zh><meta charset=utf-8><title>预训练载体与分类头保留实验</title>"+style+"<body><h1>预训练载体与分类头保留实验</h1>"+body+"</body></html>")
print("Pretrained carrier report written")
