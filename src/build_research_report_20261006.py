
from pathlib import Path
import os,json,csv,html,hashlib,zipfile,datetime,re
from zoneinfo import ZoneInfo
from xml.sax.saxutils import escape
import xml.etree.ElementTree as ET
import pandas as pd
os.sched_setaffinity(0,set(range(8,24)))
R=Path(__file__).resolve().parents[1];O=R/"reports/20261006";O.mkdir(parents=True,exist_ok=True)
now=datetime.datetime.now(ZoneInfo("Asia/Kuala_Lumpur"));stamp=now.strftime("%Y-%m-%d %H:%M %Z")
title="EEG 测试时适应项目：阶段研究报告"
base="EEG_TTT_阶段研究报告_20261006"
B=[];sources={}
def para(t):B.append(dict(kind="p",text=t))
def head(t):B.append(dict(kind="h",text=t))
def table(headers,rows):B.append(dict(kind="table",headers=headers,rows=[[str(v) for v in row] for row in rows]))
def source(n,label,relative):
 p=R/relative;assert p.exists();sources[n]=dict(label=label,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
for n,label,relative in [
 ("P114","三种子教师驱动 WRITE", "phase114/summary.csv"),
 ("P117","部署可见特征预测更新效用","phase117/summary.csv"),
 ("P120","CBraMod 静态历史目标结果","phase120/paired_comparisons.csv"),
 ("P123","LaBraM 静态历史目标结果","phase123/paired_comparisons.csv"),
 ("P126","历史分类器的熵 WRITE","phase126/summary.csv"),
 ("P131","源头选模组敏感性","phase131/primary_sensitivity.csv"),
 ("P132","时间结构差异分解","phase132/decomposition.csv"),
 ("P133","基础模型完整逐窗审计","phase133/audit.json"),
 ("P134","种子和单人敏感性","phase134/robustness.csv"),
 ("P135","EEGNet 源推进门槛","phase135/decision.json"),
 ("P136","EEGNet 静态历史目标结果","phase136/paired_comparisons.csv"),
 ("P137","三模型同人群配对比较","phase137/carrier_comparisons.csv"),
 ("P138","EEGNet CPU/GPU 逐窗审计","phase138/audit.json"),
 ("P100","输入幅度与形状校准","phase100/primary_results.csv"),
 ("P101","追加20名患者的固定校准复核","phase101/primary_results.csv")]:source(n,label,relative)
para("报告日期："+stamp+"。覆盖截至 Phase138 的已完成证据，重点汇总能改变研究判断的结果；阶段编号不代表相互独立的试验数量。")
para("运行状态：按用户要求暂停研究，原因是磁盘空间不足。当前仅进行报告整理；不再启动训练、推理、下载或扩展缓存，等待用户明确恢复指令。")
head("一、结论摘要")
para("目前最可靠的正向结果是静态因果历史读出，而不是测试时梯度更新。在相同 SleepEDF 30 名目标受试者、三个种子、EMA=0.5 的条件下，历史头相对同容量“重复当前窗口”头，在 CBraMod、LaBraM、EEGNet-style 上分别提高 3.45、2.25、2.84 个 Macro-F1 百分点，三个配对区间均为正。[P120/P123/P136]")
para("更换读出头的源选模人群后，两种基础模型仍有正向历史收益，但幅度降至 2.50 和 1.09 个百分点。收益不是原选模组独有，但大小对源选模人群敏感。父模型检查点保持原选择，所以该实验不是整个训练流程的交叉验证。[P131]")
para("外部迁移尚未得到一致支持。历史读出在外部人群的收益较小、不稳定，部分设置明确退化。教师驱动 WRITE 虽能改善某些未来窗口，仍不及直接教师或静态混合；加强静态历史分类器后，熵 WRITE 也没有通过固定门槛。[P114/P126/P136]")
para("项目已经形成可复现的静态基线、成对更新诊断和完整逐窗工程证据；尚未证明“可观察地识别有益更新并有效保留”这一核心方法主张。当前结果支持继续收窄问题，不支持宣称 TTT 已成功或已经具备稳定跨数据集适应能力。")
head("二、研究问题、数据与评价边界")
para("原建议书研究的是：在先发布当前预测、再更新参数的协议下，是否存在对未来窗口有益的合法 WRITE（H1）；该机会能否仅由部署可见信息识别（H2）；学习 KEEP/WRITE/RESET 是否优于小学习率、固定衰减和重置等简单策略（H3）。建议书作为研究参考，其中建议预算和关口不替代用户的执行授权或暂停指令。")
table(["数据部分","规模与作用","本报告中的使用范围"],[
 ["SleepEDF 源训练","10 人","冻结父模型与静态读出头的源监督训练"],
 ["原源验证 / 替代选模","各 4 人","原组 s05/15/25/35；替代组 s07/17/27/37，仅替换头选模"],
 ["SleepEDF 目标","30 人","最新三模型静态历史的主要同库比较"],
 ["ISRUC 健康外部","10 人","原始与固定校准输入；左侧代理蒙太奇为主要展示"],
 ["ISRUC 既有患者","19 人，I021–I039","外部静态比较；I040 因参考导联不明在读取标签前排除"],
 ["ISRUC 追加患者","20 人，I041–I060","固定输入校准复核；未纳入最新历史头比较，临时波形缓存未保留"]])
para("SleepEDF 处理的是完整 30 秒窗口时间线，不用标签裁剪睡眠区间。ISRUC 的左右代理导联不等同于 SleepEDF 的 Fpz-Cz/Pz-Oz，因此输入校准不能消除所有导联及人群差异。未知标签窗口不计入分类评分，但保持因果流处理规则。")
para("主要统计单位是受试者：先在每人内部平均三个种子的指标，再按人配对 bootstrap 10,000 次，报告 95% 百分位区间。Macro-F1 固定使用 W/N1/N2/N3/REM 五类，零分母类按预定规则记零。相邻窗口、左右蒙太奇和多个种子不作为新增独立受试者。")
para("所有目前报告的目标人群均已被研究观察；多个阶段是在既有结果基础上的探索或复核。各阶段局部方案与检查点虽有固定记录，仍不等于全项目预注册或新的盲测试。本报告区间没有进行全项目多重比较校正，应理解为探索性不确定性描述。")
head("三、静态历史读出：当前最明确的正向结果")
para("冻结父模型，取每窗五类中心化 logit。三个读出分支分别使用当前窗口、将当前窗口重复 11 次，以及过去 10 窗加当前窗口。后两者具有相同的 6,021 个可训练头参数、初始化和训练顺序，区别是是否获得真实历史。残差输出初始为零；额外进行 30 轮源监督训练，并使用相同的源验证保护门槛和 epoch0 回退。不存在目标参数更新。")
models=[("CBraMod",120),("LaBraM",123),("EEGNet-style",136)]
primary=[];external=[]
for name,ph in models:
 d=pd.read_csv(R/f"phase{ph}/paired_comparisons.csv")
 r=d[(d.domain=="sleepedf")&(d.montage=="native")&(d["mode"]=="raw")&(d.ema==.5)&(d.reference=="repeat11")].iloc[0]
 f=d[(d.domain=="sleepedf")&(d.montage=="native")&(d["mode"]=="raw")&(d.ema==.5)&(d.reference=="frozen")].iloc[0]
 primary.append([name,f"{f.reference_f1:.2f}",f"{r.reference_f1:.2f}",f"{r.history_f1:.2f}",f"{r.difference_pp:+.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]",f"{int(r.harm_over_1pp)}/30"])
 for domain,label in [("healthy","健康10人"),("patient","患者19人")]:
  q=d[(d.domain==domain)&(d.montage=="left")&(d["mode"]=="full")&(d.ema==.5)&(d.reference=="repeat11")].iloc[0]
  external.append([name,label,f"{q.history_f1:.2f}",f"{q.reference_f1:.2f}",f"{q.difference_pp:+.2f} [{q.ci_low:.2f}, {q.ci_high:.2f}]",f"{int(q.harm_over_1pp)}/{int(q.people)}"])
table(["模型","冻结父模型","等容量重复头","历史头","历史−重复：差值 [95%区间]","下降>1pp人数"],primary)
para("表中为 Macro-F1 百分制，主要条件固定为 SleepEDF 原始输入、EMA=0.5。负迁移人数相对本表的等容量重复头；1 个百分点只是描述性容忍线，不是临床阈值。所有种子和回退状态均保留。[P120/P123/P136]")
para("小型 EEGNet-style 也能获得历史收益，说明该现象不专属于基础模型。三种历史模型在本组 SleepEDF 的两两绝对分数差异区间均跨 0，例如 CBraMod−EEGNet 为 +0.70pp，区间 [−1.94, 3.52]。这不证明三者等效，也不能将比较解释为预训练的因果消融，因为父模型结构与训练预算不同。[P137]")
para("另一个强静态对照——context11 频谱教师——在相同 SleepEDF 条件下为 63.59。当前历史头没有建立对该教师的明确优势。不能把“超过自身冻结父模型”替代为“超过最强合法静态基线”。")
head("四、稳健性与外部迁移")
table(["基础模型","原头选模组：历史−重复","替代头选模组：历史−重复 [95%区间]"],[
 ["CBraMod","+3.45pp","+2.50pp [1.39, 3.64]"],
 ["LaBraM","+2.25pp","+1.09pp [0.47, 1.75]"]])
para("替换选模组时，18 条训练轨迹在原所选检查点和第30轮检查点均逐元素重现；只改变头检查点由哪四人选择。两种基础模型、两组选模和全部三个种子的 SleepEDF 平均历史效应均为正；逐次剔除任一目标受试者后，三种子平均效应仍为正。该证据支持已有同库现象的稳定性，不构成新的独立人群验证。[P131/P134]")
table(["模型","外部人群","历史头","重复头","差值 [95%区间]","下降>1pp人数"],external)
para("外部表固定使用左侧代理蒙太奇、既有幅度与形状校准、EMA=0.5。两种基础模型的历史−重复区间均跨 0；EEGNet 在两个外部人群中均明确退化。SleepEDF 上的正向结论不能直接推广到外部睡眠数据。[P120/P123/P136]")
para("另一方面，比较历史头的绝对分数时，基础模型在外部数据明显高于 EEGNet：健康组高约 13–14pp、患者组高约 13–16pp。轻量模型在同库表现接近，并不代表它同样耐受域变化。该比较依然受训练方案差异限制。[P137]")
head("五、TTT 核心假设：证据到哪里为止")
table(["假设","已有证据","当前判断"],[
 ["H1：存在有益 WRITE","教师驱动单次更新在源诊断中有未来收益；若干重建或熵目标未通过门槛","部分成立于受限诊断；尚未证明超过强静态对照的实用机会"],
 ["H2：机会可观察","固定部署可见特征的岭回归，对未来平衡CE效用预测较弱","尚未建立可部署的稳定识别能力"],
 ["H3：保留控制有必要","多轮门控、衰减、重置及约束目标没有形成稳定竞争优势","未成立；不支持继续叠加复杂控制器来包装结果"]])
para("教师驱动 WRITE 的三种子复核共包含 1,536 次冷启动单次写入。在源验证组未来4窗的脉冲评价中，KEEP 准确率为 88.12%，WRITE 为 89.23%，但直接教师为 92.90%，静态半混合为 93.10%。因此“WRITE 相对 KEEP 有益”并没有证明需要参数更新。教师还引入了额外模型、历史输入和训练成本。[P114]")
para("在已加强的 CBraMod 历史分类器上，主设置为 LayerNorm affine 单次熵更新、学习率0.001；全部诊断共6,144次写入。主设置的未来普通CE收益在控制组/验证组为 −0.00233/−0.00126，平衡CE收益为 −0.00885/−0.00204；负值表示恶化。更大学习率也没有挽救结果，因此没有推进在线目标更新。[P126]")
para("部署可见特征的未来效用预测没有达到可靠识别水平。固定岭回归在源评价组对平衡CE收益的平均 R² 为 −0.161，Spearman 为 −0.034；四人中只有两人的 MSE 优于固定训练均值。平均绝对 MSE 收益与相对 MSE 收益的方向不同，不能挑其中一个指标声称 H2 成立。[P117]")
para("重建有效性、损失下降、梯度方向相似以及 RESET 修复，都不等于新增 WRITE 改善未来分类。使用未来标签的 oracle 仅用于离线诊断，不是可部署策略，也不是全策略空间的严格上界。上述脉冲未来窗口可能重叠，不能当作完整在线 Macro-F1 或相互独立样本。")
head("六、域差异和输入校准提供的解释")
para("固定输入幅度与双通道形状校准，在原外部队列有一定价值：CBraMod 相对仅幅度校准，在健康10人提高 4.72pp [0.52, 8.91]，既有患者19人提高 2.91pp [0.51, 6.00]。这属于无梯度输入处理，不是参数 TTT。[P100]")
para("在追加20名患者的固定方案复核中，同一形状增量仅为 +0.10pp [0.00, 0.26]，只有2人触发并改善、18人不变，未获得可普遍推广的证据。不能只保留原人群的正向结果。[P101]")
table(["人群","平均相邻阶段转换率","相对源训练的差值","阶段占比项","阶段内转换项"],[
 ["源训练10人","3.54%","—","—","—"],
 ["SleepEDF目标30人","4.87%","+1.33pp","+0.60pp","+0.73pp"],
 ["健康外部10人","19.06%","+15.52pp","+3.72pp","+11.80pp"],
 ["患者外部19人","17.16%","+13.62pp","+4.19pp","+9.44pp"]])
para("分解先把每个人的5×5联合转换计数归一化，再在人群内等权平均，使用对称恒等式将差值分为阶段占比与阶段内部转出概率两项。最大回算误差约3.1×10⁻¹⁶。外部转换差异不能仅由睡眠/清醒占比解释；但这是描述性分解，不能把阶段内项直接归因为病理、生理或标注差异，也不证明它单独导致模型退化。[P132]")
head("七、工程可复现性与运行成本")
para("已完成两个独立的完整逐窗审计：CBraMod/LaBraM 合计30条流、40,428次窗口处理；EEGNet 的 CPU/GPU 合计同样30条流、40,428次窗口处理。每项均固定使用源s05、健康III001、患者I021三条记录，并覆盖三个种子及约定输入模式；这些处理次数不是独立受试者或独立窗口样本量。")
para("两项审计中，父模型、历史头和EMA的类别预测均与原批量结果一致。基础模型最大概率误差约1.0×10⁻⁵，EEGNet约1.8×10⁻⁶；中途联合JSON状态恢复后16步、重新开始后前16步均逐元素一致。参数保持冻结。[P133/P138]")
table(["流水线","每流中位延时范围","审计内最高p95","计算范围"],[
 ["CBraMod，GPU父模型","6.90–7.52 ms","7.78 ms","输入处理＋父模型＋CPU历史头/EMA"],
 ["LaBraM，GPU父模型","4.56–5.00 ms","5.61 ms","同上，采用各自固定预处理"],
 ["EEGNet-style，CPU父模型","0.56–0.75 ms","0.88 ms","原生100Hz输入处理＋父模型＋历史头/EMA"],
 ["EEGNet-style，GPU父模型","0.31–0.45 ms","0.53 ms","同上"]])
para("延时排除磁盘、采集等待、EDF解码、模型载入、冷启动和状态恢复，各流排除前16窗。范围是不同记录/种子的流级统计，不是置信区间，也不是汇总窗口的分位数。允许观察完整当前30秒窗口后输出，因此毫秒级计算时延不代表毫秒级信号内部因果性。")
para("EEGNet-style 父模型1,661参数，历史头6,021参数，合计7,682参数。它是历史紧凑实现，不是官方 EEGNet 精确复现。测试时输入状态及历史缓存虽小，模型权重、运行时、Python对象和GPU上下文仍需另计；本报告不以小缓存声称总内存只有数百字节。")
head("八、整体评价、局限与恢复后的优先事项")
para("按原建议书的逻辑，已实现路径的基础复现与因果工程检查基本通过；H1仅有受限支持，H2尚未建立，H3未成立。主方法超过最强合法基线及稳定跨域推广的关口尚未达到。当前最合适的项目定位是“已有模型测试时写入效用的系统诊断，以及静态时序基线对适应收益的约束”，而不是已完成的新型保留控制算法。")
para("关键局限包括：源验证组小且被反复观察；目标队列已多次用于探索；外部代理导联与记录分布不同；源/目标清醒比例差异较大；各架构父模型训练预算不同；部分受试者存在明显负迁移；建议书列出的全部统一协议强基线和正式封存测试尚未完成。因此不能宣称临床可用、全方法最优、模型等效或普适的梯度适应无效。")
para("恢复研究应优先处理三件事：第一，在用户确定存储空间安排后再恢复，保留既有证据和失败记录；第二，预先固定更强静态对照、统一预算和新的独立评价设计，再检验任何新增适应主张；第三，仅在合法写入能够重复超过强静态对照，并显示可观察效用时，再开发复杂保留控制器。当前不建议继续围绕同一已观察目标集调整门槛寻找正结果。")
para("以上为后续建议，不表示已安排新实验。报告交付后研究保持暂停；本次没有删除历史数据、清理缓存、安装软件或启动新下载。")
head("九、复核索引与交付说明")
para("数值以对应机器可读CSV/JSON为准，报告显示值按两位小数四舍五入。证据索引列出本机绝对路径，附带源文件SHA256清单；将报告移到其他机器后，需同时带上相应实验文件才能打开这些本地链接。Word与HTML包含相同正文。")
for n,v in sources.items():B.append(dict(kind="source",id=n,text=v["label"],path=v["path"]))
para("原始项目建议书：[private project proposal; not redistributed]。报告不重复引用未经本次复核的外部文献主张；研究新颖性仍需独立的文献和统一基线审查。")
payload=dict(title=title,generated_local=stamp,status="PAUSED_BY_USER",blocks=B,sources=sources)
(O/(base+"_证据清单.json")).write_text(json.dumps(payload,indent=2,ensure_ascii=False))
# Small, standalone HTML; no remote assets or copied caches.
parts=[]
for b in B:
 if b["kind"]=="h":parts.append("<h2>"+html.escape(b["text"])+"</h2>")
 elif b["kind"]=="p":parts.append("<p>"+html.escape(b["text"])+"</p>")
 elif b["kind"]=="source":parts.append('<p class="source"><b>['+b["id"]+']</b> <a href="'+html.escape(b["path"],quote=True)+'">'+html.escape(b["text"])+'</a><br><span>'+html.escape(b["path"])+'</span></p>')
 else:
  parts.append("<table><thead><tr>"+"".join("<th>"+html.escape(t)+"</th>" for t in b["headers"])+"</tr></thead><tbody>"+"".join("<tr>"+"".join("<td>"+html.escape(t)+"</td>" for t in row)+"</tr>" for row in b["rows"])+"</tbody></table>")
css="body{font-family:'Noto Sans CJK SC','Microsoft YaHei',sans-serif;max-width:1120px;margin:40px auto;padding:0 25px;color:#172a3a;line-height:1.8;font-size:16px}h1{font-size:30px;border-bottom:4px solid #216780;padding-bottom:15px}h2{font-size:22px;margin-top:34px;color:#185b73}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0;line-height:1.5}th,td{border:1px solid #ccd5db;padding:9px;text-align:left}th{background:#e9f2f6}tr:nth-child(even){background:#f8fafb}.source{font-size:12px;overflow-wrap:anywhere}.source span{color:#586c77}a{color:#176785}@media print{body{max-width:none;margin:0;padding:0;font-size:10.5pt}h1{font-size:21pt}h2{font-size:14pt;break-after:avoid}table{font-size:8.5pt}tr{break-inside:avoid}thead{display:table-header-group}.source{font-size:8pt}p{orphans:3;widows:3}}"
(O/(base+".html")).write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>'+title+'</title><style>'+css+'</style></head><body><h1>'+title+'</h1>'+"".join(parts)+'</body></html>')
# Minimal valid OOXML package, created with stdlib only; no dependency installation.
W="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
def run(t,bold=False,size=None):
 rp=('<w:b/>' if bold else '')+(f'<w:sz w:val="{size}"/>' if size else '')
 return '<w:r>'+('<w:rPr>'+rp+'</w:rPr>' if rp else '')+'<w:t xml:space="preserve">'+escape(t)+'</w:t></w:r>'
def par(t,style=None):
 return '<w:p>'+('<w:pPr><w:pStyle w:val="'+style+'"/></w:pPr>' if style else '')+run(t)+'</w:p>'
def doc_table(headers,rows):
 n=len(headers);width=9638//n
 p='<w:tbl><w:tblPr><w:tblW w:w="9638" w:type="dxa"/><w:tblLayout w:type="fixed"/><w:tblBorders>'+''.join('<w:'+s+' w:val="single" w:sz="4" w:color="CCD5DB"/>' for s in ['top','left','bottom','right','insideH','insideV'])+'</w:tblBorders></w:tblPr><w:tblGrid>'+''.join(f'<w:gridCol w:w="{width}"/>' for _ in headers)+'</w:tblGrid>'
 for i,row in enumerate([headers]+rows):
  p+='<w:tr><w:trPr><w:cantSplit/>'+('<w:tblHeader/>' if i==0 else '')+'</w:trPr>'
  for t in row:
   p+=f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>'+('<w:shd w:fill="E9F2F6"/>' if i==0 else '')+'</w:tcPr><w:p><w:pPr><w:spacing w:after="60" w:before="60"/></w:pPr>'+run(t,bold=i==0,size=18)+'</w:p></w:tc>'
  p+='</w:tr>'
 return p+'</w:tbl>'+par("")
body=par(title,"Title")
for b in B:
 if b["kind"]=="h":body+=par(b["text"],"Heading1")
 elif b["kind"]=="p":body+=par(b["text"])
 elif b["kind"]=="source":body+=par("["+b["id"]+"] "+b["text"]+"\n"+b["path"],"Source")
 else:body+=doc_table(b["headers"],b["rows"])
body+='<w:sectPr><w:footerReference w:type="default" r:id="rIdFooter"/><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" w:header="567" w:footer="567"/></w:sectPr>'
document='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="'+W+'" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><w:body>'+body+'</w:body></w:document>'
styles='<?xml version="1.0" encoding="UTF-8"?><w:styles xmlns:w="'+W+'"><w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="宋体"/><w:sz w:val="22"/><w:lang w:val="zh-CN" w:eastAsia="zh-CN"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="330" w:lineRule="auto"/><w:widowControl/></w:pPr></w:pPrDefault></w:docDefaults><w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style><w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:after="300"/></w:pPr><w:rPr><w:b/><w:color w:val="185B73"/><w:sz w:val="38"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:before="300" w:after="140"/><w:outlineLvl w:val="0"/></w:pPr><w:rPr><w:b/><w:color w:val="185B73"/><w:sz w:val="28"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Source"><w:name w:val="Source"/><w:basedOn w:val="Normal"/><w:rPr><w:sz w:val="17"/><w:color w:val="586C77"/></w:rPr></w:style></w:styles>'
footer='<?xml version="1.0" encoding="UTF-8"?><w:ftr xmlns:w="'+W+'"><w:p><w:pPr><w:jc w:val="center"/></w:pPr>'+run("EEG TTT 阶段报告 · 2026-10-06 · 已暂停研究  |  ",size=16)+'<w:fldSimple w:instr="PAGE"/></w:p></w:ftr>'
types='<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/><Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/></Types>'
rels='<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'
docrels='<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/><Relationship Id="rIdFooter" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/></Relationships>'
package={"[Content_Types].xml":types,"_rels/.rels":rels,"word/document.xml":document,"word/styles.xml":styles,"word/footer1.xml":footer,"word/_rels/document.xml.rels":docrels}
for name,data in package.items():ET.fromstring(data)
dp=O/(base+".docx")
with zipfile.ZipFile(dp,"w",zipfile.ZIP_DEFLATED) as z:
 for name,data in package.items():z.writestr(name,data.encode("utf-8"))
with zipfile.ZipFile(dp) as z:
 assert z.testzip() is None
 doc=ET.fromstring(z.read("word/document.xml"))
 texts=[e.text or "" for e in doc.iter("{"+W+"}t")]
 for b in B:
  if b["kind"] in ["p","h"]:assert b["text"] in texts
 assert len(list(doc.iter("{"+W+"}tbl")))==sum(b["kind"]=="table" for b in B)
for v in sources.values():assert hashlib.sha256(Path(v["path"]).read_bytes()).hexdigest()==v["sha256"]
audit=dict(status="PASS",source_files=len(sources),source_hashes_rechecked=True,body_blocks=len(B),tables=sum(b["kind"]=="table" for b in B),docx_zip_and_all_xml_parse=True,all_body_paragraphs_roundtrip=True,layout_rendered_in_word=False,limitation="No local Word/LibreOffice renderer available; structure/text/table checks performed; HTML provides standalone viewing version.",files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in O.glob(base+"*")})
(O/"report_audit.json").write_text(json.dumps(audit,indent=2,ensure_ascii=False))
print(json.dumps({k:v for k,v in audit.items() if k!="files"},ensure_ascii=False));print(json.dumps(audit["files"],ensure_ascii=False))
