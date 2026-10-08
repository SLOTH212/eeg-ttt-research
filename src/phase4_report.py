from pathlib import Path
import pandas as pd,json,matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parents[1];out=r/"phase4";df=pd.read_csv(out/"summary.csv");df=df[df.method=="frozen"];variants=["full52","wide","repeat11","relative","context5","context11"]
fig,axs=plt.subplots(2,3,figsize=(14,8),sharey=True)
for row,cohort in enumerate(["young","middle"]):
 for col,source in enumerate(["young","middle","mixed"]):
  ax=axs[row,col];d=df[(df.source==source)&(df.cohort==cohort)].set_index("variant").loc[variants];ax.bar(range(6),d.f1,color=["#8196a5","#8196a5","#8196a5","#dea647","#329fad","#166c91"]);ax.set_xticks(range(6),variants,rotation=35,ha="right");ax.set_ylim(40,80);ax.set_title(f"source={source}; target={cohort}");ax.set_ylabel("Participant mean Macro-F1 (%)");ax.grid(axis="y",alpha=.2)
fig.suptitle("Exploratory controls | 3 seeds per model | young n=8, middle n=7");fig.tight_layout();fig.savefig(out/"context_controls.png",dpi=180);plt.close(fig)
params={}
for phase in ["phase4","phase4_controls"]:
 for f in (r/phase/"results").glob("*/evaluation.json"):
  d=json.loads(f.read_text());params[d["variant"]]=d["parameters"]
params["full52"]=5829
html="""<!doctype html><html lang="zh"><meta charset="utf-8"><title>因果上下文探索</title><style>body{font:16px/1.65 system-ui;max-width:1200px;margin:35px auto;padding:0 25px;color:#20303d}table{border-collapse:collapse;font-size:13px;width:100%}td,th{border:1px solid #ccd5df;padding:6px}img{max-width:100%}.box{background:#eef5f7;padding:16px;border-left:4px solid #166c91}</style><h1>因果历史信息比当前测试时写入更有潜力</h1><div class="box">本页全部属于开发探索：这些受试者已被观察。历史模型的增益不能替代“有益参数写入可识别、可保留”的证明。下一轮将在新受试者上分别检验这两件事。</div><p>11窗口输入指当前窗口+之前10个30秒窗口，总跨度5.5分钟。当前窗口结束即可预测，不等待后续窗口；起始不足历史时重复首个已观察窗口。所有源训练、验证、目标受试者分离，历史构建不跨记录。预测前缀检查通过。</p><p>每种来源都使用10名训练受试者（5女5男），共享4名验证受试者。源训练年龄组成不同；目标为8名年轻人和7名中年人。先在每名受试者内部平均3个种子，再汇总。源域标签选最佳训练轮数；没有用目标标签挑轮数。</p><h2>容量匹配结果</h2><p>年轻来源模型在中年目标上的平均F1：原单窗60.82%，宽单窗62.27%，重复当前窗61.56%，11窗历史67.80%。历史模型的转换区域F1比原单窗提高6.62pp。年轻目标上11窗历史提高6.59pp，但仍有1/8人下降超过1pp；不能宣称人人受益。</p><p>参数量："""+str(params)+"</p><img src=\"context_controls.png\" alt=\"Context capacity controls\"><h2>受试者平均结果</h2>"+df.to_html(index=False,float_format=lambda x:f"{x:.3f}")+"<p>harm_rate为相对同来源full52冻结模型下降超过1pp的受试者比例。transition为真实转换边界前后2个窗口的离线分层结果；标签仅用于评分，不输入预测。</p><p><a href=\"summary.csv\">汇总CSV</a> · <a href=\"participant_metrics.csv\">受试者结果</a> · <a href=\"../phase5/protocol.json\">强模型保留控制再检验</a></p></html>"
(out/"report.html").write_text(html);print(out/"report.html",params)
