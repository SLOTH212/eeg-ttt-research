
from pathlib import Path
import json,hashlib
import pandas as pd
R=Path(__file__).resolve().parents[1];O=R/"phase138"
a=json.loads((O/"audit.json").read_text());assert a["status"]=="PASS"
assert hashlib.sha256((O/"protocol.json").read_bytes()).hexdigest()==(O/"protocol.sha256").read_text().strip()
df=pd.DataFrame(a["records"]);assert len(df)==30 and df.windows.sum()==40428
for r in a["records"]:
 path=R/"phase136/predictions"/f"{r['domain']}_{r['subject']}_{r['montage']}_{r['mode']}.npz"
 assert hashlib.sha256(path.read_bytes()).hexdigest()==r["gold_sha256"]
df.to_csv(O/"streams.csv",index=False)
timing=df.groupby(["device","domain","mode"])[["median_ms","p95_ms"]].agg(["min","max"]);timing.to_csv(O/"latency_ranges.csv")
assert df.parent_parameters.nunique()==1 and df.head_parameters.nunique()==1
counts=dict(parent=int(df.parent_parameters.iloc[0]),history_head=int(df.head_parameters.iloc[0]))
small={k:v for k,v in a.items() if k!="records"}
(O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1400px;margin:35px auto}td,th{padding:5px;border:1px solid #ddd}</style><h1>EEGNet-style 完整逐窗 CPU/GPU 复核</h1><p>三个原源选定模型，固定源s05、健康sub-III001和患者sub-I021；源raw、外部raw/full，分别在CPU和GPU执行父模型，历史读出统一使用CPU。共30条流、40,428次窗口处理。起点是已经完成通道处理、重采样、中心化/缩放的2×3000波形，没有覆盖EDF解码、设备采集或跨设备验证。</p><p>固定残差历史头与EMA=0.5；每条流在中点联合保存/恢复输入与历史状态，16步逐元素一致；重建后前16步一致。父模型/历史头/EMA类别与原批量GPU结果全部相同。没有读取分类标签或更新模型。</p><pre>'+json.dumps(small,indent=2)+'</pre><p>参数数量：'+json.dumps(counts)+'。历史头参数也计入总模型成本。</p><h2>延时，毫秒</h2><p>各格是三个种子的流级中位数/p95最小与最大值，不是置信区间或合并窗口分位数。排除前16窗、磁盘、采集、冷启动、模型加载和状态恢复；包含正常逐窗输入状态、父模型与历史读出。CPU线程设定2，亲和性8–23；GPU使用FP32。不同基础模型各有自己的预处理，因此不能将延时差只归因于参数量。</p>'+timing.to_html(float_format=lambda x:f"{x:.4f}")+'<h2>各流结果</h2>'+df.drop(columns=["gold_sha256","head_sha256"]).to_html(index=False,float_format=lambda x:f"{x:.8f}")+'<p><a href="protocol.json">固定方案</a> · <a href="audit.json">审计</a></p>')
print(counts);print(timing.to_string());print(json.dumps(small))
