
from pathlib import Path
import json,hashlib
import pandas as pd
R=Path(__file__).resolve().parents[1];O=R/"phase133"
a=json.loads((O/"audit.json").read_text());assert a["status"]=="PASS"
p=O/"protocol.json";assert hashlib.sha256(p.read_bytes()).hexdigest()==(O/"protocol.sha256").read_text().strip()
df=pd.DataFrame(a["records"]);assert len(df)==30 and df.windows.sum()==40428
for r in a["records"]:
 assert hashlib.sha256((R/f"phase{120 if r['backbone']=='cbramod' else 123}"/"predictions"/f"{r['domain']}_{r['subject']}_{r['montage']}_{r['mode']}.npz").read_bytes()).hexdigest()==r["gold_sha256"]
df.to_csv(O/"streams.csv",index=False)
timing=df.groupby(["backbone","domain","mode"])[["median_ms","p95_ms"]].agg(["min","max"])
timing.to_csv(O/"latency_ranges.csv")
small={k:v for k,v in a.items() if k!="records"}
(O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1450px;margin:35px auto}td,th{padding:5px;border:1px solid #ddd}</style><h1>完整逐窗波形到静态历史分类复核</h1><p>固定源s05、健康sub-III001和旧患者sub-I021；两个基础模型、三个种子，源raw以及外部raw/full共30条流、40,428窗。此处起点是已经完成通道相减、重采样、逐窗中心化/缩放的双导联30秒波形；没有覆盖EDF解码或设备采集。</p><p>每窗独立执行输入幅度/形状状态更新（full模式）、模型所需滤波/整形、GPU父模型推理、CPU固定历史头和EMA。父模型及头保持冻结，预测不读取分类标签。与原批量缓存比较，父模型/历史头/EMA均无类别差异，概率误差均小于预定2e-5。</p><p>每条流在中点联合序列化输入状态和读出状态，恢复后16步逐元素相同；流重建后前16步也逐元素相同。属于三条固定记录上的工程复核，不能代替所有设备和数据条件的部署验证。</p><pre>'+json.dumps(small,indent=2)+'</pre><h2>延时：三个种子的每流中位数与p95范围，毫秒</h2><p>包含逐窗输入处理、GPU父模型和CPU历史头；排除磁盘、模型载入、冷启动、采集等待和状态恢复。各流排除前16窗。表中范围是三个模型种子的流级统计最小/最大值，不是合并所有窗口的分位数，也不是置信区间；运行环境与调度会影响计时。</p>'+timing.to_html(float_format=lambda x:f"{x:.4f}")+'<h2>完整流级结果</h2>'+df.drop(columns=["gold_sha256","head_sha256"]).to_html(index=False,float_format=lambda x:f"{x:.8f}")+'<p><a href="protocol.json">固定复核协议</a> · <a href="audit.json">机器可读审计</a></p>')
print(json.dumps(small));print(timing.to_string())
