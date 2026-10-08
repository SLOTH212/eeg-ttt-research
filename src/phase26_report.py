from pathlib import Path
import pandas as pd,json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parents[1];o=r/"phase26";d=pd.read_csv(o/"cls_diagnostic.csv");a=pd.read_csv(o/"cls_attention.csv");rec=pd.read_csv(o/"reconstruction_position_control.csv");m=rec.groupby(["encoder_variant","decoder_divisor"])[["amplitude_mse","phase_mse"]].mean().reset_index();m.to_csv(o/"reconstruction_summary.csv",index=False)
fig,ax=plt.subplots(1,2,figsize=(11,4));modes=["official","cls_key_bias_minus_log8","duplicated16_same_positions","remove_cls"];labels=["Original","CLS key / 8","Duplicate channels","Remove CLS"]
for sid,g in d.groupby("subject"):
 vals=[g.set_index("mode").loc[x,"dominant_fraction"]*100 for x in modes];ax[0].plot(range(4),vals,"o-",label=f"s{sid:02d}")
ax[0].set_xticks(range(4),labels,rotation=20);ax[0].set_ylabel("Dominant code share (%)");ax[0].set_title("Source-only tokenizer diagnostic");ax[0].legend()
for mode,label in zip(modes[:2],labels[:2]):
 g=a[a["mode"]==mode].groupby("layer").patch_to_cls_mean.mean();ax[1].plot(g.index,g.values,"o-",label=label)
ax[1].set_xlabel("Encoder block");ax[1].set_ylabel("Mean patch-to-CLS attention");ax[1].set_title("CLS attention mass");ax[1].legend();fig.tight_layout();fig.savefig(o/"tokenizer_diagnostic.png",dpi=160)
html="""<!doctype html><html lang="zh"><meta charset="utf-8"><title>分词器有效性审计</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:30px auto;padding:20px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px}.box{background:#fff1d5;padding:16px}img{max-width:100%}</style><h1>双导联输入下的分词器塌缩</h1><p>2026-10-06；所有诊断使用源受试者，不读取睡眠分期标签。CPU8–23核，GPU0。</p><div class="box">原计划的联合辅助训练未启动：既有VQNSP在当前双导联15秒输入中，超过99%的片段被映射到代码5288。官方前向与手动最近码计算完全一致，码本已初始化且未变更；输入单位与官方预处理一致。直接训练此目标可能仅学会多数代码。</div><h2>机制诊断</h2><p>三名源验证受试者各抽32个30秒窗口。复制导联但保留相同位置即可恢复多样性，不能仅用空间位置差异解释。双导联带限Gaussian噪声仍塌缩；白噪声有更多代码。合成复制不是生理导联的替代。</p><p>把每层CLS键的注意力logit减去log(8)，得到的代码与相同位置复制成16导联的结果100%一致，归一化潜变量最大差异小于0.00008。删除CLS也恢复多样性。这支持当前输入设置下CLS相对注意力质量影响过强的机制，但修改注意力不是官方分词器，也不自动证明代码语义有效。</p><img src="tokenizer_diagnostic.png" alt="原始与CLS诊断对照"><h2>重建检查</h2><p>三人各64个窗口，官方标准化FFT幅度与相位目标，冻结解码器；同时记录常量代码、打乱代码和零输出。只改编码器时，幅度MSE仍大于1；编码器与解码器同时降低CLS键权重后，幅度MSE约0.70–0.75，优于零输出及常量代码，也略优于打乱代码约0.76–0.80；保留导联与时间位置分布、仅打乱不同窗口的对应关系后，优势仍存在。相位MSE仍大于零输出的约1，不能声称完整重建能力恢复。</p>"""+m.to_html(index=False,float_format=lambda x:f"{x:.4f}")+"""<h2>结论范围</h2><p>这一发现修正早期掩码更新负结果的解读：该目标在当前输入适配下退化，不能由此排除有效辅助任务。分类与熵适应的结果保持不变。后续若采用修正教师，必须作为独立探索分支，增加匹配随机数与训练预算的零辅助损失对照，并先做源验证及写入诊断。</p><p>已完成：23个原始token缓存；官方路径审计；57项输入对照；15项CLS消融；30项初始重建对照及36项位置匹配补充对照。所有权重不变、未读取任务标签。</p></html>""";(o/"report.html").write_text(html)
print(m.to_string(index=False))
