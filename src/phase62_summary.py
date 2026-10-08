from pathlib import Path
import json,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];o=ROOT/"phase62"
assert json.loads((o/"execution_audit.json").read_text())["status"]=="PASS"
df=pd.read_csv(o/"reconstruction.csv");assert len(df)==16;maximum=0.
for row in df.itertuples():
 name=f"s{row.subject:02d}_{row.preprocessing}"
 x=np.load(o/"cases"/(name+"_input.npz"))["x"];z=np.load(o/"cases"/(name+"_"+row.mask+".npz"));mask=z["mask"]
 assert np.array_equal(z["masked_target"],x[mask]);assert np.isfinite(z["masked_prediction"]).all() and np.isfinite(z["masked_shuffle_prediction"]).all()
 template=(x*(~mask)[...,None]).sum(2)/(~mask).sum(2)[...,None];offset=0;rows=[]
 for i in range(96):
  n=int(mask[i].sum());target=z["masked_target"][offset:offset+n].astype(np.float64);pred=z["masked_prediction"][offset:offset+n].astype(np.float64);shuffle=z["masked_shuffle_prediction"][offset:offset+n].astype(np.float64);visible=np.broadcast_to(template[i,:,None,:],x[i].shape)[mask[i]].astype(np.float64)
  rows.append([np.square(pred-target).mean(),np.square(target).mean(),np.square(shuffle-target).mean(),np.square(visible-target).mean(),np.square(pred).mean()]);offset+=n
 rows=np.array(rows);assert np.allclose(rows,z["window_metrics"],rtol=1e-12,atol=1e-12)
 means=rows.mean(0);err=max(abs(means[0]-row.masked_mse),abs(means[1]-row.zero_mse),abs(means[2]-row.shuffled_mse),abs(means[3]-row.visible_template_mse));maximum=max(maximum,err);assert err<1e-12
 assert bool(row.validity_pass)==bool(means[0]<means[1] and means[0]<means[2])
primary=df[(df.preprocessing=="existing")&(df["mask"]=="bernoulli")]
decision=dict(primary="existing preprocessing / Bernoulli mask",validity_pass=bool(primary.validity_pass.all()),positive_people=int(primary.validity_pass.sum()),total_people=4,no_classification_or_online_claim=True)
(o/"decision.json").write_text(json.dumps(decision,indent=2));(o/"audit.json").write_text(json.dumps(dict(status="PASS",cases=16,sampled_windows_per_person=96,unique_people=4,masked_targets_exactly_match_inputs=True,perwindow_metrics_recomputed=True,maximum_metric_error=maximum,official_state_unchanged=True,no_classification_labels_used=True),indent=2))
html='<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{font:15px/1.7 system-ui;max-width:1300px;margin:30px auto}td,th{padding:5px;border-bottom:1px solid #ddd}</style><h1>CBraMod 现成波形重建头有效性</h1><p>直接使用官方公开预训练检查点，严格加载全部211项状态，包括200→200波形输出头；共4924000参数，不新增辅助拟合。现有SleepEDF两导联，30秒内60个一秒patch；不复制导联，不读类别标签。公开模型输入允许可变通道数，但这不保证两导联上的有效性。</p><p>每位源验证受试者按时间均匀采样96个窗口，主条件沿用项目0.1–45Hz预处理与固定随机Bernoulli0.5掩码；0.3–35Hz和棋盘掩码为预先列出的次级诊断。输入已为100微伏单位，不再重复除100。滤波只涉及当前30秒，区别于官方睡眠预处理的整段滤波和六导联序列分类器；这不是官方ISRUC任务结果复现。</p>'+df.to_html(index=False,float_format=lambda x:f"{x:.7f}")+f'<p>主条件要求四人都优于零预测与输入打乱：{decision["validity_pass"]}，通过人数{decision["positive_people"]}/4。每窗先平均masked patch的MSE，再平均窗口，最后按人报告。可见patch的200点通道平均模板另列作简单基线。无掩码重建仅为诊断。</p><p>隐藏原始patch扰动和其他batch样本扰动检查通过，预处理逐窗与批处理一致，所有模型状态未变。重建有效性与有用分类更新是不同问题，本阶段不产生分类准确率或在线策略结论。</p><p><a href="https://github.com/wjq-learning/CBraMod/tree/b9e961003214326972c567eff390e75b0287e32a">官方固定源码</a> · <a href="https://huggingface.co/weighting666/CBraMod/tree/500543c7e30bda1b22bfd51a49301b238dee21fd">官方链接固定权重</a> · <a href="protocol.json">协议</a> · <a href="checkpoint_audit.json">权重审计</a> · <a href="audit.json">结果独立审计</a> · <a href="execution_audit.json">执行检查</a></p></html>'
(o/"report.html").write_text(html);print(df.to_string(index=False));print(decision)
