from common import *
import pandas as pd,hashlib,html
R=ROOT/'results';summ=pd.read_csv(R/'summary.csv');main=summ[summ.method.isin(['frozen','smooth_tuned','extended_lr_continuous','future_controller','refined_controller','retuned_future_controller','identity_future','reset_periodic'])].copy()
main['macro_f1_pct']=main.macro_f1_pct.round(2);main['delta_pp']=main.delta_pp.round(2)
small=main.pivot(index='method',columns='model',values='macro_f1_pct').round(2)
count=len(list(R.glob('*/summary.json')));supp=len(list(R.glob('*/supplement.json')));lr=len(list(R.glob('*/lr_sensitivity.json')));pol=len(list(R.glob('*/policy_diagnostics.json')));stress=len(list(R.glob('*/stress.json')));numerical=len(list(R.glob('*/numerical_audit.json')));identity=len(list(R.glob('*/identity_control.json')))
audits={}
for name in ['implementation_audit.json','real_data_audit.json','head_scope_audit.json','cpu_affinity_audit.json','data_checksum_audit.json']:
 audits[name]=json.loads((R/name).read_text()) if (R/name).exists() else None
status={'core_and_head_groups':count,'supplement_groups':supp,'lr_groups':lr,'policy_groups':pol,'stress_groups':stress,'numerical_groups':numerical,'identity_groups':identity,'expected':{'core_and_head':24,'supplement':24,'lr':24,'policy':24,'stress':4,'numerical':24,'identity':24},'cpu_only':True,'allowed_cpu_ids':list(range(8,24))}
(R/'completion_status.json').write_text(json.dumps(status,indent=2))
paras=[]
for model,g in summ.groupby('model'):
 lookup=g.set_index('method');frozen=lookup.loc['frozen','macro_f1_pct'];smooth=lookup.loc['smooth_tuned','macro_f1_pct'] if 'smooth_tuned' in lookup.index else float('nan')
 controllers=[n for n in ['future_controller','refined_controller','retuned_future_controller','identity_future'] if n in lookup.index];winner=max(controllers,key=lambda n:lookup.loc[n,'macro_f1_pct']);score=lookup.loc[winner,'macro_f1_pct']
 paras.append(f'<li><b>{html.escape(model)}</b>：冻结 {frozen:.2f}%，源域调参的因果平滑 {smooth:.2f}%；已测试的未来收益控制器最高为 {score:.2f}%（{winner}）。这里的“最高”为事后描述，并非利用评估标签选择部署策略。</li>')
content='<section><h2>当前结论与完成范围</h2><p>CPU 足以完成本轮小型载体探索，包括真实 EEG 数据、源模型训练、两折三个种子、在线适应、状态干预、控制器消融、学习率与可写集合敏感性，以及合成干扰恢复测试。</p><ul>'+''.join(paras)+'</ul>'
risk=summ[summ.model.isin(['eegnet','spectral'])&summ.method.isin(['frozen','smooth_tuned','identity_future'])][['model','method','macro_f1_pct','negative_subject_rate_1pp','worst_subject_delta_pp']]
stage=pd.read_csv(R/'stage_transition_metrics.csv');stage=stage[stage.model.isin(['eegnet','spectral'])&stage.method.isin(['frozen','smooth_tuned','identity_future'])];stage_table=(stage.groupby(['model','method','region']).macro_f1.mean()*100).unstack('region').round(2)
content+='<h3>平均分与稳定性存在取舍</h3><p>简单平滑的总体平均分更高，但在标签转换附近的 5 个窗口中 Macro-F1 下降。按三个种子先在受试者内平均，平滑使 CNN 的 2/8 人、频谱网络的 1/8 人出现超过 1 个百分点的退化；带状态约束的未来收益控制在本轮均为 0/8。这个小样本结果不构成风险保证。未来收益目标相对同信息预算的置信度控制器仍未展示稳定优势，不能仅凭更少退化证明该目标必要。</p>'+risk.round(4).to_html(index=False)+stage_table.to_html()
content+='<p>不能把正收益写入的存在、后见 oracle 的空间或熵下降直接当成可部署策略有效的证据。应先看它能否稳定超过冻结、简单平滑和源域调好的小步长更新；未超过这些对照时，不应以增加基础模型规模替代方法证据。</p>'
content+='<p>主表为平均受试者 Macro-F1（%）：</p>'+small.to_html()
content+=f'<p>完成计数：主实验及末端层敏感性 {count}/24；补充对照 {supp}/24；扩展学习率 {lr}/24；实际策略轨迹诊断 {pol}/24；合成压力测试 {stress}/4。</p>'
content+='<p>这轮未做 CBraMod、ISRUC 跨数据集、完整 EATA/SAR/StableSleep 对照或独立封存人群验证。当前结果不足以声称方法创新、普适有效或临床可用。两折与多个探索性变体共用一个小队列，后续正式实验必须另设封存评估。</p></section>'
content+='<section><h2>状态语义与步长复核</h2><p>额外对三种控制器统一施加一个可确定的约束：当参数漂移恰为零，参数重置与保持不变完全等价，因此 RESET 的预测收益强制为零。identity_future / identity_confidence / identity_current 用于排除回归器高估无效重置的影响。</p><p>初始学习率网格为 1e-4、1e-3、1e-2。CNN 源域验证出现网格下界仍退化，因此扩展至 1e-6、1e-5；选择与匹配控制器重训均限于源域。初始与重新调参的结果分别保留，不混成一次预登记实验。</p>'
if (R/'benchmark.json').exists():
 b=json.loads((R/'benchmark.json').read_text());content+=pd.DataFrame(b['results']).round(3).to_html(index=False)+'<p>'+html.escape(b['notes'])+'</p>'
content+='</section><section><h2>可复核产物</h2><ul><li><a href="summary.csv">方法汇总 CSV</a></li><li><a href="all_subject_metrics.csv">逐模型、种子和受试者指标</a></li><li><a href="diagnostic_summary.csv">写入与重置机制诊断</a></li><li><a href="policy_diagnostic_summary.csv">实际策略的写入损害诊断</a></li><li><a href="stage_transition_metrics.csv">睡眠阶段转换分层</a></li><li><a href="stress_summary.csv">合成干扰分层</a></li><li><a href="completion_status.json">完成状态</a></li></ul></section>'
p=R/'report.html';text=p.read_text();text=text.replace('<h2>比较结果</h2>',content+'<h2>完整比较结果</h2>',1);p.write_text(text)
# Content hashes for source/config provenance, excluding environment and data from source inventory.
files=[*sorted((ROOT/'src').glob('*.py')),*sorted((ROOT/'configs').glob('*.json'))]
(R/'source_sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
print(json.dumps(status,indent=2));print(small.to_string())
