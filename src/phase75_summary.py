from common import *
import pandas as pd,hashlib
o=ROOT/"phase75";pr=json.loads((o/"protocol.json").read_text());assert (o/"complete.json").exists();cw=np.array(json.loads((ROOT/"phase36/mixed_class_weights.json").read_text())["balanced_weights"]);records=[];maxerr=0;normerr=0;total=0
for sid in pr["validation_people"]:
 d=json.loads((o/f"s{sid:02d}.json").read_text());assert d["state_restored_exact"] and d["unselected_layers_exact"];truth=np.load(ROOT/("data/processed" if sid<20 else "phase2/processed")/f"s{sid:02d}_meta.npz")["y"];lookup={(r["t"],r["variant"]):r for r in d["rows"]}
 for variant in pr["variants"]:
  rr=[r for r in d["rows"] if r["variant"]==variant];assert [r["t"] for r in rr]==d["times"];g=[];b=[];corr=0;harm=0;changes=0
  for r in rr:
   y=np.array(r["future_labels"]);assert np.array_equal(y,truth[r["t"]+1:r["t"]+5]);p=np.array(r["future_base_probabilities"]);q=np.array(r["future_write_probabilities"]);assert (p>0).all() and (q>0).all() and np.isfinite(q).all() and abs(q.sum(1)-1).max()<1e-12
   gain=np.log(q[np.arange(4),y])-np.log(p[np.arange(4),y]);gg=float(gain.mean());bb=float((gain*cw[y]).mean());err=max(abs(gg-r["future_ce_gain"]),abs(bb-r["balanced_ce_gain"]));maxerr=max(maxerr,err);assert err<1e-10
   c=int(((p.argmax(1)!=y)&(q.argmax(1)==y)).sum());h=int(((p.argmax(1)==y)&(q.argmax(1)!=y)).sum());ch=int((p.argmax(1)!=q.argmax(1)).sum());assert c==r["corrected"] and h==r["harmed"] and ch==r["argmax_changes"];assert r["actual_update_norm"]<=.01001
   ref=lookup[(r["t"],"selected_layer")];assert r["future_base_probabilities"]==ref["future_base_probabilities"]
   if variant=="all_normmatched_selected":
    ne=abs(r["actual_update_norm"]-ref["actual_update_norm"]);normerr=max(normerr,ne);assert ne<2e-6
   g.append(gg);b.append(bb);corr+=c;harm+=h;changes+=ch;total+=1
  records.append(dict(subject=sid,variant=variant,ce_gain=np.mean(g),balanced_gain=np.mean(b),corrected=corr,harmed=harm,changed=changes,accuracy_gain_pp=100*(corr-harm)/(len(rr)*4)))
people=pd.DataFrame(records);people.to_csv(o/"people.csv",index=False);summary=people.groupby("variant").agg(ce_gain=("ce_gain","mean"),balanced_gain=("balanced_gain","mean"),positive_people=("ce_gain",lambda v:int((v>0).sum())),corrected=("corrected","sum"),harmed=("harmed","sum"),changed=("changed","sum"),accuracy_gain_pp=("accuracy_gain_pp","mean"));summary.to_csv(o/"summary.csv");row=summary.loc["selected_layer"];passed=bool(row.ce_gain>0 and row.positive_people>=3 and row.balanced_gain>=0 and row.corrected>=row.harmed);decision=dict(primary="selected_layer",passes_fresh_write_screen=passed,ce_gain=row.ce_gain,balanced_gain=row.balanced_gain,positive_people=int(row.positive_people),corrected=int(row.corrected),harmed=int(row.harmed),selection_future_class_labels=pr["selection_future_class_labels"]);(o/"decision.json").write_text(json.dumps(decision,indent=2))
audit=dict(status="PASS",rows=total,max_ce_error=maxerr,max_matched_norm_error=normerr,validation_disjoint_from_scope_selection=set(pr["scope_selection_people"]).isdisjoint(pr["validation_people"]),selected_parameters=800,unselected_layers_unchanged=True);assert audit["validation_disjoint_from_scope_selection"];(o/"audit.json").write_text(json.dumps(audit,indent=2))
(o/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1100px;margin:35px auto}td,th{padding:6px;border:1px solid #ddd}</style><h1>源侧层范围选择的互斥人群复核</h1><p>层选择来自4名控制器：未来普通和平衡损失一阶贡献均至少3/4人正，再选普通贡献最大的层，固定得到第一个 Transformer 块。使用1024个未来类别标签进行选择，不是无监督层选择。此处在另一组4名源验证者检查，固定学习率0.01，800个 LayerNorm 参数。</p>'+summary.to_html()+'<p>这两组人都已在项目中出现；互斥指此次层选择角色，不是新盲测。保留全层及与选定层更新幅度相同的全层对照。原全层实验的失败结论不变。</p><p>预定单步条件通过：'+str(passed)+'。即使通过，也需连续轨迹与1024标签静态校准对照，不能直接声称 TTT 成功。</p>'+people.to_html(index=False)+'<p><a href="audit.json">独立审计</a> · <a href="protocol.json">固定规则及标签成本</a></p>')
print(summary.to_string());print(decision);print(audit)
