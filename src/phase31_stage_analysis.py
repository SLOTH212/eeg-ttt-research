from common import *
import pandas as pd,warnings
warnings.filterwarnings("ignore",message="y_pred contains classes not in y_true")
o=ROOT/"phase31";rows=[];effects=[]
for sid in json.loads((o/"protocol.json").read_text())["subjects"]:
 a=np.load(o/"young_joint_0"/f"s{sid:02d}.npz");b=np.load(o/"young_joint_0.01"/f"s{sid:02d}.npz");y=b["y"];assert np.array_equal(y,a["y"]);valid=y>=0;cohort="older" if sid>=40 else "middle" if sid>=20 else "young"
 for name,before,after in [("joint_static_vs_zero",a["frozen"],b["frozen"]),("masked_vs_joint_frozen",b["frozen"],b["masked_selected"]),("entropy_vs_joint_frozen",b["frozen"],b["entropy_selected"])]:
  pb=before.argmax(1);pa=after.argmax(1);changed=(pb!=pa)&valid;improved=(pb!=y)&(pa==y)&valid;harmed=(pb==y)&(pa!=y)&valid;ce0=metrics(y,before)["ce"];ce1=metrics(y,after)["ce"]
  effects.append(dict(subject=sid,cohort=cohort,comparison=name,n=int(valid.sum()),changed_fraction=float(changed.sum()/valid.sum()),corrected_fraction=float(improved.sum()/valid.sum()),harmed_fraction=float(harmed.sum()/valid.sum()),ce_reduction=ce0-ce1))
  for cls,label in enumerate(["W","N1","N2","N3","REM"]):
   where=y==cls;n=int(where.sum())
   def f1(pred):
    tp=int(((pred==cls)&where).sum());fp=int(((pred==cls)&(y!=cls)&valid).sum());fn=n-tp;return 2*tp/max(2*tp+fp+fn,1)
   rows.append(dict(subject=sid,cohort=cohort,comparison=name,stage=label,n=n,f1_delta_pp=100*(f1(pa)-f1(pb)),recall_delta_pp=100*((pa[where]==cls).mean()-(pb[where]==cls).mean()) if n else np.nan,ce_reduction=float(log_loss(y[where],before[where],labels=range(5))-log_loss(y[where],after[where],labels=range(5))) if n else np.nan))
d=pd.DataFrame(rows);d.to_csv(o/"stage_participants.csv",index=False);s=d.groupby(["comparison","stage"]).agg(n_people=("subject","count"),n_people_present=("n",lambda a:int((a>0).sum())),n_windows=("n","sum"),f1_delta_pp=("f1_delta_pp","mean"),recall_delta_pp=("recall_delta_pp","mean"),ce_reduction=("ce_reduction","mean")).reset_index();s.to_csv(o/"stage_summary.csv",index=False);e=pd.DataFrame(effects);e.to_csv(o/"prediction_change_people.csv",index=False);es=e.groupby("comparison")[["changed_fraction","corrected_fraction","harmed_fraction","ce_reduction"]].mean().reset_index();es.to_csv(o/"prediction_change_summary.csv",index=False)
print(s.to_string(index=False));print(es.to_string(index=False))
(o/"stage_analysis.html").write_text("""<!doctype html><html lang="zh"><meta charset="utf-8"><title>迁移的阶段分解</title><style>body{font:16px/1.6 system-ui;max-width:1150px;margin:30px auto}td,th{padding:6px;border-bottom:1px solid #ccc}</style><h1>探索性事后分解：类别与预测变化</h1><p>仅对已保存的seed17、30人预测做离线分析；未改变任何更新、超参数或选择。F1按5个固定类别计分，包括缺失类别的0值；阶段召回与CE只在该人存在该类别时有定义，分别对有定义的人求均值。CE reduction正值表示改善；F1差异正值表示改善。</p>"""+s.to_html(index=False,float_format=lambda x:f"{x:.5f}")+"<h2>按人平均的标签变化比例</h2>"+es.to_html(index=False,float_format=lambda x:f"{x:.6f}")+"""<p>交叉熵还会在预测类别不变时改变，因此CE收益不能直接代替Macro-F1收益。分解用于理解当前失败模式，不是新的确证检验。</p></html>""")
