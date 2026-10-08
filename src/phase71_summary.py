from common import *
from phase2_temporal import ema
import pandas as pd,hashlib,warnings
warnings.filterwarnings("ignore",message="y_pred contains classes not in y_true")
o=ROOT/"phase71";assert (o/"evaluation_complete.json").exists();lock=json.loads((o/"selection_lock.json").read_text());rows=[];source_checks=[]
for seed in [29,43]:
 entry=lock["selections"][str(seed)];assert entry["selected"]["source_f1"]==max(c["source_f1"] for c in entry["candidates"])
 for c in entry["candidates"]:
  folder=Path(c["folder"]);meta=json.loads((folder/"complete.json").read_text());ck=torch.load(folder/"model.pt",map_location="cpu",weights_only=False);assert meta["classifier_seed"]==seed and ck["classifier_seed"]==seed
  assert hashlib.sha256((folder/"model.pt").read_bytes()).hexdigest()==c["checkpoint_sha256"]
  options=[meta["initial_val_f1"]]+[h["val_f1"] for h in meta["history"]];assert max(options)==meta["best_val_f1"]
  order=json.loads((folder/"training_orders.json").read_text());g=torch.Generator().manual_seed(seed);assert order["orders"]==[torch.randperm(meta["training_valid_windows"],generator=g).tolist() for _ in range(10)]
  initial=torch.load(ROOT/"phase63/pretrained_encoder_state.pt",map_location="cpu",weights_only=True);assert all(torch.equal(v,ck["state"]["encoder."+n]) for n,v in initial.items() if n.startswith("proj_out.") or n=="patch_embedding.mask_encoding")
  source_checks.append(dict(seed=seed,lr=c["lr"],best_epoch=c["best_epoch"],source_f1=c["source_f1"],actual_orders_verified=True,original_projection_unchanged=True))
for seed in [17,29,43]:
 oldfolder=ROOT/("phase21/evaluation/mixed_pretrained" if seed==17 else f"phase27/evaluation/mixed_pretrained_seed{seed}")
 for sid in lock["subjects"]:
  cbpath=ROOT/"phase70"/f"s{sid:02d}.npz" if seed==17 else o/"evaluation"/f"seed{seed}"/f"s{sid:02d}.npz";cb=np.load(cbpath);p=cb["pretrained" if seed==17 else "p"];lb=np.load(oldfolder/f"s{sid:02d}.npz");y=cb["y"];assert np.array_equal(y,lb["y"])
  root=ROOT/("data/processed" if sid<20 else "phase2/processed" if sid<40 else "phase6/processed");assert np.array_equal(y,np.load(root/f"s{sid:02d}_meta.npz")["y"])
  if seed!=17:
   meta=json.loads(cbpath.with_suffix(".json").read_text());assert meta["state_unchanged"] and meta["no_online_updates"]
  for name,prob in [("cbramod",p),("labram_matched_full",lb["source_tuned"])]:
   assert np.isfinite(prob).all() and abs(prob.sum(1)-1).max()<1e-5
   for smooth in [1.,.8,.5]:
    q=prob if smooth==1 else ema(prob,smooth);m=metrics(y,q)
    if name=="cbramod" and seed!=17:assert abs(m["macro_f1"]-meta["methods"][str(smooth)]["macro_f1"])<1e-12
    rows.append(dict(seed=seed,subject=sid,cohort="young" if sid<20 else "middle" if sid<40 else "older",model=name,ema=smooth,macro_f1=100*m["macro_f1"],accuracy=100*float((q[y>=0].argmax(1)==y[y>=0]).mean()),cross_entropy=m["ce"]))
df=pd.DataFrame(rows);assert len(df)==540;df.to_csv(o/"participants.csv",index=False);seedsummary=df.groupby(["seed","model","ema"])[["macro_f1","accuracy","cross_entropy"]].mean();seedsummary.to_csv(o/"seed_summary.csv")
person=df.groupby(["subject","cohort","model","ema"])[["macro_f1","accuracy","cross_entropy"]].mean().reset_index();person.to_csv(o/"seedmean_people.csv",index=False);summary=person.groupby(["model","ema"])[["macro_f1","accuracy","cross_entropy"]].mean();summary.to_csv(o/"seedmean_summary.csv")
old=pd.concat([pd.read_csv(ROOT/"phase4/all_metrics.csv"),pd.read_csv(ROOT/"phase6/all_metrics.csv")],ignore_index=True);strong=pd.read_csv(ROOT/"phase70/participants.csv");comparisons=[]
def compare(a,b,cohort,name):
 assert not a.isna().any() and not b.isna().any();delta=(a-b).to_numpy();rng=np.random.default_rng(20261006);ci=np.quantile(rng.choice(delta,(20000,len(delta))).mean(1),[.025,.975]);return dict(cohort=cohort,comparison=name,n_people=len(delta),cbramod_f1=float(a.mean()),reference_f1=float(b.mean()),difference_pp=float(delta.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]),positive_people=int((delta>0).sum()),harm_over_1pp=int((delta< -1).sum()),worst_person_pp=float(delta.min()))
for cohort in ["all","young","middle","older"]:
 group=person if cohort=="all" else person[person.cohort==cohort]
 for smooth in [1.,.8,.5]:
  a=group[(group.model=="cbramod")&(group.ema==smooth)].set_index("subject").macro_f1.sort_index();b=group[(group.model=="labram_matched_full")&(group.ema==smooth)].set_index("subject").macro_f1.reindex(a.index);comparisons.append(compare(a,b,cohort,f"vs_matched_full_LaBraM_3seeds_ema{smooth:g}"))
  b=strong[(strong.model=="labram")&(strong.ema==smooth)].set_index("subject").macro_f1.reindex(a.index);comparisons.append(compare(a,b,cohort,f"vs_strong_LaBraM_fixedseed17_ema{smooth:g}"))
  if smooth in [1.,.8]:
   oldmethod="frozen" if smooth==1 else "ema_08";b=old[(old.source=="mixed")&(old.variant=="context11")&(old.method==oldmethod)&old.seed.isin([17,29,43])].groupby("subject").f1.agg(["mean","count"]).reindex(a.index);assert (b["count"]==3).all();comparisons.append(compare(a,100*b["mean"],cohort,f"vs_context11_3seeds_ema{smooth:g}"))
comp=pd.DataFrame(comparisons);comp.to_csv(o/"paired_comparisons.csv",index=False);primary=comp[(comp.cohort=="all")&(comp.comparison=="vs_matched_full_LaBraM_3seeds_ema0.5")].iloc[0].to_dict();(o/"primary_result.json").write_text(json.dumps(primary,indent=2))
(o/"audit.json").write_text(json.dumps(dict(status="PASS",new_classifier_configs=4,new_target_classifier_person_cases=60,prediction_method_streams_recomputed=540,individual_people=30,classifier_seeds=[17,29,43],source_selections_and_hashes_verified=True,model_state_and_labels_checked=True,actual_epoch_orders_verified=True,no_probability_ensemble=True,no_seed_pseudoreplication=True,source_checks=source_checks),indent=2))
html='<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{font:15px/1.7 system-ui;max-width:1400px;margin:30px auto}td,th{padding:5px;border-bottom:1px solid #ddd}</style><h1>CBraMod完整分类器三种子静态迁移</h1><p>在单种子阶段70之后，固定架构、来源十人和验证四人、两步长十轮预算，新增分类训练顺序与dropout种子29/43；17复用。每个种子仅在源四人按原始Macro-F1选步长/轮次，含冻结第0轮；新种子目标预测之前保存选择锁。目标30人此前已观察，仍是探索性重复，不是新盲测。</p><h2>每人先平均三个分类器种子的指标</h2>'+summary.to_html(float_format=lambda x:f"{x:.5f}")+'<h2>各训练种子</h2>'+seedsummary.to_html(float_format=lambda x:f"{x:.5f}")+'<h2>全体配对比较</h2>'+comp[comp.cohort=="all"].to_html(index=False,float_format=lambda x:f"{x:.5f}")+'<h2>分年龄群体</h2>'+comp[comp.cohort!="all"].to_html(index=False,float_format=lambda x:f"{x:.5f}")+'<p>主参照使用阶段21/27相同十轮预算、三分类器种子的LaBraM；另列阶段40较强的单种子LaBraM参照及三种子历史11窗频谱基线。预训练数据、架构、池化和历史范围仍不同，不能隔离单一架构因素。区间按人重采样，先在人内平均种子，不集成概率、不把种子当新增受试者；未涵盖反复探索选择的不确定性。</p><p>这里全为静态分类，阶段69失败的熵更新没有目标运行，原阶段70单种子主比较仍独立保留。</p><p><a href="protocol.json">固定协议</a> · <a href="selection_lock.json">新种子来源选择锁</a> · <a href="audit.json">独立审计</a></p></html>';(o/"report.html").write_text(html);print(seedsummary.to_string());print(summary.to_string());print(comp[comp.cohort=="all"].to_string(index=False));print(primary)
