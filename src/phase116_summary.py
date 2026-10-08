from phase86_meta_waveform import *
from phase66_train import prepared,probabilities
from phase2_temporal import ema
import pandas as pd
O=ROOT/"phase116"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run():
 pr=json.loads((O/"protocol.json").read_text());assert sha(O/"protocol.json")==(O/"protocol.sha256").read_text().strip();assert json.loads((O/"audit.json").read_text())["status"]=="PASS"
 val={s:prepared(s) for s in pr["validation"]};rows=[];replays=[];choices=[]
 for sp in pr["models"]:
  seed=sp["seed"];ip=Path(sp["student"]["path"]);assert sha(ip)==sp["student"]["sha256"];base=torch.load(ip,map_location="cpu",weights_only=False)["state"]
  m=meta_model();m.load_state_dict(base);names=[n for n,p in ln_named(m)];assert sum(base[n].numel() for n in names)==9600
  for method in pr["methods"]:
   out=O/f"seed{seed}_{method}";done=json.loads((out/"complete.json").read_text());history=done["history"];initial={r["subject"]:r["macro_f1"] for r in history[0]["people"]};bn=float(np.mean(list(initial.values())))
   eligible=[]
   for h in history:
    ok=h["mean_f1"]>=bn-1 and all(v["macro_f1"]>=initial[v["subject"]]-2 for v in h["people"]);assert ok==h["eligible"]
    if ok:eligible.append(h)
   best=max(eligible,key=lambda h:h["mean_f1"]);assert best["epoch"]==done["selected_epoch"] and best["mean_f1"]==done["mean_f1"]
   assert sha(out/"selected_ln.pt")==done["selected_sha256"]
   choices.append(dict(seed=seed,method=method,selected_epoch=done["selected_epoch"],initial_f1=done["initial_f1"],selected_f1=done["mean_f1"],delta=done["mean_f1"]-done["initial_f1"],window_presentations=done["window_presentations"]))
   for stage,filename in [("selected","selected_ln.pt"),("final5","final5_ln.pt")]:
    ck=torch.load(out/filename,map_location="cpu",weights_only=False);assert ck["initial_checkpoint_sha256"]==sha(ip);assert set(ck["ln_state"])==set(names)
    m.load_state_dict(base);m.load_state_dict(ck["ln_state"],strict=False);m.eval()
    assert all(torch.equal(v.cpu(),base[k]) for k,v in m.state_dict().items() if k not in names)
    epoch=done["selected_epoch"] if stage=="selected" else 5
    for sid,(xx,y) in val.items():
     p=probabilities(m,xx);saved=np.load(out/f"s{sid:02d}.npz");expected=next(r["macro_f1"] for r in history[epoch]["people"] if r["subject"]==sid);f1=100*metrics(y,p)["macro_f1"];assert abs(f1-expected)<1e-10
     err=float(abs(p-saved["p"]).max()) if stage=="selected" else None
     if err is not None:assert err<2e-5 and np.array_equal(p.argmax(1),saved["p"].argmax(1))
     replays.append(dict(seed=seed,method=method,stage=stage,subject=sid,raw_macro_f1=f1,score_error=abs(f1-expected),saved_probability_error=err,unchanged_nonLN_exact=True))
     candidates={f"{stage}_{method}":p}
     if method==pr["methods"][0] and stage=="selected":
      candidates.update(frozen=saved["base"],teacher=saved["teacher"],static_half_blend=.5*saved["base"]+.5*saved["teacher"])
     for variant,prob in candidates.items():
      for alpha in [1.,.8,.5]:
       q=prob if alpha==1 else ema(prob,alpha);met=metrics(y,q);rows.append(dict(seed=seed,subject=sid,variant=variant,ema=alpha,macro_f1=100*met["macro_f1"],balanced_accuracy=100*met["balanced_accuracy"],ce=met["ce"]))
   print("REPLAY",seed,method,flush=True)
  del m;torch.cuda.empty_cache()
 df=pd.DataFrame(rows);assert len(df)==324;df.to_csv(O/"source_streams.csv",index=False);people=df.groupby(["subject","variant","ema"])[["macro_f1","balanced_accuracy","ce"]].mean().reset_index();people.to_csv(O/"people.csv",index=False);summary=people.groupby(["variant","ema"])[["macro_f1","balanced_accuracy","ce"]].mean().reset_index();summary.to_csv(O/"summary.csv",index=False);pd.DataFrame(choices).to_csv(O/"selections.csv",index=False)
 comparisons=[]
 for alpha in [1.,.8,.5]:
  table=people[people.ema==alpha].pivot(index="subject",columns="variant",values="macro_f1")
  for method in pr["methods"]:
   for ref in ["frozen","selected_true_label","teacher","static_half_blend"]:
    delta=table[f"selected_{method}"]-table[ref];comparisons.append(dict(method=method,reference=ref,ema=alpha,mean_delta_pp=float(delta.mean()),positive_people=int((delta>0).sum()),harm_over1pp=int((delta < -1).sum()),worst_delta_pp=float(delta.min())))
 comp=pd.DataFrame(comparisons);comp.to_csv(O/"comparisons.csv",index=False)
 audit=dict(status="PASS",reconstructed_condition_subjects=len(replays),source_metric_streams=len(df),max_f1_replay_error=max(r["score_error"] for r in replays),max_saved_probability_error=max(r["saved_probability_error"] or 0 for r in replays),selected_and_final_states_verified=True,selection_recomputed=True,only9600LN_changes=True,checkpoints=replays,no_targets_or_online_updates=True)
 (O/"source_replay_audit.json").write_text(json.dumps(audit,indent=2))
 (O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1300px;margin:35px auto}td,th{padding:5px;border:1px solid #ddd}</style><h1>静态LN蒸馏：相同参数和训练预算对照</h1><p>原10位源训练者、4位已观察源验证者，三个学生/教师种子。只训练教师WRITE相同的9600个LN参数，其他权重和缓冲区不变。真实标签、同种子教师软标签、固定半混合三个训练目标，均五轮、同窗口顺序、相同AdamW和人均衡权重；eval模式关闭dropout。教师训练目标来自训练集内预测，教师此前已学习这10人的标签，并有额外历史和训练预算。不能声称标签信息或总训练预算与没有教师的学生完全一致。</p><p>原始输入Macro-F1选最早最佳轮次，保留均值下降不超1点、每人不超2点的门槛，epoch0为回退。EMA仅为固定次要评分，不影响选模；最终第5轮单独保留，不能代替失败的主选择。所有结果仅源数据，属于离线校准，不是TTT。</p><h2>源选择</h2>'+pd.DataFrame(choices).to_html(index=False,float_format=lambda v:f"{v:.7f}")+'<h2>三个种子先在人内平均</h2>'+summary.to_html(index=False,float_format=lambda v:f"{v:.7f}")+'<h2>主比较：原始概率</h2>'+comp[comp.ema==1].to_html(index=False,float_format=lambda v:f"{v:.7f}")+'<p>每个小型LN检查点从原始完整模型重新装载；独立重算选模、验证未训练参数逐元素相同、所选及末轮指标可重现。<a href="source_replay_audit.json">重放审计</a> · <a href="protocol.json">协议</a> · <a href="comparisons.csv">全部比较</a></p>')
 print(pd.DataFrame(choices).to_string(index=False));print(summary[summary.ema==1].to_string(index=False));print(comp[(comp.ema==1)&(comp.method=="teacher_soft")].to_string(index=False))
if __name__=="__main__":
 with sdpa_kernel(SDPBackend.MATH):run()
