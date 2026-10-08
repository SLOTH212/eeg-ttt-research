from phase112_contrast_augmentation import *
import pandas as pd
def run_summary():
 pr=json.loads((O/"protocol.json").read_text());selection=json.loads((O/"selection.json").read_text());assert json.loads((O/"audit.json").read_text())["status"]=="PASS";assert json.loads((O/"independent_augmentation_audit.json").read_text())["status"]=="PASS";rows=[];replay=[];native_repeat={}
 for sid in pr["validation"]:
  vx,vy,_=prepare(sid,"validation")
  for method in pr["methods"]:
   history=json.loads((O/method/"history.json").read_text())["history"]
   for label,file in [("selected","model.pt"),("final5_diagnostic","final5.pt")]:
    path=O/method/file;digest=hashlib.sha256(path.read_bytes()).hexdigest();ck=torch.load(path,map_location="cpu",weights_only=False)
    if label=="selected":assert digest==selection["selection"][method]["checkpoint_sha256"] and ck["epoch"]==selection["selection"][method]["epoch"]
    else:assert ck["epoch"]==5
    m=Carrier("pretrained").cuda();m.load_state_dict(ck["state"]);m.eval().requires_grad_(False);snap=snapshot(m);expected={(r["gain"],r["ratio"]):r["macro_f1"] for r in history[ck["epoch"]]["person_scores"] if r["subject"]==sid}
    for j,(g,r) in enumerate([(g,r) for g in [.1,1.,10.] for r in [.1,1.,10.]]):
     p=probabilities(m,vx[:,j]);score=100*metrics(vy,p)["macro_f1"];err=abs(score-expected[g,r]);assert err<1e-8;rows.append(dict(subject=sid,method=method,checkpoint=label,epoch=ck["epoch"],gain=g,ratio=r,macro_f1=score));replay.append(err)
     if method=="repeat_raw" and label=="final5_diagnostic" and g==1 and r==1:native_repeat[sid]=p
    assert all(torch.equal(v.cpu(),snap[k]) for k,v in m.state_dict().items());del m;torch.cuda.empty_cache()
  # Numerical repeatability comparison against phase94 is diagnostic, not a new source selection.
  old=torch.load(ROOT/"phase94/repeat_raw/resume.pt",map_location="cpu",weights_only=False)["state"];m=Carrier("pretrained").cuda();m.load_state_dict(old);m.eval().requires_grad_(False);p=probabilities(m,vx[:,4]);diff=float(abs(p-native_repeat[sid]).max());changes=int((p.argmax(1)!=native_repeat[sid].argmax(1)).sum());native_repeat[sid]=dict(max_probability_difference=diff,argmax_changes=changes);del m,old;torch.cuda.empty_cache();print("SOURCE REPLAY",sid,flush=True)
 df=pd.DataFrame(rows);assert len(df)==216;df.to_csv(O/"source_replay.csv",index=False);summary=df.groupby(["method","checkpoint","epoch"]).macro_f1.mean().rename("robust_f1").reset_index();native=df[(df.gain==1)&(df.ratio==1)].groupby(["method","checkpoint","epoch"]).macro_f1.mean().rename("native_f1").reset_index();summary=summary.merge(native,on=["method","checkpoint","epoch"]);summary.to_csv(O/"source_summary.csv",index=False)
 histories=[];counts=[]
 for method in pr["methods"]:
  d=json.loads((O/method/"history.json").read_text());histories.append(d["orders"]);counts.append(json.loads((O/method/"complete.json").read_text())["label_presentations"])
 assert all(x==histories[0] for x in histories) and len(set(counts))==1
 audit=dict(status="PASS",source_condition_model_rows=216,max_source_f1_replay_error=max(replay),same_order=True,same_label_presentations=counts[0],advance=selection["advance"],historical_repeat_probability_comparison=native_repeat,historical_parameter_bitwise_reproduction=False,numerical_difference_recorded_in="repeat_reproduction_audit.json",no_target_inference_yet=True)
 (O/"source_replay_audit.json").write_text(json.dumps(audit,indent=2))
 (O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1300px;margin:35px auto}td,th{padding:6px;border:1px solid #ddd}</style><h1>共同幅值与通道对比度增强：源seed17机制消融</h1><p>三个分支使用相同初始CBraMod、10位源训练者、3次呈现/窗口、5轮与固定训练顺序。共同幅值为固定随机因子及倒数，通道对比度分支额外改变两路相对强弱，并在裁剪前恢复到共同幅值分支相同的联合能量。裁剪可能破坏裁剪后的能量完全一致，计数单独保留。对照是本次同步训练的共同幅值分支，不能直接当作旧固定三幅值方法的严格复现。</p><p>九个固定源验证条件用于鲁棒均值；原输入的平均下降不超过1点、任何一人不超过2点才可选择，保留epoch0回退。所有源验证者此前已观察；九个视图不是九位独立受试者。</p>'+summary.to_html(index=False,float_format=lambda v:f"{v:.8f}")+'<p>进入目标描述评估的固定源条件：'+str(selection["advance"])+'。最终第5轮只作诊断，不可替代原定选中模型。没有新增独特源标签，每个分支408405次标签呈现。不是测试时训练。</p><p>重复原输入分支与旧实验的native F1训练轨迹一致，但参数最大差异约2.2e-6，未声明逐参数复现；同一运行中的配对条件及源检查点评分已独立复核。<a href="repeat_reproduction_audit.json">历史数值差异</a>与<a href="source_replay_audit.json">源概率/指标重放</a>均保留。</p><p><a href="protocol.json">固定协议</a> · <a href="independent_augmentation_audit.json">独立能量与前缀核对</a> · <a href="preparation_audit.json">裁剪与能量计数</a> · <a href="selection.json">原定选择</a></p>')
 print(summary.to_string(index=False));print(audit)
if __name__=="__main__":run_summary()
