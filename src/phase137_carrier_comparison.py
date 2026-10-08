
import os
os.sched_setaffinity(0,set(range(8,24)))
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];O=R/"phase137";O.mkdir(exist_ok=True)
specs={"cbramod":120,"labram":123,"eegnet":136}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pr=dict(purpose="Descriptive same-person three-carrier comparison, after all respective results observed",inputs={b:dict(path=str(R/f"phase{p}/participants.csv"),sha256=sha(R/f"phase{p}/participants.csv")) for b,p in specs.items()},conditions="EMA.5, native SleepEDF/source, left external montage, allavailable raw/full modes; original source-selected heads; allthree seeds retained",comparisons="CBr-EEGNet,LaBraM-EEGNet,CBr-LaBraM for each frozen/current/repeat/history; difference of matched history-repeat effects",statistics="Average metrics across seeds within each person, pairedpersonbootstrap10000 seed137017; descriptiveuncertainty, no architecture selection or newblindclaim",limitations="Same cohort repeated acrosscarriers, differentparentarchitecture/pretraining/sourcebudgets. Notcausalpretrainingablation; no equivalence claim from CI containingzero.")
(O/"protocol.json").write_text(json.dumps(pr,indent=2))
frames=[]
for b,p in specs.items():
 d=pd.read_csv(R/f"phase{p}/participants.csv");d["backbone"]=b;frames.append(d)
d=pd.concat(frames,ignore_index=True)
assert len(d)==24300
d=d[(d.ema==.5)&d.montage.isin(["native","left"])]
keys=["backbone","domain","subject","montage","mode","variant"]
assert (d.groupby(keys).seed.nunique()==3).all()
people=d.groupby(keys)[["macro_f1","balanced_accuracy","accuracy","ce"]].mean().reset_index()
people.to_csv(O/"people.csv",index=False)
summary=people.groupby(["backbone","domain","montage","mode","variant"])[["macro_f1","balanced_accuracy","accuracy","ce"]].mean().reset_index()
summary.to_csv(O/"summary.csv",index=False)
rows=[];effects=[];teachererrors=[]
pairs=[("cbramod","eegnet"),("labram","eegnet"),("cbramod","labram")]
for (domain,montage,mode),g in people.groupby(["domain","montage","mode"]):
 table=g.pivot(index="subject",columns=["backbone","variant"],values="macro_f1")
 for b in ["cbramod","labram"]:
  teachererrors.append(float(abs(table[(b,"teacher")]-table[("eegnet","teacher")]).max()))
 rng=np.random.default_rng(137017);ix=rng.integers(0,len(table),(10000,len(table)))
 for a,b in pairs:
  for variant in ["frozen","current","repeat11","history11"]:
   z=table[(a,variant)]-table[(b,variant)];ci=np.quantile(z.to_numpy()[ix].mean(1),[.025,.975])
   rows.append(dict(domain=domain,montage=montage,mode=mode,variant=variant,comparison=a+"-"+b,people=len(z),first_f1=table[(a,variant)].mean(),second_f1=table[(b,variant)].mean(),difference_pp=z.mean(),ci_low=ci[0],ci_high=ci[1],positive_people=int((z>0).sum()),harm_over1pp=int((z< -1).sum())))
  ea=table[(a,"history11")]-table[(a,"repeat11")];eb=table[(b,"history11")]-table[(b,"repeat11")]
  z=ea-eb;ci=np.quantile(z.to_numpy()[ix].mean(1),[.025,.975])
  effects.append(dict(domain=domain,montage=montage,mode=mode,comparison=a+"-"+b,people=len(z),first_history_effect=ea.mean(),second_history_effect=eb.mean(),effect_difference_pp=z.mean(),ci_low=ci[0],ci_high=ci[1]))
assert max(teachererrors)<1e-12
comp=pd.DataFrame(rows);eff=pd.DataFrame(effects)
comp.to_csv(O/"carrier_comparisons.csv",index=False);eff.to_csv(O/"history_effect_comparisons.csv",index=False)
show=comp[(comp.variant=="history11")&((comp.domain=="sleepedf")|((comp.domain.isin(["healthy","patient"]))&(comp["mode"]=="full")))]
aud=dict(status="PASS",original_streams=24300,person_metric_rows=len(people),same_teacher_max_error=max(teachererrors),allseeds_retained=True,paired_by_sameperson=True,input_hashes_rechecked=all(sha(Path(v["path"]))==v["sha256"] for v in pr["inputs"].values()))
assert aud["input_hashes_rechecked"]
(O/"audit.json").write_text(json.dumps(aud,indent=2))
(O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1450px;margin:35px auto}td,th{padding:5px;border:1px solid #ddd}</style><h1>三种载体的同人群静态历史读出比较</h1><p>固定原源选模头；对每个人先平均三个种子的指标，再进行同人配对比较。沿用EMA=0.5，SleepEDF原始输入、外部左侧full校准输入作为下表显示条件；完整raw/full及各头结果保留CSV。所有目标此前已观察，此为描述性横向比较，没有根据结果选择新模型。</p><p>EEGNet-style父模型为1661参数，历史头6021参数，合计7682；历史头容量必须计入。它的父模型训练预算、架构和预训练情况与两种基础模型不同，因此不能据此单独归因于预训练或证明模型等价。置信区间跨0表示尚不能确定差异，不是等效性结论。</p><h2>历史头对历史头：同受试者比较</h2>'+show.to_html(index=False,float_format=lambda x:f"{x:.6f}")+'<h2>相对各自等容量对照的历史效应差值</h2>'+eff.to_html(index=False,float_format=lambda x:f"{x:.6f}")+'<h2>完整固定EMA0.5均值</h2>'+summary.to_html(index=False,float_format=lambda x:f"{x:.6f}")+'<p><a href="../phase136/report.html">EEGNet原主比较</a> · <a href="../phase138/report.html">EEGNet CPU/GPU成本</a> · <a href="../phase133/report.html">基础模型流式成本</a></p>')
print(show.to_string(index=False));print(eff[eff.domain=="sleepedf"].to_string(index=False));print(aud)
