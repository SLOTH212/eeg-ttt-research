from common import *
from phase2_temporal import ema
import pandas as pd,hashlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
O=ROOT/"phase109";pr=json.loads((O/"protocol.json").read_text());assert json.loads((O/"audit.json").read_text())["status"]=="PASS";assert hashlib.sha256((O/"protocol.json").read_bytes()).hexdigest()==(O/"protocol.sha256").read_text();rows=[];binsrows=[];state_rows=[]
for sid in pr["subjects"]:
 folder=ROOT/("data/processed" if sid<20 else "phase2/processed");y=np.load(folder/f"s{sid:02d}_meta.npz")["y"];z=np.load(O/"predictions"/f"s{sid:02d}.npz");tr=np.load(O/"predictions"/f"s{sid:02d}_traces.npz");a=int(tr["downstep"]);b=int(tr["recovery"]);n=len(y);bins=np.full(n,"warmup",dtype="U24")
 for event,start,end in [("drop",a,b),("recovery",b,n)]:
  for lo,hi,label in [(0,10,"0_9"),(10,50,"10_49"),(50,200,"50_199"),(200,end-start,"200plus")]:
   if min(end,start+hi)>start+lo:bins[start+lo:min(end,start+hi)]=event+"_"+label
 for short in ["guard","floor"]:
  gain=tr[short+"_corrupted_gain"];baseline=tr[short+"_reference_gain"];effective=tr["factor"]*gain;ratio=effective/baseline
  for label in np.unique(bins):
   mask=bins==label;state_rows.append(dict(subject=sid,method=short,bin=label,windows=int(mask.sum()),median_effective_gain=float(np.median(effective[mask])),median_gain_vs_unperturbed_method=float(np.median(ratio[mask])),max_gain=float(gain[mask].max()),trigger_fraction=float((gain[mask]>1).mean())))
 for key in z.files:
  method,seed=key.rsplit("_cbramod_seed",1);p=z[key]
  for alpha in pr["ema"]:
   q=p if alpha==1 else ema(p,alpha);mm=metrics(y,q);cm=np.array(mm["confusion"]);base=dict(subject=sid,method=method,seed=int(seed),ema=alpha);rows.append(dict(**base,macro_f1=100*mm["macro_f1"],accuracy=100*cm.trace()/cm.sum(),ce=mm["ce"]))
   for label in np.unique(bins):
    mask=(bins==label)&(y>=0)
    if not mask.any():continue
    m=metrics(y[mask],q[mask]);c=np.array(m["confusion"]);binsrows.append(dict(**base,bin=label,n=int(mask.sum()),accuracy=100*c.trace()/c.sum(),ce=m["ce"]))
df=pd.DataFrame(rows);assert len(df)==216;df.to_csv(O/"participants.csv",index=False);person=df.groupby(["subject","method","ema"])[["macro_f1","accuracy","ce"]].mean().reset_index();person.to_csv(O/"seedmean_people.csv",index=False);summary=person.groupby(["method","ema"])[["macro_f1","accuracy","ce"]].mean().reset_index();summary.to_csv(O/"summary.csv",index=False)
comp=[]
for alpha,g in person.groupby("ema"):
 f=g.pivot(index="subject",columns="method",values="macro_f1")
 for new,old in [("corrupted_raw","reference_raw"),("corrupted_guard","reference_guard"),("corrupted_floor","reference_floor"),("corrupted_guard","corrupted_raw"),("corrupted_floor","corrupted_raw")]:
  d=f[new]-f[old];comp.append(dict(ema=alpha,new_method=new,old_method=old,people=4,new_f1=f[new].mean(),old_f1=f[old].mean(),difference_pp=d.mean(),positive_people=int((d>0).sum()),worst_pp=d.min()))
comp=pd.DataFrame(comp);comp.to_csv(O/"comparisons.csv",index=False);bd=pd.DataFrame(binsrows);bp=bd.groupby(["subject","method","ema","bin","n"])[["accuracy","ce"]].mean().reset_index();bp.to_csv(O/"bin_people.csv",index=False);bs=bp.groupby(["method","ema","bin"])[["accuracy","ce"]].mean().reset_index();bs.to_csv(O/"bin_summary.csv",index=False);pd.DataFrame(state_rows).to_csv(O/"state_bins.csv",index=False)
# First prespecified source person shown, not chosen on results.
sid=5;tr=np.load(O/"predictions"/f"s{sid:02d}_traces.npz");n=len(tr["factor"]);a=int(tr["downstep"]);b=int(tr["recovery"]);t=np.arange(n)*.5;fig,ax=plt.subplots(3,1,figsize=(12,9),sharex=True)
ax[0].plot(t,tr["guard_reference_rms"]*100,label="Original current RMS",lw=.7,alpha=.65);ax[0].plot(t,tr["guard_corrupted_rms"]*100,label="Perturbed current RMS",lw=.7,alpha=.65);ax[0].plot(t,np.exp(tr["guard_corrupted_log_state"])*100,label="Causal log-EMA estimate",lw=1.4);ax[0].axhline(json.loads((ROOT/"phase79/protocol.json").read_text())["lower_source_rms_quantile"]*100,color="gray",ls=":",label="Fixed source lower bound");ax[0].set_yscale("log");ax[0].set_ylabel("Joint RMS (uV)");ax[0].legend(fontsize=8,ncol=2)
for short,col in [("guard","#1767a2"),("floor","#d67d20")]:
 ax[1].plot(t,tr[short+"_corrupted_gain"],label=short,color=col);ax[2].plot(t,tr["factor"]*tr[short+"_corrupted_gain"]/tr[short+"_reference_gain"],label=short,color=col)
ax[1].set_ylabel("Applied gain");ax[1].legend();ax[2].set_ylabel("Effective gain / unperturbed control");ax[2].axhline(1,color="gray",ls=":");ax[2].set_yscale("log");ax[2].set_xlabel("Minutes from recording start")
for aa in ax:
 aa.axvline(a*.5,color="#a33",ls="--",lw=.9);aa.axvline(b*.5,color="#a33",ls="--",lw=.9);aa.grid(alpha=.2)
fig.suptitle("Fixed source stress: middle-third amplitude x0.1, then recovery\nPrespecified s05; existing methods unchanged; no gradient updates");fig.tight_layout(rect=(0,0,1,.94));fig.savefig(O/"state_response.png",dpi=140);plt.close(fig)
audit=dict(status="PASS",whole_record_streams=216,source_people=4,seeds=3,three_seed_metrics_within_person=True,boundaries_from_duration_only=True,existing_rules_unchanged=True,no_new_target_data=True,no_parameterupdates=True,plot_person_prespecified_first_source=True)
(O/"summary_audit.json").write_text(json.dumps(audit,indent=2))
(O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1400px;margin:35px auto}td,th{padding:6px;border:1px solid #ddd}</style><h1>固定源幅值下降与恢复压力测试</h1><p>4位既有源验证者的完整记录，中间三分之一两路信号统一乘0.1，其余时段不变；边界只依记录长度决定。沿用原始输入、固定guard与floor三种方法，每种均有未扰动参照；CBraMod三个原检查点冻结。人工缩放假设真实睡眠标签不变，只用于受控状态响应诊断，不能代表真实临床漂移。</p>'+comp[comp.ema==.5].to_html(index=False,float_format=lambda v:f"{v:.6f}")+'<p><img src="state_response.png" style="width:100%" alt="Fixed-source amplitude-step state response"></p><h2>幅值变化后的固定时间段</h2>'+bs[bs.ema==.5].to_html(index=False,float_format=lambda v:f"{v:.6f}")+'<p>时间段为突变后0–9、10–49、50–199和200窗后，均预先固定。分段类别组成不同，因此以accuracy/CE描述局部影响；全夜Macro-F1单独报告。曲线固定展示首位s05，不按表现选例。未调整EMA、阈值或学习模型；没有任何目标集参数更新。</p><p><a href="audit.json">因果与模型状态审计</a> · <a href="summary_audit.json">指标汇总</a> · <a href="state_bins.csv">输入状态分段</a></p>')
print(comp.to_string(index=False));print(audit)
