from pathlib import Path
import json,pandas as pd,numpy as np
r=Path(__file__).resolve().parents[1];out=r/"phase11";rows=[]
for f in out.glob("*_s*.json"):
 d=json.loads(f.read_text())
 if isinstance(d,list):rows.extend(d)
df=pd.DataFrame(rows);df["beneficial"]=df.actual_gain>1e-6;df["harmful"]=df.actual_gain< -1e-6;df["high_conf_wrong"]=(~df.current_correct)&(df.confidence>=.9);df["abs_error"]=(df.actual_gain-df.first_order_gain).abs();df.to_csv(out/"all_pulses.csv",index=False)
a=df.groupby(["case","cohort","subject","current_correct","future_stable"])[["actual_gain","gradient_cosine","beneficial","harmful","confidence"]].mean().reset_index();a.to_csv(out/"participant_strata.csv",index=False);s=a.groupby(["case","cohort","current_correct","future_stable"])[["actual_gain","gradient_cosine","beneficial","harmful","confidence"]].mean().reset_index();s.to_csv(out/"strata_summary.csv",index=False);print(s[(s.cohort=="older")&(s.future_stable)].to_string(index=False,float_format=lambda x:f"{x:.6f}"))
summary=[]
for (case,cohort),g in df.groupby(["case","cohort"]):
 by=g.groupby("subject")[["actual_gain","high_conf_wrong","abs_error"]].mean();summary.append(dict(case=case,cohort=cohort,subjects=g.subject.nunique(),pulses=len(g),mean_gain=by.actual_gain.mean(),high_conf_wrong_fraction=by.high_conf_wrong.mean(),first_order_mae=by.abs_error.mean(),first_order_correlation=g[["actual_gain","first_order_gain"]].corr().iloc[0,1]))
sumdf=pd.DataFrame(summary);sumdf.to_csv(out/"summary.csv",index=False);print(sumdf.to_string(index=False,float_format=lambda x:f"{x:.6f}"))
