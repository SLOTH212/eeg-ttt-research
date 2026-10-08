from pathlib import Path
import json,numpy as np,pandas as pd
r=Path(__file__).resolve().parents[1];out=r/"phase16";rows=[]
for f in out.glob("*/s[0-9][0-9].json"):
 d=json.loads(f.read_text());sid=d["subject"];root=r/("data/processed" if sid<20 else "phase2/processed");y=np.load(root/f"s{sid:02d}_meta.npz")["y"]
 for v in d["rows"]:
  yy=y[v["t"]+1:v["t"]+5];valid=yy[yy>=0];stable=bool(len(valid)==4 and np.all(valid==valid[0]));rows.append(dict(**v,future_stable=stable,future_stage=int(valid[0]) if stable else -1,beneficial=v["future_ce_gain"]>1e-6,harmful=v["future_ce_gain"]< -1e-6,oracle_keep_write=max(v["future_ce_gain"],0.)))
a=pd.DataFrame(rows);a.to_csv(out/"all_pulses.csv",index=False);cols=["future_ce_gain","beneficial","harmful","oracle_keep_write","gradient_cosine","gradient_norm","first_order_gain"]
per=a.groupby(["source","subject","objective","lr"])[cols].mean().reset_index();per.to_csv(out/"participant_metrics.csv",index=False);s=per.groupby(["source","objective","lr"])[cols].mean().reset_index();s.to_csv(out/"summary.csv",index=False)
strata=a.groupby(["source","subject","objective","lr","future_stage"])[cols].mean().reset_index().groupby(["source","objective","lr","future_stage"])[cols].mean().reset_index();strata.to_csv(out/"stage_strata.csv",index=False)
print(s.to_string(index=False,float_format=lambda x:f"{x:.6f}"));print("STAGE STRATA lr.01");print(strata[strata.lr==.01][["source","objective","future_stage","future_ce_gain","beneficial","harmful"]].to_string(index=False,float_format=lambda x:f"{x:.6f}"))
