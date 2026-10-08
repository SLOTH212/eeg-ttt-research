from common import *
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import pandas as pd,pickle
OUT=ROOT/"phase20";a=pd.read_csv(ROOT/"phase16/all_pulses.csv");rows=[];details=[]
for source in ["young","mixed"]:
 with (ROOT/"phase14/probes"/f"{source}_pretrained400/model.pkl").open("rb") as f:carrier=pickle.load(f)
 signals={}
 for sid in [7,17,27,37]:
  root=ROOT/("data/processed" if sid<20 else "phase2/processed");hf=np.load(root/f"s{sid:02d}_meta.npz")["features"];z=np.load(ROOT/"phase14/features"/f"s{sid:02d}.npz")["pretrained"];p=carrier["model"].predict_proba(carrier["scaler"].transform(z));signals[sid]=(hf,p)
 for (objective,lr),g in a[a.source==source].groupby(["objective","lr"]):
  g=g.reset_index(drop=True);features=[]
  for v in g.itertuples():
   hf,p=signals[v.subject];q=p[v.t];entropy=float(-np.sum(q*np.log(np.maximum(q,1e-12))));distance=0. if v.t==0 else float(np.linalg.norm(hf[v.t]-hf[v.t-1])/np.sqrt(52));features.append(np.r_[entropy,q.max(),v.aux_loss,v.gradient_norm,q,distance,hf[v.t]])
  X=np.array(features);Y=g.future_ce_gain.to_numpy();S=g.subject.to_numpy();groups=dict(confidence=[0,1],signals=list(range(10)),signals_and_spectrum=list(range(X.shape[1])))
  for held in [7,17,27,37]:
   train=S!=held;test=S==held;methods={"keep_all":np.zeros(test.sum(),bool),"write_all":np.ones(test.sum(),bool),"low_entropy_05":X[test,0]<.5};predicted={};counts={s:int(np.sum(S[train]==s)) for s in np.unique(S[train])};weights=np.array([len(S[train])/(len(counts)*counts[s]) for s in S[train]])
   for name,idx in groups.items():
    reg=make_pipeline(StandardScaler(),Ridge(alpha=100.));reg.fit(X[train][:,idx],Y[train],ridge__sample_weight=weights);est=reg.predict(X[test][:,idx]);methods[name]=est>0;predicted[name]=est
   for method,decisions in methods.items():
    gain=Y[test]*decisions;row=dict(source=source,objective=objective,lr=lr,held_subject=held,method=method,pulses=int(test.sum()),mean_gain=float(gain.mean()),write_rate=float(decisions.mean()),harmful_write_rate=float(((Y[test]<-1e-6)&decisions).mean()),oracle_gain=float(np.maximum(Y[test],0).mean()),write_all_gain=float(Y[test].mean()));rows.append(row)
    for t,decision,realized in zip(g.loc[test,"t"],decisions,gain):details.append(dict(source=source,objective=objective,lr=lr,subject=held,t=int(t),method=method,write=bool(decision),realized_gain=float(realized)))
per=pd.DataFrame(rows);per.to_csv(OUT/"participant_metrics.csv",index=False);pd.DataFrame(details).to_csv(OUT/"decisions.csv",index=False);s=per.groupby(["source","objective","lr","method"])[["mean_gain","write_rate","harmful_write_rate","oracle_gain","write_all_gain"]].mean().reset_index();s.to_csv(OUT/"summary.csv",index=False);protocol=dict(scope="Exploratory SOURCE controller-subject leave-one-person-out freshwrite observability. Coldstart parameter state eachpulse, notretention oronline performance.",people=[7,17,27,37],pulse_source="phase16",labels="FutureH4CEgain used onlytraining3sourcepeople;heldpersonscoresoffline.",features="Whitelisted currentpredictiveentropy/maxprob,auxiliaryloss,gradientnorm,current5sourceprobabilities,current-vs-previous52featuredistance,current52spectralfeatures. No futureCE,gradientcosine,trueclassorlabel-derivedfield entersX.",models="Ridgealpha100,sourcefoldStandardScaler,equalpersonweights. Confidencealsolearnsfuturegain;featureablationonly.",baselines=["keep_all","write_all","fixedentropy<.5"],cpu_ids=list(range(8,24)),gpu=False);(OUT/"protocol.json").write_text(json.dumps(protocol,indent=2));print(s[s.lr==.01].to_string(index=False,float_format=lambda x:f"{x:.6f}"))
