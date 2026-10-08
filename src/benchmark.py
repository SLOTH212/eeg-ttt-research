from adaptation import *
from scipy.signal import welch
import resource,platform
results=[]
for kind in ['spectral','spectralhead','eegnet','eegnethead']:
 path=ROOT/f'results/{kind}_A_17/model.pt'
 if not path.exists():continue
 m,ck=load_model(path);x,y=load_subject(16,ck['kind']);a=Adapter(m,.001);xt=torch.tensor(np.array(x[:1]),dtype=torch.float32)
 for _ in range(20):a.candidate(xt)
 for mode in ['forward_only','candidate_then_reject','candidate_then_commit']:
  times=[]
  for i in range(300):
   xx=torch.tensor(np.array(x[i:i+1]),dtype=torch.float32);t=time.perf_counter()
   if mode=='forward_only':
    with torch.no_grad():m(xx)
   else:
    pp,c,features=a.candidate(xx)
    if mode.endswith('commit'):a.restore(c)
   times.append((time.perf_counter()-t)*1000)
  results.append({'model':kind,'operation':mode,'p50_ms':float(np.quantile(times,.5)),'p95_ms':float(np.quantile(times,.95)),'mean_ms':float(np.mean(times)),'n':300})
from controllers import fit_gate

def extract_features(x):
    f,p=welch(x,fs=100,nperseg=400,noverlap=200,axis=-1);edges=[.5,2,4,6,8,10,12,14,16,20,25,30,40]
    bands=np.stack([p[:,:,(f>=a)&(f<b)].sum(-1) for a,b in zip(edges[:-1],edges[1:])],axis=-1)
    logpower=np.log(bands+1e-10);rel=np.log((bands+1e-10)/(bands.sum(-1,keepdims=True)+1e-10));sd=x.std(-1);diff=np.diff(x,axis=-1).std(-1)/(sd+1e-6)
    return np.concatenate([logpower.reshape(len(x),-1),rel.reshape(len(x),-1),np.log(sd+1e-6),diff],axis=1).astype('float32')
raw=np.load(ROOT/'data/processed/s16_x.npy',mmap_mode='r')
for kind in ['spectral','spectralhead','eegnet','eegnethead']:
    out=ROOT/f'results/{kind}_A_17';m,ck=load_model(out/'model.pt')
    gate=fit_gate(json.loads((out/'controller_diagnostics.json').read_text())+json.loads((out/'source_replay_diagnostics.json').read_text()))
    rate=json.loads((out/'selection.json').read_text())['lr'];a=Adapter(m,rate);previous=None;times=[]
    for i in range(320):
        t=time.perf_counter();window=np.array(raw[i:i+1])*100.;window-=window.mean(-1,keepdims=True);window=np.clip(window/100.,-20,20)
        hf=extract_features(window);xx=hf if ck['kind']=='spectral' else window
        pp,c,ss=a.candidate(torch.from_numpy(xx));change=0. if previous is None else float(np.linalg.norm(hf[0]-previous)/np.sqrt(hf.shape[1]))
        signals=np.r_[ss,change,float(previous is None)];act=int(gate(signals))
        if act==1:a.restore(c)
        elif act==2:a.restore(a.source)
        previous=hf[0].copy()
        if i>=20:times.append((time.perf_counter()-t)*1000)
    results.append({'model':kind,'operation':'full_window_compute_preprocess_gradient_controller_action','p50_ms':float(np.quantile(times,.5)),'p95_ms':float(np.quantile(times,.95)),'mean_ms':float(np.mean(times)),'n':len(times)})
raw=np.load(ROOT/'data/processed/s16_x.npy',mmap_mode='r');times=[]
for i in range(300):
 t=time.perf_counter();xx=np.array(raw[i:i+1]);xx-=xx.mean(-1,keepdims=True);f,p=welch(xx,fs=100,nperseg=400,noverlap=200,axis=-1);times.append((time.perf_counter()-t)*1000)
results.append({'operation':'window_centering_and_welch_only','p50_ms':float(np.quantile(times,.5)),'p95_ms':float(np.quantile(times,.95))})
report={'results':results,'cpu_ids':sorted(os.sched_getaffinity(0)),'torch_threads':torch.get_num_threads(),'torch':torch.__version__,'platform':platform.platform(),'rss_peak_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'notes':'Wall-clock CPU benchmark under concurrent host load. Not isolated performance. Candidate includes forward, backward, projection, copy and reject/commit; the full_window_compute rows additionally include centering/scaling, complete Welch feature assembly, controller and state action; acquisition/EDF decoding excluded; actual closed-loop timings are in evaluation/supplement results. Welch entry excludes full feature assembly.'}
(ROOT/'results/benchmark.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
