from adaptation import *
seed_all(17);checks={}
for kind,shape in [('spectral',(32,52)),('eegnet',(32,2,3000))]:
    m=make_model(kind).eval();x=np.random.randn(*shape).astype('float32')
    a=Adapter(m,0);before=copy.deepcopy(a.model.state_dict());p,c,s=a.candidate(torch.from_numpy(x[:1]));a.restore(c)
    checks[kind+'_zero_lr_exact_state']=all(torch.equal(before[k],v) for k,v in a.model.state_dict().items())
    a=Adapter(m,.01);before=copy.deepcopy(a.model.state_dict());p,c,s=a.candidate(torch.from_numpy(x[:1]));a.restore(c)
    checks[kind+'_nonzero_update']=any(not torch.equal(before[k],v) for k,v in a.model.state_dict().items() if k in a.names)
    checks[kind+'_only_affine_changed']=all(torch.equal(before[k],v) for k,v in a.model.state_dict().items() if k not in a.names)
    a.restore(a.source);checks[kind+'_reset_exact']=all(torch.equal(before[k],v) for k,v in a.model.state_dict().items())
    pp,_,_=stream(m,x,'continuous');qq,_,_=stream(m,x[:12],'continuous');checks[kind+'_prefix_causal']=bool(np.array_equal(pp[:12],qq))
    zero,_,_=stream(m,x,'continuous',lr=0);frozen=predict(m,x)
    checks[kind+'_zero_lr_prediction_tolerance']=bool(np.allclose(zero,frozen,atol=1e-6,rtol=1e-5))
    checks[kind+'_first_prediction_before_update']=bool(np.allclose(pp[0],frozen[0],atol=1e-6))
    checks[kind+'_finite']=bool(np.isfinite(pp).all())
checks['cpu_affinity']=sorted(os.sched_getaffinity(0))==list(range(8,24))
checks['split_disjoint']=all(len(sum(s.values(),[]))==len(set(sum(s.values(),[]))) for s in CFG['folds'].values())
(ROOT/'results/implementation_audit.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2));assert all(checks.values())
