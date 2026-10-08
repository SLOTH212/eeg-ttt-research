from adaptation import *
checks={};nfiles=0
for kind in ['spectral','spectralhead','eegnet','eegnethead']:
 for fold in ['A','B']:
  for seed in [17,29,43]:
   out=ROOT/f'results/{kind}_{fold}_{seed}'
   for name in ['summary.json','supplement.json','lr_sensitivity.json','policy_diagnostics.json','numerical_audit.json','identity_control.json']:
    assert (out/name).exists(),str(out/name);json.loads((out/name).read_text());nfiles+=1
   m,ck=load_model(out/'model.pt');split=CFG['folds'][fold]
   original=kind.removesuffix('head') if kind.endswith('head') else kind
   source=torch.load(ROOT/f'results/{original}_{fold}_{seed}/model.pt',map_location='cpu',weights_only=False)
   assert all(torch.equal(ck['state'][k],v) for k,v in source['state'].items())
   for sid in split['evaluation']:
    _,expected_y=load_subject(sid,ck['kind'])
    for prefix in ['predictions','supplement_predictions','lr_predictions','identity_predictions']:
     d=np.load(out/f'{prefix}_s{sid:02d}.npz');assert np.array_equal(d['y'],expected_y)
     for method in d.files:
      if method=='y':continue
      p=d[method];assert p.shape==(len(expected_y),5) and np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=2e-6),(out,sid,method)
    if kind.endswith('head'):
     a=np.load(out/f'predictions_s{sid:02d}.npz')['frozen'];b=np.load(ROOT/f'results/{original}_{fold}_{seed}/predictions_s{sid:02d}.npz')['frozen'];assert np.array_equal(a,b)
   num=json.loads((out/'numerical_audit.json').read_text());assert num['all_parameters_on_cpu'] and num['all_writable_fp32']
   checks[out.name]=True
# Learned-model causality/equivalence checks across both carriers and writable sets.
for kind in ['spectral','spectralhead','eegnet','eegnethead']:
 out=ROOT/f'results/{kind}_A_17';m,ck=load_model(out/'model.pt');x,y=load_subject(16,ck['kind']);hf=np.load(ROOT/'data/processed/s16_meta.npz')['features'];lr=json.loads((out/'selection.json').read_text())['lr']
 p,_,_=stream(m,x[:32],'continuous',lr,history_features=hf[:32]);saved=np.load(out/'predictions_s16.npz')['continuous'][:32];assert np.allclose(p,saved,atol=1e-6,rtol=1e-5)
 checks[kind+'_learned_prefix_matches_saved_full_stream']=True
for kind in ['spectral','eegnet']:
 for fold in ['A','B']:
  for seed in [17,29,43]:
   out=ROOT/f'results/{kind}_{fold}_{seed}';train=json.loads((out/'train.json').read_text());expected=30 if kind=='spectral' else 25;assert [r['epoch'] for r in train['history']]==list(range(1,expected+1))
   ck=torch.load(out/'model.pt',map_location='cpu',weights_only=False);assert abs(ck['val_f1']-max(r['val_macro_f1'] for r in train['history']))<1e-12
checks['source_checkpoint_selection_matches_validation_history']=True
checks['all_24_complete']=True;checks['all_predictions_finite_normalized_correct_timeline']=True;checks['frozen_predictions_identical_across_writable_sets']=True;checks['all_writable_fp32_and_cpu']=True
report={'checks':checks,'validated_result_json_files':nfiles,'status':'PASS'};(ROOT/'results/final_integrity_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
