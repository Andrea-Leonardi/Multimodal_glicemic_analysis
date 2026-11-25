"""Verify v3 provenance, learned updates, causal ledgers and outcome-free replay."""

from dataclasses import replace
from datetime import datetime, timezone
import json
try:
    import resource
except ImportError:
    resource = None  # type: ignore[assignment]
import time

import joblib
import numpy as np
import pandas as pd

from src.data import ROOT, sha256
from src.v3.common import atomic_json, load_forecast, periods, score, regimes
from src.v3.ensemble import apply_policy
from src.v3.predict import predict_at


def run(dataset, base, cfg, output, device):
    started=time.perf_counter()
    assert base==dataset.metadata['config'],'Base configuration differs from dataset'
    done=json.loads((output/'completed.json').read_text())
    manifest=json.loads((output/'final-manifest.json').read_text())['fingerprint']
    assert manifest==done['fingerprint'] and manifest['settings']==cfg and manifest['data']==dataset.metadata['fingerprint']
    for path in output.glob('*-manifest.json'):
        for name,expected in json.loads(path.read_text())['fingerprint']['code'].items():
            assert sha256(ROOT/name)==expected,(path.name,name)
    protected=json.loads((output/'v2-protected.json').read_text())
    for name,expected in protected.items():assert sha256(ROOT/name)==expected,('v2 changed',name)
    proof=json.loads((output/'selection-before-scoring.json').read_text())
    assert proof['test_metrics_exist'] is False
    assert proof['point_sha256']==sha256(output/'selection-frozen.json')
    assert proof['interval_sha256']==sha256(output/'interval-selection-frozen.json')
    selection=json.loads((output/'selection-frozen.json').read_text())
    interval_selection=json.loads((output/'interval-selection-frozen.json').read_text())
    artifacts={}
    phase_positions=periods(dataset,base)
    loaded={}
    for phase,pos in phase_positions.items():
        loaded[phase]={}
        for spec in selection['candidates']:
            path=output/'predictions'/f"{phase}_{spec['name']}.npz"
            loaded[phase][spec['name']]=load_forecast(path,pos)
            artifacts[str(path.relative_to(output))]=sha256(path)
            if spec.get('policy')=='weekly':
                with np.load(path) as content:assert (content['training_cutoff']<=dataset.times[pos]).all()
            if spec['origin']=='v3' and spec['family'] in {'trajectory','finetune'}:
                with np.load(path) as content:
                    assert content['curve'].shape==(len(pos),36) and np.isfinite(content['curve']).all()
                    np.testing.assert_allclose(content['curve'][:,[5,11,23,35]],content['prediction'],atol=1e-5)
                    if 'quantiles' in content:assert (np.diff(content['quantiles'],axis=-1)>=-1e-5).all()
        matrix=np.stack([loaded[phase][s['name']] for s in selection['candidates']],axis=-1)
        recomputed=apply_policy(matrix,regimes(dataset,pos,cfg),selection['weights'])
        selected=load_forecast(output/'predictions'/f'{phase}_selected_v3.npz',pos)
        np.testing.assert_array_equal(recomputed,selected)
        loaded[phase]['selected_v3']=selected
        for mode,weights in selection['policy_weights'].items():
            name='ensemble_'+mode
            pred=load_forecast(output/'predictions'/f'{phase}_{name}.npz',pos)
            np.testing.assert_array_equal(pred,apply_policy(matrix,regimes(dataset,pos,cfg),weights))
            loaded[phase][name]=pred

    fixed_models=weekly_models=0
    cutoff=pd.Timestamp(base['calibration_start'],tz=base['timezone']).timestamp()
    for path in (output/'models').glob('*.joblib'):
        artifact=joblib.load(path)
        assert artifact['fingerprint']==manifest
        assert artifact['label_upper_bound']<artifact['train_cutoff']==cutoff
        if artifact['spec']['family']=='finetune':
            keys=list(artifact['state'])
            allowed_last=artifact['spec']['last_block']
            assert keys and all(k.startswith('head.') or (allowed_last and k.startswith('transformer.layers.')) for k in keys)
            assert all(np.isfinite(value.numpy()).all() for value in artifact['state'].values())
        artifacts[str(path.relative_to(output))]=sha256(path)
        del artifact
        fixed_models+=1
    for path in (output/'weekly-models').glob('*/*/*.joblib'):
        artifact=joblib.load(path)
        assert artifact['label_upper_bound']<artifact['training_cutoff']
        assert artifact['training_origin_last']+10800+base['target_tolerance_seconds']<artifact['training_cutoff']
        artifacts[str(path.relative_to(output))]=sha256(path)
        del artifact
        weekly_models+=1

    summary=pd.read_csv(output/'test-metrics.csv')
    pos=phase_positions['test'];actual=np.asarray(dataset.targets[pos],np.float64)
    maximum=0.
    for name,pred in loaded['test'].items():
        measured=score(actual,pred.astype(np.float64))
        rows=summary.loc[summary.model.eq(name)].set_index('horizon_minutes').loc[base['horizons_minutes']]
        for key in ['mae','rmse']:
            difference=float(np.max(np.abs(rows[key].to_numpy()-measured[key])));maximum=max(maximum,difference)
            assert difference<1e-8,(name,key,difference)
        np.testing.assert_array_equal(rows.n,measured['n'])
    interval_rows=pd.read_csv(output/'test-intervals.csv')
    for mode in cfg['intervals']['modes']:
        path=output/'intervals'/f'test_{mode}.npz'
        with np.load(path) as content:
            np.testing.assert_array_equal(content['positions'],pos)
            if mode=='adaptive_rolling':assert (content['availability_upper']<dataset.times[pos,None]).all()
            for level in ['0.8','0.9']:
                lo,hi=content[level+'_lower'],content[level+'_upper']
                assert np.isfinite(lo).all() and np.isfinite(hi).all()
                assert (lo<=content['prediction']).all() and (content['prediction']<=hi).all()
            assert (content['0.9_lower']<=content['0.8_lower']).all() and (content['0.9_upper']>=content['0.8_upper']).all()
            rows=interval_rows.loc[interval_rows['mode'].eq(mode)].set_index('horizon_minutes')
            for k,h in enumerate(base['horizons_minutes']):
                valid=np.isfinite(actual[:,k]);lo,hi=content['0.9_lower'][:,k],content['0.9_upper'][:,k]
                coverage=np.mean((actual[valid,k]>=lo[valid])&(actual[valid,k]<=hi[valid]))
                np.testing.assert_allclose(rows.loc[h,'coverage90'],coverage,atol=1e-12)
        artifacts[str(path.relative_to(output))]=sha256(path)
    scale=joblib.load(output/'interval-scale.joblib')
    assert scale['label_upper_bound']<cutoff
    del scale
    replay=[]
    blind=replace(dataset,targets=None,target_indices=None)
    for row in [0,len(pos)//2,len(pos)-1]:
        at=pd.Timestamp(float(dataset.times[pos[row]]),unit='s',tz='UTC').isoformat()
        normal=predict_at(dataset,base,cfg,output,at,device)
        without=predict_at(blind,base,cfg,output,at,device)
        assert normal==without,'Outcome arrays changed prediction'
        difference=float(np.max(np.abs(np.asarray(normal['prediction_mgdl'])-loaded['test']['selected_v3'][row])))
        assert difference<=.02,('replay mismatch',difference)
        assert len(normal['dense_prediction_mgdl'])==36 and np.isfinite(normal['dense_prediction_mgdl']).all()
        replay.append({'origin':at,'targets_removed_identical':True,'maximum_difference_mgdl':difference})
        print(f'Replay {at}: identico senza target; scarto {difference:.6f} mg/dL',flush=True)
    for name in ['report.md','comparison.png','comparison.pdf','uncertainty.png','examples.png','examples.pdf','trajectory-shape.json','test-metrics.csv','test-cohorts.csv','test-intervals.csv','test-dense-metrics.csv','validation-metrics.csv','finetune-matched-validation.csv','selection-frozen.json','interval-selection-frozen.json']:
        path=output/name;assert path.is_file() and path.stat().st_size>0,name
        artifacts[name]=sha256(path)
    result={'status':'passed','utc':datetime.now(timezone.utc).isoformat(),'audit_code_sha256':sha256(ROOT/'src/v3/verify.py'),
            'seconds':time.perf_counter()-started,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024 if resource is not None else 0.0,
            'fixed_models':fixed_models,'weekly_models':weekly_models,'test_origins':len(pos),'metric_max_difference_mgdl':maximum,
            'replay':replay,'v2_unchanged':True,'artifact_sha256':artifacts}
    atomic_json(output/'verified.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='artifact_sha256'},indent=2),flush=True)
