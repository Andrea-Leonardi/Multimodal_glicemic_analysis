"""V3 invariants: actual gradient path, causal updates and complementary models."""

from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from src.v3.common import dense_targets, origin_weights, settings, training_positions
from src.personal_data import configuration
from src.v3.ensemble import greedy_weights, apply_policy
from src.v3.finetune import differentiable_forecast, trainable_parameters, FineTuner
from src.v3.rolling import cutoffs
from src.v3.uncertainty import available_history, bands, conformal_factors


class V3Tests(unittest.TestCase):
    def test_dense_labels_align_with_four_horizons_and_keep_gaps(self):
        times = np.arange(200.) * 300
        d = SimpleNamespace(times=times, glucose=np.arange(200.,dtype=np.float32))
        result = dense_targets(d,np.array([10,198]),configuration())
        np.testing.assert_array_equal(result[0,[5,11,23,35]],np.array([16,22,34,46]))
        self.assertTrue(np.isnan(result[1,1:]).all())

    def test_weekly_training_cannot_use_labels_after_update(self):
        times = pd.date_range('2026-01-01','2026-04-15',freq='5min',tz='UTC').as_unit('ns').asi8/1e9
        d=SimpleNamespace(times=times,eligible=np.ones(len(times),bool))
        base=configuration();cut=pd.Timestamp('2026-04-01',tz='UTC').timestamp()
        selected=training_positions(d,base,cut)
        self.assertTrue((times[selected]+10800+90<cut).all())
        epoch=cutoffs(np.array([cut,cut+6*86400,cut+8*86400]),cut,7)
        np.testing.assert_array_equal(epoch,np.array([cut,cut,cut+7*86400]))

    def test_training_weights_depend_on_known_context_only(self):
        d=SimpleNamespace(features=np.array([[0.,0.],[45.,0.],[-45.,30.],[0.,50.]]),feature_names=['glucose_delta_15m','carbs_sum_180m'])
        np.testing.assert_array_equal(origin_weights(d,np.arange(4),settings()),[1.,2.,2.,1.25])

    def test_worse_individual_forecast_can_improve_an_ensemble(self):
        target=np.arange(60.)
        # Candidate 4 is excluded by an individual top-three MAE ranking, but
        # cancels the common positive bias of the individually stronger models.
        p=np.column_stack([target+5,target+6,target+7,target-10])
        weights=greedy_weights(p,target,20)
        self.assertGreater(weights[3],0)
        self.assertLess(np.mean(np.abs(p@weights-target)),1.)
        self.assertAlmostEqual(weights.sum(),1.)
        self.assertTrue((weights>=0).all())

    def test_context_gating_uses_only_origin_regime(self):
        matrix=np.ones((4,4,2));matrix[:,:,1]=3
        weights=np.zeros((3,4,2));weights[:,:,0]=1;weights[2,:,0]=0;weights[2,:,1]=1
        out=apply_policy(matrix,np.array([0,1,2,0]),weights)
        np.testing.assert_array_equal(out[:,0],[1,1,3,1])

    def test_calibration_availability_includes_horizon_and_tolerance(self):
        times=np.arange(20.)*300
        cutoff=6000.
        mask=available_history(times,cutoff,30,90,28,1)
        self.assertTrue((times[mask]+1800+90<cutoff).all())
        self.assertFalse(mask[-1])

    def test_future_calibration_outcomes_cannot_change_rolling_bands(self):
        cfg=settings();cfg['intervals']['minimum_calibration_rows']=10
        base=configuration();times=np.arange(200.)*300
        pred=np.full((200,4),150.);actual=pred+np.arange(200.)[:,None]/10
        ledger={'times':times,'prediction':pred,'actual':actual.copy(),'scale':np.ones_like(pred),'group':np.zeros(200,int)}
        at=np.array([60000.]);p=np.full((1,4),150.)
        first,used=bands('adaptive_rolling',p,np.ones_like(p),np.zeros(1,int),{},cfg,base,times=at,ledger=ledger,anchor=60000.)
        # Last three hours are still unavailable at this origin.
        ledger['actual'][times+10800+90>=60000,3]=99999.
        second,_=bands('adaptive_rolling',p,np.ones_like(p),np.zeros(1,int),{},cfg,base,times=at,ledger=ledger,anchor=60000.)
        np.testing.assert_array_equal(first['0.9']['upper'][:,3],second['0.9']['upper'][:,3])
        self.assertTrue((used<at[:,None]).all())

    def test_finetune_inputs_do_not_change_with_future_glucose_or_events(self):
        from src.personal_timestral import PHYSICAL
        d=SimpleNamespace(times=np.arange(200.)*300,glucose=np.arange(200.,dtype=np.float32),features=np.ones((200,4),np.float32),feature_names=PHYSICAL)
        learner=object.__new__(FineTuner);learner.dataset=d;learner.base=configuration();learner.spec={'context_hours':1,'mode':'physical'}
        a=learner.inputs(np.array([100]));d.glucose[101:]=999;d.features[101:]=999
        b=learner.inputs(np.array([100]))
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)


class GradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def forecaster(self):
        import importlib.util
        if importlib.util.find_spec("timestral") is None:
            self.skipTest("Install timestral to run differentiable forecaster tests")
        import torch
        from timestral.config import TimestralConfig
        from timestral.model import TimestralModel
        from timestral.forecaster import TimestralForecaster
        torch.manual_seed(42)
        c=TimestralConfig(patch_size=4,d_model=16,num_heads=2,mlp_hidden_dim=32,
            block_schedule=('time','time','variate','time'),quantile_knots=(.1,.5,.9),block_size=4,context_length=64)
        return TimestralForecaster(TimestralModel(c),max_context=64,per_shot_horizon_patches=4)

    def test_differentiable_path_matches_native_with_covariates_and_missing(self):
        import torch
        f=self.forecaster();rng=np.random.default_rng(42)
        x=(150+rng.normal(size=(2,1,32))*10).astype(np.float32);x[0,0,3]=np.nan
        past=rng.normal(size=(2,1,32)).astype(np.float32);known=rng.normal(size=(2,1,40)).astype(np.float32)
        expected=f.forecast(x,prediction_length=8,past_covariates=past,known_covariates=known).quantiles
        actual=differentiable_forecast(f,x,past,known,horizon=8)
        torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-6)

    def test_optimizer_updates_only_head_and_requested_last_block(self):
        import torch
        for last in [False,True]:
            f=self.forecaster();parameters=trainable_parameters(f.model,last)
            before={name:p.detach().clone() for name,p in f.model.named_parameters()}
            optimizer=torch.optim.SGD(parameters.values(),lr=.001)
            x=np.linspace(120,180,32,dtype=np.float32)[None,None,:]
            predicted=differentiable_forecast(f,x,horizon=8)
            loss=(predicted-160).square().mean()/2500
            loss.backward()
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in parameters.values()))
            self.assertTrue(all(p.grad is None for n,p in f.model.named_parameters() if n not in parameters))
            optimizer.step()
            changed=[]
            for name,p in f.model.named_parameters():
                if not torch.equal(p,before[name]):changed.append(name)
                if name not in parameters:torch.testing.assert_close(p,before[name],atol=0,rtol=0)
            self.assertTrue(changed)
            self.assertTrue(all(n.startswith('head.') or (last and n.startswith('transformer.layers.3.')) for n in changed))


if __name__=='__main__':unittest.main()
