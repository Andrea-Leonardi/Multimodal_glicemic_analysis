"""Optional integration check using only synthetic inputs and cached weights."""

import importlib.util
import unittest

import numpy as np

from src.data import ROOT, read_config


@unittest.skipUnless(importlib.util.find_spec("timestral") and (ROOT / ".cache/huggingface").exists(), "Install Timestral and download weights first")
class TimestralIntegrationTests(unittest.TestCase):
    def test_quantiles_and_batch_isolation(self):
        from src.forecasting import TimestralAdapter
        c = read_config()
        m = TimestralAdapter("timestral-28m", c["model_revisions"]["timestral-28m"], ROOT / ".cache/huggingface", threads=2)
        x = np.random.default_rng(42).normal(size=(1, 72)).cumsum(1) + 150
        cov = np.zeros((1, 2, 72))
        alone = m.predict(x, 36, cov)
        batched = m.predict(np.concatenate([x, x + 100]), 36, np.concatenate([cov, cov + 50]))
        self.assertEqual(alone.shape, (1, 36, 21))
        self.assertTrue(np.isfinite(alone).all())
        self.assertTrue((np.diff(alone, axis=-1) >= 0).all())
        np.testing.assert_allclose(alone[0], batched[0], atol=.002, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
