import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from retail_demand_forecasting import RetailDemandForecaster, create_lag_features


class BacktestTests(unittest.TestCase):
    def test_current_target_cannot_enter_rolling_features(self):
        dates = pd.date_range('2024-01-01', periods=45)
        frame = pd.DataFrame({
            'Date': dates, 'Store ID': 'S1', 'Product ID': 'P1',
            'Demand': np.arange(45, dtype=float),
        })
        original = create_lag_features(frame)
        frame.loc[40, 'Demand'] = 100000
        changed = create_lag_features(frame)
        columns = [c for c in original if c.startswith('demand_')]
        pd.testing.assert_series_equal(original.loc[40, columns], changed.loc[40, columns])
        self.assertNotEqual(original.loc[41, 'demand_lag_1'], changed.loc[41, 'demand_lag_1'])

    def test_bottom_up_totals_match_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = RetailDemandForecaster(str(Path(tmp) / 'unused.csv'))
            model.models['lightgbm'] = {
                'test_keys': pd.DataFrame({
                    'Date': pd.to_datetime(['2024-01-01'] * 3),
                    'Store ID': ['S1', 'S1', 'S2'],
                    'Product ID': ['P1', 'P2', 'P1'],
                }),
                'predictions': np.array([2., 3., 4.]),
                'actual': np.array([1., 2., 3.]),
            }
            levels = model.hierarchical_reconciliation()
            self.assertEqual(levels['total']['Predicted'].iloc[0], 9)
            self.assertEqual(levels['store']['Predicted'].sum(), 9)
            self.assertEqual(levels['product']['Predicted'].sum(), 9)


if __name__ == '__main__':
    unittest.main()
