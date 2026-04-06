"""
Unit tests for app.prediction_client module.
Covers CSV source, API source, gap-horizon computation, and failure handling.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from app.prediction_client import (
    CsvPredictionSource,
    ApiPredictionSource,
    make_prediction_source,
    compute_gap_horizons,
    _no_prediction,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_pred_df():
    """Merged prediction DataFrame mimicking heuristic-strategy CSV format."""
    idx = pd.date_range("2020-01-01", periods=5, freq="h")
    data = {}
    for i in range(1, 7):
        data[f"Prediction_h_{i}"] = np.random.rand(5)
        data[f"Prediction_d_{i}"] = np.random.rand(5)
    df = pd.DataFrame(data, index=idx)
    df.index.name = "DATE_TIME"
    return df


@pytest.fixture
def csv_source(sample_pred_df):
    return CsvPredictionSource(sample_pred_df, num_hourly=6, num_daily=6)


# ---------------------------------------------------------------------------
# CsvPredictionSource
# ---------------------------------------------------------------------------

class TestCsvPredictionSource:
    def test_available_when_timestamp_exists(self, csv_source, sample_pred_df):
        dt = sample_pred_df.index[0]
        result = csv_source.get_prediction(dt, 1.1000, tp=1.1050, sl=1.0950, horizon=6)
        assert result["available"] is True
        assert len(result["hourly"]) == 6
        assert len(result["daily"]) == 6

    def test_unavailable_when_timestamp_missing(self, csv_source):
        missing_dt = pd.Timestamp("1999-01-01")
        result = csv_source.get_prediction(missing_dt, 1.1000, tp=1.1050, sl=1.0950, horizon=6)
        assert result["available"] is False

    def test_close_is_noop(self, csv_source):
        csv_source.close()  # should not raise


# ---------------------------------------------------------------------------
# ApiPredictionSource
# ---------------------------------------------------------------------------

class TestApiPredictionSource:
    def _mock_response(self, json_data, status_code=200):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = json_data
        resp.raise_for_status = MagicMock()
        return resp

    @patch("app.prediction_client.ApiPredictionSource.__init__", lambda self, *a, **kw: None)
    def _make_source(self):
        src = ApiPredictionSource.__new__(ApiPredictionSource)
        src._api_url = "http://test:8000/api/v1/predict"
        src._timeout = 2.0
        src._session = MagicMock()
        return src

    def test_success_with_binary_fields(self):
        src = self._make_source()
        src._session.post.return_value = self._mock_response({
            "long_binary_prediction": 1,
            "short_term_binary_prediction": 0,
        })
        result = src.get_prediction(pd.Timestamp("2020-01-01"), 1.1, tp=1.11, sl=1.09, horizon=6)
        assert result["available"] is True
        assert result["long_binary_prediction"] == 1
        assert result["short_term_binary_prediction"] == 0

    def test_server_error_returns_no_prediction(self):
        src = self._make_source()
        src._session.post.return_value = self._mock_response({}, status_code=500)
        result = src.get_prediction(pd.Timestamp("2020-01-01"), 1.1, tp=1.11, sl=1.09, horizon=6)
        assert result["available"] is False

    def test_timeout_returns_no_prediction(self):
        import requests
        src = self._make_source()
        src._session.post.side_effect = requests.exceptions.Timeout("timed out")
        result = src.get_prediction(pd.Timestamp("2020-01-01"), 1.1, tp=1.11, sl=1.09, horizon=6)
        assert result["available"] is False

    def test_malformed_json_returns_no_prediction(self):
        src = self._make_source()
        resp = self._mock_response({})
        resp.json.return_value = {"unexpected": "data"}
        src._session.post.return_value = resp
        result = src.get_prediction(pd.Timestamp("2020-01-01"), 1.1, tp=1.11, sl=1.09, horizon=6)
        assert result["available"] is False

    def test_timestamp_format(self):
        src = self._make_source()
        src._session.post.return_value = self._mock_response({
            "long_binary_prediction": 1,
            "short_term_binary_prediction": 0,
        })
        ts = pd.Timestamp("2020-03-15 14:00:00")
        src.get_prediction(ts, 1.1, tp=1.11, sl=1.09, horizon=6)
        call_args = src._session.post.call_args
        payload = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
        assert payload["timestamp"] == "15.03.2020 14:00:00.000"

    def test_close_closes_session(self):
        src = self._make_source()
        src.close()
        src._session.close.assert_called_once()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestMakePredictionSource:
    def test_csv_mode(self, sample_pred_df):
        config = {"prediction_source": "CSV"}
        src = make_prediction_source(config, pred_df=sample_pred_df, num_hourly=6, num_daily=6)
        assert isinstance(src, CsvPredictionSource)

    def test_api_mode(self):
        config = {
            "prediction_source": "API",
            "pp_api_url": "http://localhost:9999/api/v1/predict",
            "pp_timeout": 3.0,
        }
        src = make_prediction_source(config)
        assert isinstance(src, ApiPredictionSource)
        src.close()

    def test_csv_mode_requires_df(self):
        config = {"prediction_source": "CSV"}
        with pytest.raises(ValueError):
            make_prediction_source(config, pred_df=None)


# ---------------------------------------------------------------------------
# Gap-horizon computation
# ---------------------------------------------------------------------------

class TestComputeGapHorizons:
    def test_no_gaps(self):
        idx = pd.date_range("2020-01-01", periods=10, freq="h")
        horizons = compute_gap_horizons(idx)
        assert len(horizons) == 10
        # Last bar should have 0 ticks until "gap" (end of data)
        assert horizons[idx[-1]] == 0

    def test_gap_detected(self):
        # Create index with a 4-hour gap in the middle
        part1 = pd.date_range("2020-01-01 00:00", periods=5, freq="h")
        part2 = pd.date_range("2020-01-01 09:00", periods=5, freq="h")
        idx = part1.append(part2)
        horizons = compute_gap_horizons(idx, expected_freq_hours=1)
        # Bar at index 4 (last before gap) should have 0 ticks until gap
        assert horizons[part1[-1]] == 0
        # Bar at index 3 should have 1 tick until gap
        assert horizons[part1[-2]] == 1
