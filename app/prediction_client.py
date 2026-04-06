"""
Prediction Client — abstracts CSV vs API prediction sources.

CSV mode: wraps a preloaded DataFrame of merged predictions (existing flow).
API mode: performs synchronous POST calls to the Prediction Provider entry/exit
          endpoints, using a persistent requests.Session for connection reuse.

The strategy code calls get_entry_prediction() / get_exit_prediction() and
receives a normalised result dict, regardless of source.
"""

import os as _os
_QUIET = _os.environ.get("STRATEGY_QUIET", "0") == "1"

import pandas as pd


# ---------------------------------------------------------------------------
# Normalised results returned to strategy
# ---------------------------------------------------------------------------

def _no_prediction():
    """Return a 'no action' sentinel."""
    return {"available": False}


def _entry_prediction(buy_binary, sell_binary):
    """Wrap entry signals."""
    return {
        "available": True,
        "buy_entry_binary": buy_binary,
        "sell_entry_binary": sell_binary,
    }


def _exit_prediction(exit_binary):
    """Wrap exit signal."""
    return {
        "available": True,
        "exit_binary": exit_binary,
    }


def _csv_prediction(hourly_list, daily_list):
    """Wrap CSV-sourced price-path predictions into normalised form."""
    return {
        "available": True,
        "hourly": hourly_list,
        "daily": daily_list,
    }


# ---------------------------------------------------------------------------
# CSV source — wraps existing DataFrame lookup (unchanged)
# ---------------------------------------------------------------------------

class CsvPredictionSource:
    """Reads predictions from a preloaded merged DataFrame (existing flow)."""

    def __init__(self, pred_df, num_hourly, num_daily):
        self.pred_df = pred_df
        self.num_hourly = num_hourly
        self.num_daily = num_daily

    def get_prediction(self, dt_hour, current_price, tp, sl, horizon):
        if dt_hour not in self.pred_df.index:
            return _no_prediction()
        row = self.pred_df.loc[dt_hour]
        try:
            hourly = [row.get(f'Prediction_h_{i}', current_price)
                      for i in range(1, self.num_hourly + 1)]
            daily = [row.get(f'Prediction_d_{i}', current_price)
                     for i in range(1, self.num_daily + 1)]
        except Exception:
            return _no_prediction()
        if all(pd.isna(daily)):
            return _no_prediction()
        return _csv_prediction(hourly, daily)

    def close(self):
        pass


# ---------------------------------------------------------------------------
# API source — calls PP entry/exit endpoints via requests.Session
# ---------------------------------------------------------------------------

class ApiPredictionSource:
    """Calls the Prediction Provider entry/exit endpoints."""

    def __init__(self, base_url, timeout=5.0):
        """
        Parameters
        ----------
        base_url : str
            Base URL of the PP API, e.g. ``http://127.0.0.1:8000``.
            Entry endpoint: ``{base_url}/api/v1/predict/entry``
            Exit  endpoint: ``{base_url}/api/v1/predict/exit``
        timeout : float
            HTTP request timeout in seconds.
        """
        import requests
        self._session = requests.Session()
        # Strip trailing /api/v1/predict/sync if user passed old-style URL
        self._base = base_url.rstrip("/")
        if self._base.endswith("/api/v1/predict/sync"):
            self._base = self._base.rsplit("/api/v1/predict/sync", 1)[0]
        elif self._base.endswith("/api/v1/predict/entry"):
            self._base = self._base.rsplit("/api/v1/predict/entry", 1)[0]
        self._entry_url = f"{self._base}/api/v1/predict/entry"
        self._exit_url = f"{self._base}/api/v1/predict/exit"
        self._info_url = f"{self._base}/api/v1/model/info"
        self._timeout = timeout

    def _fmt_ts(self, dt_hour):
        return dt_hour.strftime("%d.%m.%Y %H:%M:%S.000")

    def get_entry_prediction(self, dt_hour, tp_pips, sl_pips):
        """
        Ask PP: "should I open a buy or sell order at this tick?"

        Returns dict with keys: available, buy_entry_binary, sell_entry_binary.
        """
        import requests
        ts = self._fmt_ts(dt_hour)
        payload = {
            "datetime": ts,
            "tp": float(tp_pips),
            "sl": float(sl_pips),
        }
        try:
            resp = self._session.post(self._entry_url, json=payload,
                                      timeout=self._timeout)
            if resp.status_code >= 500:
                if not _QUIET:
                    print(f"[PredictionClient] Entry server error {resp.status_code} for {ts}")
                return _no_prediction()
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.Timeout:
            if not _QUIET:
                print(f"[PredictionClient] Entry timeout for {ts}")
            return _no_prediction()
        except Exception as exc:
            if not _QUIET:
                print(f"[PredictionClient] Entry request failed for {ts}: {exc}")
            return _no_prediction()

        buy_bin = data.get("buy_entry_binary")
        sell_bin = data.get("sell_entry_binary")
        if buy_bin is None and sell_bin is None:
            return _no_prediction()

        return _entry_prediction(buy_bin, sell_bin)

    def get_exit_prediction(self, dt_hour, direction, tp_price, sl_price):
        """
        Ask PP: "should I keep or close my open order?"

        Parameters
        ----------
        direction : str
            'buy' or 'sell'.
        tp_price : float
            Absolute take-profit price level.
        sl_price : float
            Absolute stop-loss price level.

        Returns dict with keys: available, exit_binary.
        """
        import requests
        ts = self._fmt_ts(dt_hour)
        payload = {
            "datetime": ts,
            "direction": direction,
            "tp_price": float(tp_price),
            "sl_price": float(sl_price),
        }
        try:
            resp = self._session.post(self._exit_url, json=payload,
                                      timeout=self._timeout)
            if resp.status_code >= 500:
                if not _QUIET:
                    print(f"[PredictionClient] Exit server error {resp.status_code} for {ts}")
                return _no_prediction()
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.Timeout:
            if not _QUIET:
                print(f"[PredictionClient] Exit timeout for {ts}")
            return _no_prediction()
        except Exception as exc:
            if not _QUIET:
                print(f"[PredictionClient] Exit request failed for {ts}: {exc}")
            return _no_prediction()

        exit_bin = data.get("exit_binary")
        if exit_bin is None:
            return _no_prediction()

        return _exit_prediction(exit_bin)

    def get_model_info(self):
        """Fetch predictor metadata (window_size, etc.)."""
        try:
            resp = self._session.get(self._info_url, timeout=self._timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return {}

    def close(self):
        try:
            self._session.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_prediction_source(config, pred_df=None, num_hourly=0, num_daily=0):
    """
    Build the right source based on config['prediction_source'].
    For CSV: pass the preloaded pred_df.
    For API: pred_df is ignored; uses config['pp_api_url'] and config['pp_timeout'].
    """
    source = config.get("prediction_source", "CSV").upper()
    if source == "API":
        url = config.get("pp_api_url", "http://127.0.0.1:8000")
        timeout = float(config.get("pp_timeout", 5.0))
        return ApiPredictionSource(url, timeout)
    else:
        if pred_df is None:
            raise ValueError("CSV prediction source requires a prediction DataFrame.")
        return CsvPredictionSource(pred_df, num_hourly, num_daily)


# ---------------------------------------------------------------------------
# Gap-horizon helper
# ---------------------------------------------------------------------------

def compute_gap_horizons(dt_index, expected_freq_hours=1):
    """
    Pre-compute ticks-until-next-gap for each bar in *dt_index*.

    A gap is a jump larger than 2× the expected bar frequency.  For each bar
    we store how many bars remain before that gap.  If no gap follows, we
    store the distance to the end of the dataset.

    Returns a dict {Timestamp: int}.
    """
    gap_threshold = pd.Timedelta(hours=expected_freq_hours * 2)
    n = len(dt_index)
    horizons = {}

    # Scan backwards so each bar inherits the nearest forward gap
    next_gap_at = n  # sentinel: no gap found yet
    for i in range(n - 1, -1, -1):
        if i < n - 1:
            delta = dt_index[i + 1] - dt_index[i]
            if delta > gap_threshold:
                next_gap_at = i + 1  # gap starts right after bar i
        horizons[dt_index[i]] = next_gap_at - i - 1  # ticks remaining before gap
    return horizons
