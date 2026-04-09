"""
Tests for src/data/loader.py.

yfinance is mocked so these tests run without network access.
"""

from __future__ import annotations

import pandas as pd
import pytest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_ohlcv() -> pd.DataFrame:
    """Minimal OHLCV DataFrame resembling yfinance output."""
    dates = pd.date_range("2023-01-03", periods=10, freq="B")
    return pd.DataFrame({
        "Date": dates,
        "Open": [150.0 + i for i in range(10)],
        "High": [155.0 + i for i in range(10)],
        "Low":  [148.0 + i for i in range(10)],
        "Close": [152.0 + i for i in range(10)],
        "Volume": [1_000_000 + i * 10_000 for i in range(10)],
    })


@pytest.fixture()
def tmp_raw_dir(tmp_path, monkeypatch):
    """Redirect RAW_DIR to a temporary directory for cache tests."""
    import src.data.loader as loader_mod
    import src.config as config_mod
    monkeypatch.setattr(config_mod, "RAW_DIR", tmp_path)
    monkeypatch.setattr(loader_mod, "RAW_DIR", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestGetPrices:

    def test_returns_long_format_dataframe(self, sample_ohlcv, tmp_raw_dir):
        """get_prices should return a long-format DataFrame with the right columns."""
        with patch("yfinance.download", return_value=sample_ohlcv):
            from src.data.loader import get_prices
            df = get_prices(["AAPL"], start="2023-01-01", end="2023-01-15")

        assert not df.empty
        required_cols = {"date", "ticker", "open", "high", "low", "close", "volume"}
        assert required_cols.issubset(set(df.columns)), (
            f"Missing columns: {required_cols - set(df.columns)}"
        )

    def test_ticker_column_populated(self, sample_ohlcv, tmp_raw_dir):
        with patch("yfinance.download", return_value=sample_ohlcv):
            from src.data.loader import get_prices
            df = get_prices(["MSFT"], start="2023-01-01", end="2023-01-15")
        assert (df["ticker"] == "MSFT").all()

    def test_date_column_is_datetime(self, sample_ohlcv, tmp_raw_dir):
        with patch("yfinance.download", return_value=sample_ohlcv):
            from src.data.loader import get_prices
            df = get_prices(["AAPL"], start="2023-01-01", end="2023-01-15")
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_failed_ticker_skipped_gracefully(self, tmp_raw_dir):
        """A ticker that raises on download should be skipped, not crash the whole call."""
        with patch("yfinance.download", side_effect=Exception("Network error")):
            from src.data.loader import get_prices
            df = get_prices(["BADDTICKER"], start="2023-01-01", end="2023-01-15")
        # Should return empty, not raise
        assert df.empty or len(df) == 0

    def test_empty_download_skipped(self, tmp_raw_dir):
        with patch("yfinance.download", return_value=pd.DataFrame()):
            from src.data.loader import get_prices
            df = get_prices(["AAPL"], start="2023-01-01", end="2023-01-15")
        assert df.empty

    def test_multiple_tickers_all_present(self, sample_ohlcv, tmp_raw_dir):
        with patch("yfinance.download", return_value=sample_ohlcv):
            from src.data.loader import get_prices
            df = get_prices(["AAPL", "MSFT"], start="2023-01-01", end="2023-01-15")
        tickers_returned = set(df["ticker"].unique())
        assert "AAPL" in tickers_returned
        assert "MSFT" in tickers_returned

    def test_sorted_by_ticker_and_date(self, sample_ohlcv, tmp_raw_dir):
        with patch("yfinance.download", return_value=sample_ohlcv):
            from src.data.loader import get_prices
            df = get_prices(["AAPL", "MSFT"], start="2023-01-01", end="2023-01-15")
        for ticker, group in df.groupby("ticker"):
            assert group["date"].is_monotonic_increasing, (
                f"Dates not sorted for {ticker}"
            )


class TestCaching:

    def test_cache_written_after_download(self, sample_ohlcv, tmp_raw_dir):
        """A parquet cache file should be created after a successful download."""
        with patch("yfinance.download", return_value=sample_ohlcv):
            from src.data.loader import get_prices
            get_prices(["AAPL"], start="2023-01-01", end="2023-01-15")
        cache_file = tmp_raw_dir / "AAPL.parquet"
        assert cache_file.exists(), "Cache file was not created."

    def test_cache_hit_avoids_redownload(self, sample_ohlcv, tmp_raw_dir):
        """Second call with same range should use cache; yfinance.download not called again."""
        with patch("yfinance.download", return_value=sample_ohlcv) as mock_dl:
            from src.data.loader import get_prices
            # First call: downloads and caches
            get_prices(["AAPL"], start="2023-01-01", end="2023-01-10")
            first_call_count = mock_dl.call_count
            # Second call: should hit cache
            get_prices(["AAPL"], start="2023-01-03", end="2023-01-08")
            assert mock_dl.call_count == first_call_count, (
                "yfinance.download was called again despite cache hit."
            )

    def test_force_refresh_bypasses_cache(self, sample_ohlcv, tmp_raw_dir):
        """force_refresh=True should always re-download even if cache exists."""
        with patch("yfinance.download", return_value=sample_ohlcv) as mock_dl:
            from src.data.loader import get_prices
            get_prices(["AAPL"], start="2023-01-01", end="2023-01-10")
            get_prices(["AAPL"], start="2023-01-01", end="2023-01-10", force_refresh=True)
            assert mock_dl.call_count == 2, (
                "force_refresh did not trigger a re-download."
            )
