from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover - download is optional during static setup.
    yf = None


def get_plt():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional in headless contexts.
        raise RuntimeError("matplotlib is required for plotting.") from exc
    return plt


DEFAULT_FEATURES = ("ret", "vol20", "log_volume", "dlog_volume")


@dataclass(frozen=True)
class RevisionConfig:
    start_date: str = "2020-01-01"
    end_date: str = "2025-12-31"
    rolling_window: int = 60
    volatility_window: int = 20
    qews_window: int = 60
    min_coverage: float = 0.90
    features: tuple[str, ...] = DEFAULT_FEATURES
    robustness_windows: tuple[int, ...] = (40, 60, 90)


def zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if std is None or np.isclose(std, 0.0):
        return series * 0.0
    return (series - series.mean()) / std


def normalize_yfinance_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [str(level0) for level0, _ in frame.columns]
    frame.columns = [str(col).strip() for col in frame.columns]
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame[frame.index.notna()]
    frame = frame.sort_index()
    return frame


def read_market_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    # First handle the current yfinance CSV export, which may carry two header rows
    # plus an extra "Date" row marker.
    try:
        df = pd.read_csv(path, header=[0, 1], skiprows=[2], index_col=0, parse_dates=True)
        df = normalize_yfinance_frame(df)
        if "Close" in df.columns:
            return df
    except Exception:
        pass

    # Fallback to a standard single-header CSV.
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return normalize_yfinance_frame(df)


def load_tickers_from_excel(excel_path: str | Path, column: int = 1) -> list[str]:
    df = pd.read_excel(excel_path, header=None)
    tickers = df[column].astype(str).str.strip().tolist()
    tickers = [ticker for ticker in tickers if ticker and ticker.isalpha()]
    return sorted(dict.fromkeys(tickers))


def load_tickers_from_text(path: str | Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    raw = [line.strip() for line in text.splitlines() if line.strip()]
    return sorted(dict.fromkeys(raw))


def download_yahoo_panel(
    tickers: Sequence[str],
    index_ticker: str,
    config: RevisionConfig,
    out_dir: str | Path,
    force: bool = False,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    if yf is None:
        raise ImportError("yfinance is required for market-data download.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dict: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        csv_path = out_dir / f"{ticker}_{config.start_date}_{config.end_date}_daily.csv"
        if csv_path.exists() and not force:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        else:
            df = yf.download(
                ticker,
                start=config.start_date,
                end=config.end_date,
                auto_adjust=True,
                progress=False,
            )
            if df.empty:
                continue
            normalize_yfinance_frame(df).to_csv(csv_path)
        data_dict[ticker] = df

    index_path = out_dir / f"{index_ticker}_{config.start_date}_{config.end_date}_daily.csv"
    if index_path.exists() and not force:
        index_df = pd.read_csv(index_path, index_col=0, parse_dates=True)
    else:
        index_df = yf.download(
            index_ticker,
            start=config.start_date,
            end=config.end_date,
            auto_adjust=True,
            progress=False,
        )
        if index_df.empty:
            raise ValueError(f"No index data downloaded for {index_ticker}.")
        normalize_yfinance_frame(index_df).to_csv(index_path)

    return data_dict, index_df


def load_local_market_panel(
    data_dir: str | Path,
    tickers: Sequence[str],
    config: RevisionConfig,
    index_ticker: str = "^IXIC",
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    data_dir = Path(data_dir)
    data_dict: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        csv_path = data_dir / f"{ticker}_{config.start_date}_{config.end_date}_daily.csv"
        if not csv_path.exists():
            continue
        data_dict[ticker] = read_market_csv(csv_path)

    index_path = data_dir / f"{index_ticker}_{config.start_date}_{config.end_date}_daily.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index CSV: {index_path}")
    index_df = read_market_csv(index_path)
    return data_dict, index_df


def _wide_field(data_dict: dict[str, pd.DataFrame], field: str) -> pd.DataFrame:
    series_map = {}
    for ticker, df in data_dict.items():
        if field not in df.columns:
            continue
        series = pd.to_numeric(df[field], errors="coerce")
        series_map[ticker] = series
    if not series_map:
        return pd.DataFrame()
    return pd.DataFrame(series_map).sort_index()


def prepare_market_panels(
    data_dict: dict[str, pd.DataFrame],
    volatility_window: int = 20,
) -> dict[str, pd.DataFrame]:
    close = _wide_field(data_dict, "Close")
    volume = _wide_field(data_dict, "Volume")
    returns = np.log(close / close.shift(1))
    vol20 = returns.rolling(volatility_window).std()
    log_volume = np.log(volume.replace(0, np.nan))
    dlog_volume = log_volume.diff()

    return {
        "close": close,
        "volume": volume,
        "ret": returns,
        "vol20": vol20,
        "log_volume": log_volume,
        "dlog_volume": dlog_volume,
    }


def build_asset_feature_matrix(
    panels: dict[str, pd.DataFrame],
    date: pd.Timestamp,
    window: int,
    feature_names: Sequence[str],
    min_coverage: float,
) -> pd.DataFrame:
    if "ret" not in panels:
        raise KeyError("The return panel is required to build the asset-feature matrix.")

    base_index = panels["ret"].index
    end_loc = base_index.get_loc(date)
    if isinstance(end_loc, slice) or end_loc < window - 1:
        return pd.DataFrame()

    window_index = base_index[end_loc - window + 1 : end_loc + 1]
    min_obs = int(np.ceil(window * min_coverage))

    candidate_assets = None
    for feature in feature_names:
        if feature not in panels:
            raise KeyError(f"Unknown feature '{feature}'.")
        frame = panels[feature].reindex(window_index)
        valid_assets = set(frame.columns[frame.notna().sum(axis=0) >= min_obs])
        candidate_assets = valid_assets if candidate_assets is None else candidate_assets.intersection(valid_assets)

    if not candidate_assets:
        return pd.DataFrame()

    rows: dict[str, np.ndarray] = {}
    for asset in sorted(candidate_assets):
        vector_parts = []
        valid = True
        for feature in feature_names:
            series = pd.to_numeric(panels[feature].reindex(window_index)[asset], errors="coerce")
            if series.notna().sum() < min_obs:
                valid = False
                break
            if series.isna().any():
                series = series.interpolate(limit_direction="both").ffill().bfill()
            series = zscore(series)
            vector_parts.append(series.to_numpy(dtype=float))
        if not valid:
            continue
        rows[asset] = np.concatenate(vector_parts)

    if not rows:
        return pd.DataFrame()

    matrix = pd.DataFrame.from_dict(rows, orient="index")
    matrix = matrix.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    return matrix


def filter_return_window(
    returns_window: pd.DataFrame,
    min_coverage: float,
) -> pd.DataFrame:
    min_obs = int(np.ceil(len(returns_window) * min_coverage))
    filtered = returns_window.dropna(axis=1, thresh=min_obs)
    return filtered.dropna(axis=0, how="all")


def normalized_psd(matrix: np.ndarray, jitter: float = 1e-10) -> np.ndarray:
    matrix = np.array(matrix, dtype=float)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = 0.5 * (matrix + matrix.T)
    trace = np.trace(matrix)
    if not np.isfinite(trace) or trace <= 0:
        matrix = matrix + np.eye(matrix.shape[0]) * jitter
        trace = np.trace(matrix)
    return matrix / trace


def spectral_weights(psd_matrix: np.ndarray) -> np.ndarray:
    vals = np.linalg.eigvalsh(psd_matrix)
    vals = np.real(vals)
    vals = np.clip(vals, 0.0, None)
    total = vals.sum()
    if total <= 0:
        return np.array([1.0])
    return vals / total


def spectral_entropy(weights: np.ndarray, base: float = np.e) -> float:
    safe = weights[weights > 1e-12]
    if len(safe) == 0:
        return 0.0
    return float(-(safe * (np.log(safe) / np.log(base))).sum())


def effective_rank(weights: np.ndarray) -> float:
    return float(np.exp(spectral_entropy(weights)))


def participation_ratio(weights: np.ndarray) -> float:
    denom = float(np.square(weights).sum())
    if np.isclose(denom, 0.0):
        return 0.0
    return float(1.0 / denom)


def compute_classical_benchmarks(returns_window: pd.DataFrame) -> dict[str, float]:
    returns_window = returns_window.dropna(axis=1, how="all")
    if returns_window.shape[1] < 2:
        return {
            "num_assets": float(returns_window.shape[1]),
            "mean_abs_corr": np.nan,
            "cov_spectral_entropy": np.nan,
            "corr_spectral_entropy": np.nan,
            "effective_rank": np.nan,
            "participation_ratio": np.nan,
        }

    cov = returns_window.cov().to_numpy(dtype=float)
    corr = returns_window.corr().to_numpy(dtype=float)
    cov_rho = normalized_psd(cov)
    corr_rho = normalized_psd(corr)
    cov_weights = spectral_weights(cov_rho)
    corr_weights = spectral_weights(corr_rho)

    upper = np.triu_indices_from(corr, k=1)
    mean_abs_corr = float(np.nanmean(np.abs(corr[upper])))

    return {
        "num_assets": float(returns_window.shape[1]),
        "mean_abs_corr": mean_abs_corr,
        "cov_spectral_entropy": spectral_entropy(cov_weights),
        "corr_spectral_entropy": spectral_entropy(corr_weights),
        "effective_rank": effective_rank(cov_weights),
        "participation_ratio": participation_ratio(cov_weights),
    }


def compute_qna_metrics(snapshot: pd.DataFrame) -> dict[str, float]:
    if snapshot.empty or len(snapshot) < 2:
        return {
            "qna_num_assets": float(len(snapshot)),
        "qna_dimension": float(snapshot.shape[1] if not snapshot.empty else 0),
            "qna_entropy": np.nan,
            "qna_purity": np.nan,
            "eri": np.nan,
        }

    matrix = snapshot.to_numpy(dtype=float)
    norms = np.linalg.norm(matrix, axis=1)
    valid = norms > 1e-12
    matrix = matrix[valid]
    norms = norms[valid]
    if len(matrix) < 2:
        return {
            "qna_num_assets": float(len(matrix)),
            "qna_dimension": float(snapshot.shape[1]),
            "qna_entropy": np.nan,
            "qna_purity": np.nan,
            "eri": np.nan,
        }

    psi = matrix / norms[:, None]

    # Use the smaller Gram representation whenever possible. The non-zero
    # eigenvalues of psi.T @ psi / N and psi @ psi.T / N are identical, but the
    # asset-space Gram matrix is often much smaller and therefore substantially
    # faster for rolling evaluation.
    if psi.shape[0] <= psi.shape[1]:
        operator = normalized_psd((psi @ psi.T) / len(psi))
    else:
        operator = normalized_psd((psi.T @ psi) / len(psi))

    weights = spectral_weights(operator)
    purity = float(np.square(weights).sum())

    return {
        "qna_num_assets": float(len(psi)),
        "qna_dimension": float(snapshot.shape[1]),
        "qna_entropy": spectral_entropy(weights),
        "qna_purity": purity,
        "eri": float(1.0 - purity),
    }


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0.0, np.nan)


def add_standardized_benchmarks(metrics: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    metrics = metrics.copy()
    benchmark_cols = [
        "cov_spectral_entropy",
        "effective_rank",
        "participation_ratio",
        "mean_abs_corr",
        "realized_vol",
    ]
    for column in benchmark_cols:
        if column in metrics.columns:
            metrics[f"z_{column}"] = rolling_zscore(metrics[column], window)
    return metrics


def bootstrap_correlation(
    left: pd.Series,
    right: pd.Series,
    iterations: int = 1000,
    seed: int = 7,
) -> dict[str, float]:
    aligned = pd.concat([left, right], axis=1).dropna()
    if len(aligned) < 5:
        return {"corr": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    values = aligned.to_numpy(dtype=float)
    observed = float(np.corrcoef(values[:, 0], values[:, 1])[0, 1])

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(iterations):
        idx = rng.integers(0, len(values), size=len(values))
        sample = values[idx]
        draws.append(float(np.corrcoef(sample[:, 0], sample[:, 1])[0, 1]))

    ci_low, ci_high = np.quantile(draws, [0.025, 0.975])
    return {"corr": observed, "ci_low": float(ci_low), "ci_high": float(ci_high)}


def build_index_risk_proxies(index_df: pd.DataFrame, future_window: int = 20) -> pd.DataFrame:
    close_col = "Close" if "Close" in index_df.columns else "Adj Close"
    close = pd.to_numeric(index_df[close_col], errors="coerce")
    close = close.dropna()

    log_ret = np.log(close / close.shift(1))
    realized_vol = log_ret.rolling(future_window).std()

    future_min = close.shift(-1).rolling(future_window).min()
    future_drawdown = future_min / close - 1.0

    return pd.DataFrame(
        {
            "index_close": close,
            "index_log_ret": log_ret,
            "realized_vol": realized_vol,
            "future_drawdown": future_drawdown,
        }
    )


def compute_revision_metrics(
    data_dict: dict[str, pd.DataFrame],
    index_df: pd.DataFrame,
    config: RevisionConfig,
    feature_names: Sequence[str] | None = None,
    rolling_window: int | None = None,
) -> pd.DataFrame:
    feature_names = tuple(feature_names or config.features)
    window = rolling_window or config.rolling_window

    panels = prepare_market_panels(data_dict, volatility_window=config.volatility_window)
    returns = panels["ret"].copy()
    risk_df = build_index_risk_proxies(index_df)

    rows: list[dict[str, float | pd.Timestamp]] = []
    for date in returns.index:
        end_loc = returns.index.get_loc(date)
        if isinstance(end_loc, slice) or end_loc < window:
            continue

        returns_window = returns.iloc[end_loc - window + 1 : end_loc + 1]
        returns_window = filter_return_window(returns_window, config.min_coverage)
        if returns_window.shape[1] < 2:
            continue

        try:
            snapshot = build_asset_feature_matrix(
                panels=panels,
                date=date,
                window=window,
                feature_names=feature_names,
                min_coverage=config.min_coverage,
            )
        except KeyError:
            continue

        classical = compute_classical_benchmarks(returns_window)
        qna = compute_qna_metrics(snapshot)

        row: dict[str, float | pd.Timestamp] = {"date": pd.Timestamp(date)}
        row.update(classical)
        row.update(qna)
        rows.append(row)

    metrics = pd.DataFrame(rows).set_index("date").sort_index()
    if metrics.empty:
        return metrics

    metrics["qews_eri"] = rolling_zscore(metrics["eri"], config.qews_window)
    metrics["qews_entropy"] = rolling_zscore(metrics["qna_entropy"], config.qews_window)

    merged = metrics.join(risk_df, how="left")
    return merged


def run_feature_robustness(
    data_dict: dict[str, pd.DataFrame],
    index_df: pd.DataFrame,
    config: RevisionConfig,
    feature_sets: dict[str, Sequence[str]],
) -> pd.DataFrame:
    frames = []
    for label, feature_names in feature_sets.items():
        df = compute_revision_metrics(
            data_dict=data_dict,
            index_df=index_df,
            config=config,
            feature_names=feature_names,
        )
        if df.empty:
            continue
        df = df.copy()
        df["feature_spec"] = label
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames)


def run_window_robustness(
    data_dict: dict[str, pd.DataFrame],
    index_df: pd.DataFrame,
    config: RevisionConfig,
    windows: Iterable[int] | None = None,
) -> pd.DataFrame:
    frames = []
    for window in windows or config.robustness_windows:
        df = compute_revision_metrics(
            data_dict=data_dict,
            index_df=index_df,
            config=config,
            rolling_window=int(window),
        )
        if df.empty:
            continue
        df = df.copy()
        df["rolling_window"] = int(window)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames)


def bootstrap_mean_difference(
    series: pd.Series,
    left_index: Sequence[pd.Timestamp],
    right_index: Sequence[pd.Timestamp],
    iterations: int = 1000,
    seed: int = 7,
    absolute: bool = False,
) -> dict[str, float]:
    left = series.loc[list(left_index)].dropna().to_numpy(dtype=float)
    right = series.loc[list(right_index)].dropna().to_numpy(dtype=float)
    if len(left) == 0 or len(right) == 0:
        return {"observed_diff": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    rng = np.random.default_rng(seed)
    left_eval = np.abs(left) if absolute else left
    right_eval = np.abs(right) if absolute else right
    observed = float(left_eval.mean() - right_eval.mean())
    draws = []
    for _ in range(iterations):
        left_draw = rng.choice(left, size=len(left), replace=True)
        right_draw = rng.choice(right, size=len(right), replace=True)
        if absolute:
            left_draw = np.abs(left_draw)
            right_draw = np.abs(right_draw)
        draws.append(float(left_draw.mean() - right_draw.mean()))

    ci_low, ci_high = np.quantile(draws, [0.025, 0.975])
    return {"observed_diff": observed, "ci_low": float(ci_low), "ci_high": float(ci_high)}


def permutation_mean_difference(
    series: pd.Series,
    left_index: Sequence[pd.Timestamp],
    right_index: Sequence[pd.Timestamp],
    iterations: int = 3000,
    seed: int = 7,
    absolute: bool = False,
) -> dict[str, float]:
    left = series.loc[list(left_index)].dropna().to_numpy(dtype=float)
    right = series.loc[list(right_index)].dropna().to_numpy(dtype=float)
    if len(left) == 0 or len(right) == 0:
        return {"perm_pvalue": np.nan}

    if absolute:
        left = np.abs(left)
        right = np.abs(right)

    observed = float(left.mean() - right.mean())
    pool = np.concatenate([left, right])
    n_left = len(left)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(iterations):
        perm = rng.permutation(pool)
        diff = float(perm[:n_left].mean() - perm[n_left:].mean())
        exceed += abs(diff) >= abs(observed)
    pvalue = (exceed + 1) / (iterations + 1)
    return {"perm_pvalue": float(pvalue)}


def evaluate_event_windows(
    metrics: pd.DataFrame,
    events: pd.DataFrame,
    signal_col: str = "qews_eri",
    pre_days: int = 20,
    post_days: int = 20,
    bootstrap_iterations: int = 1000,
    absolute: bool = False,
    permutation_iterations: int = 3000,
) -> pd.DataFrame:
    if metrics.empty or events.empty:
        return pd.DataFrame()

    metrics = metrics.sort_index()
    results = []
    for _, event in events.iterrows():
        event_date = pd.Timestamp(event["event_date"])
        if event_date not in metrics.index:
            nearest = metrics.index[metrics.index.get_indexer([event_date], method="nearest")]
            if len(nearest) == 0:
                continue
            event_date = nearest[0]

        loc = metrics.index.get_loc(event_date)
        if isinstance(loc, slice):
            continue

        pre_index = metrics.index[max(0, loc - pre_days) : loc]
        post_index = metrics.index[loc + 1 : loc + 1 + post_days]
        boot = bootstrap_mean_difference(
            metrics[signal_col],
            left_index=pre_index,
            right_index=post_index,
            iterations=bootstrap_iterations,
            absolute=absolute,
        )
        perm = permutation_mean_difference(
            metrics[signal_col],
            left_index=pre_index,
            right_index=post_index,
            iterations=permutation_iterations,
            absolute=absolute,
        )
        results.append(
            {
                "event_label": event["event_label"],
                "event_date": event_date,
                "event_type": event.get("event_type", ""),
                "signal_col": signal_col,
                "pre_mean": float(metrics.loc[pre_index, signal_col].mean()) if len(pre_index) else np.nan,
                "post_mean": float(metrics.loc[post_index, signal_col].mean()) if len(post_index) else np.nan,
                "pre_abs_mean": float(metrics.loc[pre_index, signal_col].abs().mean()) if len(pre_index) else np.nan,
                "post_abs_mean": float(metrics.loc[post_index, signal_col].abs().mean()) if len(post_index) else np.nan,
                **boot,
                **perm,
            }
        )

    return pd.DataFrame(results)


def load_event_catalog(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["event_date"])
    required = {"event_label", "event_date", "event_type"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Event catalog missing columns: {sorted(missing)}")
    return df.sort_values("event_date").reset_index(drop=True)


def summarize_metrics(metrics: pd.DataFrame) -> pd.Series:
    wanted = [
        "mean_abs_corr",
        "cov_spectral_entropy",
        "corr_spectral_entropy",
        "effective_rank",
        "participation_ratio",
        "qna_entropy",
        "qna_purity",
        "eri",
        "qews_eri",
        "qews_entropy",
        "realized_vol",
    ]
    available = [column for column in wanted if column in metrics.columns]
    return metrics[available].agg(["mean", "std", "min", "max"]).stack()


def summarize_benchmark_relationships(
    metrics: pd.DataFrame,
    bootstrap_iterations: int = 1000,
) -> pd.DataFrame:
    metrics = add_standardized_benchmarks(metrics, window=60)
    rows = []

    level_pairs = {
        "QNA entropy": [
            ("Covariance spectral entropy", "cov_spectral_entropy"),
            ("Effective rank", "effective_rank"),
            ("Participation ratio", "participation_ratio"),
            ("Mean absolute correlation", "mean_abs_corr"),
            ("Realized volatility", "realized_vol"),
        ],
        "ERI": [
            ("Covariance spectral entropy", "cov_spectral_entropy"),
            ("Effective rank", "effective_rank"),
            ("Participation ratio", "participation_ratio"),
            ("Mean absolute correlation", "mean_abs_corr"),
            ("Realized volatility", "realized_vol"),
        ],
    }
    series_lookup = {"QNA entropy": "qna_entropy", "ERI": "eri"}

    for qna_label, benchmarks in level_pairs.items():
        qna_col = series_lookup[qna_label]
        for benchmark_label, benchmark_col in benchmarks:
            stats = bootstrap_correlation(
                metrics[qna_col],
                metrics[benchmark_col],
                iterations=bootstrap_iterations,
            )
            rows.append(
                {
                    "panel": "Level relationships",
                    "qna_metric": qna_label,
                    "benchmark_metric": benchmark_label,
                    **stats,
                }
            )

    deviation_pairs = [
        ("Rolling-z covariance spectral entropy", "z_cov_spectral_entropy"),
        ("Rolling-z effective rank", "z_effective_rank"),
        ("Rolling-z mean absolute correlation", "z_mean_abs_corr"),
        ("Rolling-z realized volatility", "z_realized_vol"),
    ]
    for benchmark_label, benchmark_col in deviation_pairs:
        stats = bootstrap_correlation(
            metrics["qews_eri"],
            metrics[benchmark_col],
            iterations=bootstrap_iterations,
        )
        rows.append(
            {
                "panel": "Deviation relationships",
                "qna_metric": "QEWS (ERI)",
                "benchmark_metric": benchmark_label,
                **stats,
            }
        )

    return pd.DataFrame(rows)


def summarize_feature_robustness(feature_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, df in feature_metrics.groupby("feature_spec"):
        df = add_standardized_benchmarks(df.set_index("date"), window=60)
        rows.append(
            {
                "specification": label,
                "qna_dimension_mean": float(df["qna_dimension"].mean()),
                "qna_entropy_mean": float(df["qna_entropy"].mean()),
                "qna_entropy_std": float(df["qna_entropy"].std()),
                "eri_mean": float(df["eri"].mean()),
                "corr_entropy_cov": float(df[["qna_entropy", "cov_spectral_entropy"]].dropna().corr().iloc[0, 1]),
                "corr_entropy_effrank": float(df[["qna_entropy", "effective_rank"]].dropna().corr().iloc[0, 1]),
                "corr_qews_covz": float(df[["qews_eri", "z_cov_spectral_entropy"]].dropna().corr().iloc[0, 1]),
            }
        )
    return pd.DataFrame(rows)


def summarize_window_robustness(window_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window, df in window_metrics.groupby("rolling_window"):
        df = add_standardized_benchmarks(df.set_index("date"), window=60)
        rows.append(
            {
                "rolling_window": int(window),
                "qna_dimension_mean": float(df["qna_dimension"].mean()),
                "qna_entropy_mean": float(df["qna_entropy"].mean()),
                "qna_entropy_std": float(df["qna_entropy"].std()),
                "eri_mean": float(df["eri"].mean()),
                "corr_entropy_cov": float(df[["qna_entropy", "cov_spectral_entropy"]].dropna().corr().iloc[0, 1]),
                "corr_entropy_effrank": float(df[["qna_entropy", "effective_rank"]].dropna().corr().iloc[0, 1]),
                "corr_qews_covz": float(df[["qews_eri", "z_cov_spectral_entropy"]].dropna().corr().iloc[0, 1]),
            }
        )
    return pd.DataFrame(rows)


def plot_entropy_comparison(metrics: pd.DataFrame, save_path: str | Path | None = None) -> None:
    plt = get_plt()
    if metrics.empty:
        raise ValueError("No metrics available for plotting.")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(metrics.index, metrics["cov_spectral_entropy"], label="Covariance Spectral Entropy", lw=1.8)
    ax.plot(metrics.index, metrics["qna_entropy"], label="QNA Entropy", lw=1.8)
    ax.set_title("Classical Spectral Entropy vs QNA Entropy")
    ax.set_ylabel("Entropy")
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300)


def plot_standardized_benchmark_series(
    metrics: pd.DataFrame,
    save_path: str | Path | None = None,
    focus_start: str = "2024-01-01",
    focus_end: str = "2025-12-31",
) -> None:
    plt = get_plt()
    if metrics.empty:
        raise ValueError("No metrics available for plotting.")

    subset = metrics[["qna_entropy", "cov_spectral_entropy", "effective_rank"]].dropna().copy()
    if subset.empty:
        raise ValueError("No overlapping benchmark observations are available.")

    standardized = subset.apply(zscore)
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(standardized.index, standardized["qna_entropy"], label="QNA entropy", lw=2.1, color="#B13A3A")
    ax.plot(
        standardized.index,
        standardized["cov_spectral_entropy"],
        label="Covariance spectral entropy",
        lw=1.8,
        color="#2F6690",
    )
    ax.plot(
        standardized.index,
        standardized["effective_rank"],
        label="Effective rank",
        lw=1.6,
        color="#4C956C",
    )
    ax.axvspan(pd.Timestamp(focus_start), pd.Timestamp(focus_end), color="#D8E2DC", alpha=0.45)
    ax.set_ylabel("Full-sample z-score")
    ax.set_title("Standardized benchmark dynamics, 2020--2025")
    ax.legend(ncol=3, frameon=False, loc="upper right")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300)


def plot_qews_vs_index(
    metrics: pd.DataFrame,
    signal_col: str,
    start: str,
    end: str,
    save_path: str | Path | None = None,
) -> None:
    plt = get_plt()
    if metrics.empty:
        raise ValueError("No metrics available for plotting.")

    subset = metrics.loc[start:end].copy()
    if subset.empty:
        raise ValueError("Selected plot window is empty.")

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(subset.index, subset[signal_col], color="tab:red", label=signal_col, lw=1.8)
    ax1.set_ylabel(signal_col, color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.plot(subset.index, subset["index_close"], color="tab:blue", label="NASDAQ index", lw=1.4, alpha=0.8)
    ax2.set_ylabel("Index level", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    ax1.set_title(f"{signal_col} vs NASDAQ index")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300)


def plot_focus_event_comparison(
    metrics: pd.DataFrame,
    start: str,
    end: str,
    event_date: str | None = None,
    event_dates: Sequence[str] | None = None,
    benchmark_cols: Sequence[str] | None = None,
    benchmark_labels: Sequence[str] | None = None,
    title: str = "Tariff-cluster comparison: QEWS and classical spectral benchmarks",
    save_path: str | Path | None = None,
) -> None:
    plt = get_plt()
    if metrics.empty:
        raise ValueError("No metrics available for plotting.")

    metrics = add_standardized_benchmarks(metrics, window=60)
    benchmark_cols = tuple(benchmark_cols or ("z_cov_spectral_entropy", "z_effective_rank"))
    benchmark_labels = tuple(
        benchmark_labels
        or ("Rolling-z covariance spectral entropy", "Rolling-z effective rank")
    )
    if len(benchmark_cols) != len(benchmark_labels):
        raise ValueError("benchmark_cols and benchmark_labels must have the same length.")

    plot_cols = ["qews_eri", *benchmark_cols]
    subset = metrics.loc[start:end, plot_cols].dropna()
    if subset.empty:
        raise ValueError("Selected focus window is empty.")

    fig, ax = plt.subplots(figsize=(13.5, 5.5))
    ax.plot(subset.index, subset["qews_eri"], label="QEWS (ERI)", lw=2.1, color="#B13A3A")
    palette = ("#2F6690", "#4C956C", "#6C5B7B")
    for idx, (column, label) in enumerate(zip(benchmark_cols, benchmark_labels)):
        ax.plot(
            subset.index,
            subset[column],
            label=label,
            lw=1.8 if idx == 0 else 1.6,
            color=palette[idx % len(palette)],
        )

    event_dates = list(event_dates or ([] if event_date is None else [event_date]))
    event_styles = [("black", "--"), ("#555555", ":"), ("#777777", "-.")]
    for idx, value in enumerate(event_dates):
        color, line_style = event_styles[idx % len(event_styles)]
        label = pd.Timestamp(value).strftime("%d %b %Y")
        ax.axvline(pd.Timestamp(value), color=color, lw=1.1, ls=line_style, label=label)
    ax.set_ylabel("Rolling z-score")
    ax.set_title(title)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300)


def plot_benchmark_scatter(
    metrics: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    save_path: str | Path | None = None,
) -> None:
    plt = get_plt()
    if metrics.empty:
        raise ValueError("No metrics available for plotting.")

    subset = metrics[[x_col, y_col]].dropna()
    if subset.empty:
        raise ValueError(f"No overlapping observations for {x_col} and {y_col}.")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(subset[x_col], subset[y_col], alpha=0.6, s=18, color="tab:blue")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300)
