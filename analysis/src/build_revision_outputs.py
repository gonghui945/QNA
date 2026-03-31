from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from qna_revision_pipeline import (
    RevisionConfig,
    add_standardized_benchmarks,
    compute_revision_metrics,
    evaluate_event_windows,
    load_event_catalog,
    load_local_market_panel,
    plot_focus_event_comparison,
    plot_standardized_benchmark_series,
    run_feature_robustness,
    run_window_robustness,
    summarize_benchmark_relationships,
    summarize_feature_robustness,
    summarize_metrics,
    summarize_window_robustness,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build revised QFE analysis outputs.")
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--index-ticker", default="^NDX")
    parser.add_argument("--data-dir", default="data/raw/market_data")
    parser.add_argument("--reference-dir", default="data/reference")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--figures-dir", default="paper/figures")
    parser.add_argument("--tables-dir", default="paper/tables")
    parser.add_argument("--focus-start", default="2024-10-01")
    parser.add_argument("--focus-end", default="2025-06-30")
    parser.add_argument("--reuse-processed", action="store_true")
    return parser.parse_args()


def format_corr(corr: float, ci_low: float, ci_high: float) -> str:
    if pd.isna(corr):
        return "--"
    return f"{corr:.3f} [{ci_low:.3f}, {ci_high:.3f}]"


def latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def build_benchmark_table_tex(benchmark_summary: pd.DataFrame) -> str:
    panel_a = benchmark_summary[benchmark_summary["panel"] == "Level relationships"]
    panel_b = benchmark_summary[benchmark_summary["panel"] == "Deviation relationships"]

    def panel_rows(frame: pd.DataFrame) -> list[str]:
        rows: list[str] = []
        for qna_metric, group in frame.groupby("qna_metric", sort=False):
            cells = [latex_escape(qna_metric)]
            for _, row in group.iterrows():
                cells.append(format_corr(row["corr"], row["ci_low"], row["ci_high"]))
            rows.append("        " + " & ".join(cells) + r" \\")
        return rows

    panel_a = panel_a.copy()
    panel_a["benchmark_metric"] = pd.Categorical(
        panel_a["benchmark_metric"],
        [
            "Covariance spectral entropy",
            "Effective rank",
            "Participation ratio",
            "Mean absolute correlation",
            "Realized volatility",
        ],
        ordered=True,
    )
    panel_a = panel_a.sort_values(["qna_metric", "benchmark_metric"])

    panel_b = panel_b.copy()
    panel_b["benchmark_metric"] = pd.Categorical(
        panel_b["benchmark_metric"],
        [
            "Rolling-z covariance spectral entropy",
            "Rolling-z effective rank",
            "Rolling-z mean absolute correlation",
            "Rolling-z realized volatility",
        ],
        ordered=True,
    )
    panel_b = panel_b.sort_values(["qna_metric", "benchmark_metric"])

    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\small",
        r"\caption{Benchmark correlations between QNA diagnostics and classical dependence summaries. Entries report full-sample correlations with bootstrap 95\% confidence intervals.}",
        r"\label{tab:benchmark_summary}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"\textbf{Panel A: Level diagnostics} & \textbf{Cov. spectral entropy} & \textbf{Effective rank} & \textbf{Participation ratio} & \textbf{Mean abs. corr.} & \textbf{Realized vol.} \\",
        r"\hline",
    ]
    lines.extend(panel_rows(panel_a))
    lines.extend(
        [
            r"\hline",
            r"\textbf{Panel B: Standardized deviations} & \textbf{Rolling-z cov. entropy} & \textbf{Rolling-z eff. rank} & \textbf{Rolling-z mean abs. corr.} & \textbf{Rolling-z realized vol.} & \textbf{} \\",
            r"\hline",
        ]
    )
    qews_row = {
        "Rolling-z covariance spectral entropy": "",
        "Rolling-z effective rank": "",
        "Rolling-z mean absolute correlation": "",
        "Rolling-z realized volatility": "",
    }
    for _, row in panel_b.iterrows():
        qews_row[row["benchmark_metric"]] = format_corr(row["corr"], row["ci_low"], row["ci_high"])
    lines.append(
        "        "
        + " & ".join(
            [
                "QEWS (ERI)",
                qews_row["Rolling-z covariance spectral entropy"],
                qews_row["Rolling-z effective rank"],
                qews_row["Rolling-z mean absolute correlation"],
                qews_row["Rolling-z realized volatility"],
                "",
            ]
        )
        + r" \\"
    )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def build_sensitivity_table_tex(
    feature_summary: pd.DataFrame,
    window_summary: pd.DataFrame,
) -> str:
    feature_map = {
        "returns_only": "Returns only",
        "returns_vol": "Returns + volatility",
        "baseline_full": "Baseline full",
    }
    feature_summary = feature_summary.copy()
    feature_summary["display"] = feature_summary["specification"].map(feature_map).fillna(feature_summary["specification"])
    feature_summary["order"] = feature_summary["specification"].map(
        {"returns_only": 1, "returns_vol": 2, "baseline_full": 3}
    )
    feature_summary = feature_summary.sort_values("order")
    window_summary = window_summary.sort_values("rolling_window")

    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\small",
        r"\caption{Sensitivity of QNA diagnostics to feature construction and rolling-window length. The baseline window is 60 trading days; Panel B reports the 40/60/90-day comparison used to preserve stable multi-feature coverage at the short end.}",
        r"\label{tab:sensitivity_summary}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\hline",
        r"\textbf{Panel A: Feature specification} & \textbf{Mean dim.} & $\mathbf{E[S_{\mathrm{QNA}}]}$ & $\mathbf{SD[S_{\mathrm{QNA}}]}$ & $\mathbf{Corr(S_{\mathrm{QNA}}, S_{\mathrm{cov}})}$ & $\mathbf{Corr(S_{\mathrm{QNA}}, r_{\mathrm{eff}})}$ & $\mathbf{Corr(QEWS, zS_{\mathrm{cov}})}$ \\",
        r"\hline",
    ]
    for _, row in feature_summary.iterrows():
        lines.append(
            "        "
            + " & ".join(
                [
                    latex_escape(row["display"]),
                    f"{row['qna_dimension_mean']:.1f}",
                    f"{row['qna_entropy_mean']:.3f}",
                    f"{row['qna_entropy_std']:.3f}",
                    f"{row['corr_entropy_cov']:.3f}",
                    f"{row['corr_entropy_effrank']:.3f}",
                    f"{row['corr_qews_covz']:.3f}",
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\hline",
            r"\textbf{Panel B: Rolling window} & \textbf{Mean dim.} & $\mathbf{E[S_{\mathrm{QNA}}]}$ & $\mathbf{SD[S_{\mathrm{QNA}}]}$ & $\mathbf{Corr(S_{\mathrm{QNA}}, S_{\mathrm{cov}})}$ & $\mathbf{Corr(S_{\mathrm{QNA}}, r_{\mathrm{eff}})}$ & $\mathbf{Corr(QEWS, zS_{\mathrm{cov}})}$ \\",
            r"\hline",
        ]
    )
    for _, row in window_summary.iterrows():
        lines.append(
            "        "
            + " & ".join(
                [
                    f"{int(row['rolling_window'])}-day",
                    f"{row['qna_dimension_mean']:.1f}",
                    f"{row['qna_entropy_mean']:.3f}",
                    f"{row['qna_entropy_std']:.3f}",
                    f"{row['corr_entropy_cov']:.3f}",
                    f"{row['corr_entropy_effrank']:.3f}",
                    f"{row['corr_qews_covz']:.3f}",
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def build_event_table_tex(event_results: pd.DataFrame) -> str:
    event_results = event_results.copy()
    event_results = event_results.dropna(subset=["observed_diff"]).sort_values(["event_type", "event_date"])
    type_order = {"macro": 1, "policy": 2, "placebo": 3, "market": 4}
    event_results["order"] = event_results["event_type"].map(type_order).fillna(99)
    event_results = event_results.sort_values(["order", "event_date"])

    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\small",
        r"\caption{Event-window evidence for QEWS (ERI) across selected macro/policy and placebo windows. Pre- and post-event means use a symmetric 20-trading-day window. The reported difference is pre minus post, with bootstrap confidence intervals and a permutation $p$-value.}",
        r"\label{tab:event_summary}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\hline",
        r"\textbf{Event} & \textbf{Date} & \textbf{Type} & \textbf{Pre QEWS} & \textbf{Post QEWS} & \textbf{Pre - Post} & \textbf{95\% CI / Perm. $p$} \\",
        r"\hline",
    ]
    for _, row in event_results.iterrows():
        ci_text = f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]; p={row['perm_pvalue']:.3f}"
        event_date = pd.to_datetime(row["event_date"]).strftime("%Y-%m-%d")
        lines.append(
            "        "
            + " & ".join(
                [
                    latex_escape(str(row["event_label"])),
                    event_date,
                    latex_escape(str(row["event_type"])),
                    f"{row['pre_mean']:.3f}",
                    f"{row['post_mean']:.3f}",
                    f"{row['observed_diff']:.3f}",
                    ci_text,
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = RevisionConfig(start_date=args.start_date, end_date=args.end_date)

    reference_dir = Path(args.reference_dir)
    processed_dir = Path(args.processed_dir)
    figures_dir = Path(args.figures_dir)
    tables_dir = Path(args.tables_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    tickers = pd.read_csv(reference_dir / "nasdaq100_tickers.csv")["ticker"].dropna().astype(str).tolist()
    events = load_event_catalog(reference_dir / "revision_event_catalog.csv")

    feature_sets = {
        "returns_only": ("ret",),
        "returns_vol": ("ret", "vol20"),
        "baseline_full": ("ret", "vol20", "log_volume", "dlog_volume"),
    }

    if args.reuse_processed:
        metrics = pd.read_csv(processed_dir / "baseline_metrics.csv", parse_dates=["date"]).set_index("date")
        feature_robustness = pd.read_csv(processed_dir / "feature_robustness.csv", parse_dates=["date"])
        window_robustness = pd.read_csv(processed_dir / "window_robustness.csv", parse_dates=["date"])
        data_dict = {ticker: None for ticker in tickers}
    else:
        data_dict, index_df = load_local_market_panel(
            data_dir=args.data_dir,
            tickers=tickers,
            config=config,
            index_ticker=args.index_ticker,
        )

        metrics = compute_revision_metrics(data_dict=data_dict, index_df=index_df, config=config)
        metrics = add_standardized_benchmarks(metrics, window=config.qews_window)
        metrics.to_csv(processed_dir / "baseline_metrics.csv")

        feature_robustness = run_feature_robustness(data_dict, index_df, config, feature_sets)
        feature_robustness.to_csv(processed_dir / "feature_robustness.csv")

        window_robustness = run_window_robustness(data_dict, index_df, config, windows=config.robustness_windows)
        window_robustness.to_csv(processed_dir / "window_robustness.csv")

    summary = summarize_metrics(metrics)
    summary.to_csv(tables_dir / "baseline_summary.csv", header=["value"])
    benchmark_summary = summarize_benchmark_relationships(metrics, bootstrap_iterations=1000)
    benchmark_summary.to_csv(tables_dir / "benchmark_relationships.csv", index=False)

    feature_summary = summarize_feature_robustness(feature_robustness.reset_index() if "date" not in feature_robustness.columns else feature_robustness)
    feature_summary.to_csv(tables_dir / "feature_robustness_summary.csv", index=False)

    window_summary = summarize_window_robustness(window_robustness.reset_index() if "date" not in window_robustness.columns else window_robustness)
    window_summary.to_csv(tables_dir / "window_robustness_summary.csv", index=False)

    event_results = evaluate_event_windows(
        metrics,
        events,
        signal_col="qews_eri",
        pre_days=20,
        post_days=20,
        bootstrap_iterations=1000,
        absolute=False,
        permutation_iterations=3000,
    )
    event_results.to_csv(tables_dir / "event_study_qews_eri.csv", index=False)

    plot_standardized_benchmark_series(
        metrics,
        save_path=figures_dir / "benchmark_dynamics_2020_2025.png",
        focus_start="2024-01-01",
        focus_end="2025-12-31",
    )
    plot_focus_event_comparison(
        metrics,
        start="2024-12-01",
        end="2025-05-15",
        event_dates=["2025-02-18", "2025-04-02"],
        benchmark_cols=("z_cov_spectral_entropy", "z_effective_rank"),
        benchmark_labels=("Rolling-z covariance spectral entropy", "Rolling-z effective rank"),
        title="Two-stage tariff episode: QEWS versus classical spectral benchmarks",
        save_path=figures_dir / "tariff_cluster_benchmarks.png",
    )

    (tables_dir / "benchmark_summary.tex").write_text(
        build_benchmark_table_tex(benchmark_summary),
        encoding="utf-8",
    )
    (tables_dir / "sensitivity_summary.tex").write_text(
        build_sensitivity_table_tex(feature_summary, window_summary),
        encoding="utf-8",
    )
    (tables_dir / "event_summary.tex").write_text(
        build_event_table_tex(event_results),
        encoding="utf-8",
    )

    run_metadata = {
        "date_range": [args.start_date, args.end_date],
        "index_ticker": args.index_ticker,
        "num_stock_panels": len(data_dict),
        "num_metric_rows": int(len(metrics)),
        "feature_specs": list(feature_sets.keys()),
        "robustness_windows": list(config.robustness_windows),
        "reuse_processed": bool(args.reuse_processed),
    }
    (processed_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    print("Saved synchronized metrics, manuscript tables, and manuscript figures.")
    print(f"Processed dir: {processed_dir.resolve()}")
    print(f"Figures dir: {figures_dir.resolve()}")
    print(f"Tables dir: {tables_dir.resolve()}")


if __name__ == "__main__":
    main()
