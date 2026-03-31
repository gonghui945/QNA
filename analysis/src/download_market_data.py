from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from qna_revision_pipeline import RevisionConfig, download_yahoo_panel, load_tickers_from_excel, load_tickers_from_text


WIKI_NASDAQ100_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"


def normalize_tickers(tickers: list[str]) -> list[str]:
    cleaned = []
    for ticker in tickers:
        ticker = str(ticker).strip().upper()
        if not ticker:
            continue
        # Yahoo Finance expects "-" instead of "." for share-class tickers.
        ticker = ticker.replace(".", "-")
        cleaned.append(ticker)
    return sorted(dict.fromkeys(cleaned))


def fetch_nasdaq100_tickers_from_wikipedia() -> list[str]:
    request = Request(
        WIKI_NASDAQ100_URL,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        },
    )
    with urlopen(request) as response:
        html = response.read()

    tables = pd.read_html(html)
    candidates = []
    for table in tables:
        lowered = {str(col).strip().lower(): col for col in table.columns}
        if "ticker" in lowered:
            candidates = table[lowered["ticker"]].astype(str).tolist()
            break
        if "ticker symbol" in lowered:
            candidates = table[lowered["ticker symbol"]].astype(str).tolist()
            break
    if not candidates:
        raise ValueError("Could not locate a Nasdaq-100 ticker column on Wikipedia.")
    return normalize_tickers(candidates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download revised QFE market data.")
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--index-ticker", default="^NDX")
    parser.add_argument(
        "--ticker-source",
        choices=("wikipedia", "excel", "text"),
        default="wikipedia",
        help="How to source the Nasdaq-100 ticker universe.",
    )
    parser.add_argument(
        "--ticker-file",
        default="",
        help="Optional local Excel or text file when ticker-source is not wikipedia.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/raw/market_data",
        help="Directory where raw downloaded CSV files will be stored.",
    )
    parser.add_argument(
        "--reference-dir",
        default="data/reference",
        help="Directory where ticker reference files will be written.",
    )
    parser.add_argument("--force", action="store_true", help="Redownload even if local CSV files exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RevisionConfig(start_date=args.start_date, end_date=args.end_date)

    if args.ticker_source == "wikipedia":
        tickers = fetch_nasdaq100_tickers_from_wikipedia()
    elif args.ticker_source == "excel":
        if not args.ticker_file:
            raise ValueError("--ticker-file is required when --ticker-source excel is used.")
        tickers = normalize_tickers(load_tickers_from_excel(args.ticker_file))
    else:
        if not args.ticker_file:
            raise ValueError("--ticker-file is required when --ticker-source text is used.")
        tickers = normalize_tickers(load_tickers_from_text(args.ticker_file))

    reference_dir = Path(args.reference_dir)
    reference_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": tickers}).to_csv(reference_dir / "nasdaq100_tickers.csv", index=False)
    (reference_dir / "nasdaq100_tickers.txt").write_text("\n".join(tickers) + "\n", encoding="utf-8")

    print(f"Ticker source: {args.ticker_source}")
    print(f"Universe size: {len(tickers)}")
    print(f"Index ticker: {args.index_ticker}")
    print(f"Date range: {args.start_date} -> {args.end_date}")

    data_dict, index_df = download_yahoo_panel(
        tickers=tickers,
        index_ticker=args.index_ticker,
        config=config,
        out_dir=args.out_dir,
        force=args.force,
    )

    print(f"Downloaded stock panels: {len(data_dict)}")
    print(f"Index observations: {len(index_df)}")
    print(f"Raw data directory: {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()
