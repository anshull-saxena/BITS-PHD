"""Entry point: reproduce every table and figure in the DeepPIPE NIFTY 50 paper.

Replaces the Colab upload (Cell 1) and download (Cell 7) with plain file I/O.

Usage:
    python main.py --data path/to/market_data.csv
    python main.py --data market_data.csv --out-dir outputs --no-figures

`market_data.csv` must contain columns: Date, NIFTY_Open, NIFTY_High,
NIFTY_Low, NIFTY_Close, NIFTY_Volume, INDIA_VIX_Close, SP500_Close,
NASDAQ_Close, DOW_Close, NIKKEI_Close, HANGSENG_Close, BRENT_Close,
GOLD_Close, USDINR_Close (technical indicators are derived automatically).
"""
import argparse
import json
import os
import shutil
import zipfile

from config import DATA_PATH
from experiments import run_all
from report import print_tables
from figures import make_figures


def main():
    ap = argparse.ArgumentParser(description="Reproduce the DeepPIPE NIFTY 50 paper results.")
    ap.add_argument("--data", default=DATA_PATH,
                    help="Path to market_data.csv (default: %(default)s)")
    ap.add_argument("--out-dir", default=".",
                    help="Directory for results JSON, figures and the output ZIP.")
    ap.add_argument("--results", default="results_all.json",
                    help="Filename for the results JSON (inside --out-dir).")
    ap.add_argument("--no-figures", action="store_true", help="Skip figure generation.")
    ap.add_argument("--no-zip", action="store_true", help="Skip building the output ZIP.")
    ap.add_argument("--reuse-results", action="store_true",
                    help="Load an existing results JSON instead of re-running experiments.")
    args = ap.parse_args()

    if not os.path.isfile(args.data):
        raise SystemExit(f"Data file not found: {args.data}")

    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, args.results)
    figures_dir = os.path.join(args.out_dir, "figures")

    if args.reuse_results and os.path.isfile(results_path):
        print(f"Loading existing results from {results_path}")
        with open(results_path) as fh:
            RESULTS = json.load(fh)
    else:
        RESULTS = run_all(args.data, results_path=results_path)

    print_tables(RESULTS)

    if not args.no_figures:
        make_figures(RESULTS, out_dir=figures_dir)

    if not args.no_zip:
        zip_base = os.path.join(args.out_dir, "deeppipe_outputs")
        if os.path.isdir(figures_dir):
            shutil.make_archive(zip_base, "zip", args.out_dir, "figures")
        if os.path.isfile(results_path):
            with zipfile.ZipFile(zip_base + ".zip", "a") as z:
                z.write(results_path, arcname=os.path.basename(results_path))
        print(f"Wrote {results_path} and {zip_base}.zip")


if __name__ == "__main__":
    main()
