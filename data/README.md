# Market data

Place `market_data.csv` here (or pass `--data` to the CLI).

Required columns: `Date`, `NIFTY_Open`, `NIFTY_High`, `NIFTY_Low`, `NIFTY_Close`,
`NIFTY_Volume`, `INDIA_VIX_Close`, `SP500_Close`, `NASDAQ_Close`, `DOW_Close`,
`NIKKEI_Close`, `HANGSENG_Close`, `BRENT_Close`, `GOLD_Close`, `USDINR_Close`.

If you have the file in `attachments/market_data.csv` locally:

```bash
cp attachments/market_data.csv data/market_data.csv
# or: ln -s ../attachments/market_data.csv data/market_data.csv
```
