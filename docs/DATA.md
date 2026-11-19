# Market observation, calendar and synthetic inputs

| Input | Origin | Role |
| --- | --- | --- |
| `examples/yahoo_spxw/raw_snapshot.json` | Yahoo Finance via yfinance, capture `attempt_01` on 26 October 2026 | Raw option books before the local-variance filters |
| `examples/yahoo_spxw/session_closes.json` | Frozen schedule supplied with the market certificate | PM session closes for ACT/365F maturity |
| `examples/exact_initial/normalized.csv` | Author's 27 synthetic European call quotes | Nine-month and one-year gamma certificates and R3 |
| `examples/general_informative/normalized.csv` | Author's 14 synthetic European call quotes | Informative continuous-frontier example |

The downloader pins upstream commit `423b81938ac4661811119497b03ee3487acaacc5`. The upstream repository contains several captures on the same date. The local-variance result uses `data/raw/yahoo/2026-10-26/attempt_01.json`. Both SHA-256 and byte length were matched to the raw file named in the local certificate. The session table is versioned with the code and checked independently. The local file paths in the table for the synthetic chains are those of the numerical supplement. The main reproduction driver independently generates both CSV inputs under each run’s `synthetic-inputs/` directory. [Machine-readable record](../data/provenance.json) · [Upstream data catalogue](https://github.com/malikberrada/DupireLocalVolatilityForward/blob/423b81938ac4661811119497b03ee3487acaacc5/docs/DATA.md)

## Download the observed books

```console
python scripts/download_data.py
```

For the original raw file alone, this is the underlying HTTPS request. Modern Windows includes `curl.exe`. On Linux and macOS use `curl`.

```console
curl --fail --location --output attempt_01.json https://raw.githubusercontent.com/malikberrada/DupireLocalVolatilityForward/423b81938ac4661811119497b03ee3487acaacc5/data/raw/yahoo/2026-10-26/attempt_01.json
```

The acquisition ran from 19:31:29.106182 to 19:31:43.546751 UTC. It received 799 call rows and 1,042 put rows over six expiries. For 9 November the archive contains 102 call rows and 153 put rows. The audit finds 78 matched strikes and selects five pairs, at strikes 7,525, 7,605, 7,680, 7,755 and 7,830. Selection uses the raw spot only as a strike-selection reference. The selected slice's underlying timestamp is 19:16:27 UTC, with a reported 15-minute delay. Each contract's `lastTradeDate` refers to its latest trade. The paper's common-time interpretation of bid and ask remains an assumption. [Contract-level audit](../examples/yahoo_spxw/audit.json)

## Run the original yfinance collection protocol on a new date

The upstream collector selects six expiries spread across the eligible calendar window before examining prices. In a separate directory, the following commands run that collector with the recorded symbol, root and expiry window. They obtain a **new capture** during a regular market session. The input records the actual expiries chosen on 26 October. The legacy snapshot omits the installed yfinance version. The upstream replication lock specifies yfinance 1.7.0. [Collector source](https://github.com/malikberrada/DupireLocalVolatilityForward/blob/423b81938ac4661811119497b03ee3487acaacc5/collect_coherent_yahoo.py)

```console
git clone https://github.com/malikberrada/DupireLocalVolatilityForward.git upstream-collector
git -C upstream-collector checkout --detach 423b81938ac4661811119497b03ee3487acaacc5
cd upstream-collector
python -m pip install -r requirements-lock.txt
python -m pip install --no-build-isolation --no-deps .
python collect_coherent_yahoo.py --symbol "^SPX" --root SPXW --min-days 14 --max-days 180 --max-expiries 6 --attempts 1 --out runs/new-capture
```

Use a separate Python 3.12 environment for the upstream lock. From this local-variance repository, its own live collector can be run as follows. This version chooses the earliest eligible expiries. It stores each returned table and the session calendar in the new snapshot.

```console
python -m pip install -r requirements/live.txt
python -I scripts/yahoo_spxw.py collect --min-days 14 --max-days 180 --max-expiries 6 --out runs/live-spxw
```

Both programs call `yf.Ticker("^SPX").option_chain(expiry)` for each selected expiry. Passing an old date to this method selects a contract expiration; replaying the historical order book requires the frozen file above. [yfinance option-chain API](https://github.com/ranaroussi/yfinance/blob/main/yfinance/ticker.py)
