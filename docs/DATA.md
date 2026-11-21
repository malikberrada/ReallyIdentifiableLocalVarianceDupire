# TXO market data and synthetic controls

The study uses regular-session bid and ask quotes for European TAIEX options from the Taiwan Futures Exchange (TAIFEX). Daily TAIEX closes from the Taiwan Stock Exchange (TWSE) determine the strike anchors. The checkout includes 104,918 regular-session option rows and 33 index closes from October 1 through November 19, 2026. All quote prices are in index points. [Study data and manifest](../data/taifex_txo/) · [Official TAIFEX history](https://www.taifex.com.tw/enl/eng3/optDailyMarketView)

| File | Contents |
| --- | --- |
| `data/taifex_txo/YYYY-MM-DD.csv` | Every regular-session TXO row returned for that date. Columns give date, product, series, contract date, strike, call/put, best bid, best ask, session and halt indicator. Missing, crossed and unpaired observations remain visible. |
| `data/taifex_txo/YYYYMM.json` | Trading dates and closing TAIEX levels for the sessions in the study. Dates use the calendar and field names in the TWSE report. |
| `data/taifex_txo/manifest.json` | Official request, response SHA-256, stored file size, row count and stored SHA-256 for each file. The stored option records are sorted and the stored index records are ordered by date. |
| `examples/txo_optimization/panels/` | One reference per eligible session–expiry combination, including settings, panel and result hashes, and any verified witness and hedge positions. |
| `examples/txo_optimization/panels.csv` | Results for all 126 combinations, including cases where the verifier could establish only one bound or none. |

## October 7, 2026

The `202610F3` series expires on October 16. Its 318 regular-session records yield 95 eligible pairs. The five strikes nearest 98%, 99%, 100%, 101% and 102% of the 49,806.37 TAIEX close are 48,800, 49,300, 49,800, 50,400 and 50,800. The paired quote table flags those five strikes. [95 paired quotes](../examples/txo_optimization/20261007_202610F3_pairs.csv) · [TAIFEX daily CSV](https://www.taifex.com.tw/enl/eng3/optDataDown?down_type=1&commodity_id=TXO&queryStartDate=2026%2F10%2F07&queryEndDate=2026%2F10%2F07) · [TWSE monthly report](https://www.twse.com.tw/indicesReport/MI_5MINS_HIST?date=20261007&response=json)

The canonical panel digest includes the close and all 318 series records before quote filtering. It is `4d32fdb66a3265e03c4ee9a7a355a9ba95a052afada03306de0c820356281aa4`. The canonical result digest is `d702aca3eb93b7eb846759ce41725c5b45d08a35ee67e1415d69bef43c3ef9da`. Canonical hashes refer to the verifier's serialized objects. The manifest separately identifies the bytes of each input file. [Case provenance](../data/provenance.json) · [Full witness and hedge certificate](../examples/txo_optimization/20261007_202610F3_certificate.json)

A discount factor of 1 and a forward of 49,777 satisfy the parity strips for all 95 pairs. The common diffusion witness is checked against the ten contracts at the five selected strikes. The two certificates bound gamma-weighted annual local variance for the 49,800 call in a model of the TXO settlement value. TAIFEX bases this value on the average index level during the final 30 minutes before the close on the settlement day. The bounds are conditional on treating the daily spreads as simultaneous, with deterministic carry and annual variance in [0.0025, 0.25]. [TXO contract specifications](https://www.taifex.com.tw/enl/eng2/tXO)

## Reacquisition and offline verification

`scripts/download_data.py` requests official historical reports, keeps the fields listed above and verifies the resulting files against the published hashes. The default prepares `options.csv` and `index.json` for the main case. `--history` prepares the full dated cache. `--offline` applies the same hash checks to the files in the checkout. `scripts/taifex_txo.py replay` can also save the original responses and verify the published panel even when unrelated report formatting changes. [Reproduction commands](../README.md#download-fresh-copies-from-both-exchanges)

The retrospective scan enumerates series expiring in 7–60 calendar days on the 33 trading sessions. It ranks cases with two certificates by the larger of the two displayed error bounds. Ties are resolved by observation date, expiry and series. Both bounds were obtained in 89 of 126 cases. Nineteen met the 0.01 target under both gamma references, across seven sessions. [Protocol](../examples/txo_optimization/protocol.json) · [All outcomes](../examples/txo_optimization/panels.csv)

## Attribution

Securities and Futures Bureau, Financial Supervisory Commission, Executive Yuan, R.O.C. (2026). Taiwan Futures Exchange, *Options daily trading information*, dataset 11320. The stored CSVs contain the regular-session study fields from the dated TXO reports and the contract date supplied by TAIFEX. [Government data catalog](https://data.gov.tw/en/datasets/11320) · [Historical report fields](https://www.taifex.com.tw/enl/eng3/optDailyMarketView)

Securities and Futures Bureau, Financial Supervisory Commission, Executive Yuan, R.O.C. (2026). Taiwan Stock Exchange, *Weighted Stock Price Index Historical Data*, dataset 11755. The stored index files contain the trading date and closing level for each study session. [Government data catalog](https://data.gov.tw/en/datasets/11755)

The cataloged observations are available for reuse under the terms of the Open Government Data License, version 1.0. The MIT license covers the project software. [Data license](https://data.gov.tw/en/license) · [Software license](../LICENSE)

## Synthetic inputs

The driver regenerates the 27-call and 14-call synthetic panels from their analytical definitions, then verifies the two CSV digests and the two published gamma certificates. The R3 campaign uses a separately pinned numerical supplement. [Synthetic inputs](../scripts/synthetic_inputs.py) · [Input and certificate digests](../data/provenance.json) · [R3 supplement record](../data/supplement.json)
