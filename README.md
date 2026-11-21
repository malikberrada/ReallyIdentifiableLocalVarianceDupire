# Certified bounds on averages of Dupire local variance from option quotes

This repository reproduces bounds on gamma-weighted annual local variance from European TAIEX options (TXO). It includes regular-session option quotes from the Taiwan Futures Exchange (TAIFEX), closing index levels from the Taiwan Stock Exchange (TWSE), and the published case files for 33 trading sessions from October 1 through November 19, 2026. [Market data](data/taifex_txo) · [126 session–expiry results](examples/txo_optimization/panels.csv)

## Results for the October 7 session

The TXO call with strike 49,800 belongs to series `202610F3` and expires on October 16, 2026. All **95 eligible call–put pairs** satisfy parity with a common discount factor and forward. A common diffusion witness prices the **10 contracts at five selected strikes** within their bid–ask spreads. The bounds assume the daily spreads can be imposed simultaneously and concern a model of the TXO settlement value. [Quotes and witness selection](examples/txo_optimization/20261007_202610F3_pairs.csv) · [Full certificate](examples/txo_optimization/20261007_202610F3_certificate.json)

| Gamma reference | Estimate | Guaranteed absolute error | Outward-rounded interval |
| --- | ---: | ---: | --- |
| 0.04 | 0.039411 | 0.003327 | [0.036084, 0.042737] |
| Automatic, 16619791/419430400 | 0.039760 | 0.003334 | [0.036427, 0.043094] |

The fixed reference gives the slightly smaller error bound on this panel. The October 7 case minimizes the larger of the two bounds in a retrospective search over 126 session–expiry combinations. Both errors are at most 0.01 in 19 combinations across seven sessions. [Selection protocol](examples/txo_optimization/protocol.json) · [Search results](examples/txo_optimization/summary.json)

## Reproduce the results on CPU

From the repository root, install the reference environment with Python 3.12. On Windows, run in PowerShell.

```powershell
py -3.12 -m venv .venv
$py = '.\.venv\Scripts\python.exe'
& $py -m pip install -r requirements/review.txt
& $py -m pip install --no-build-isolation --no-deps .
& $py -I scripts/reproduce.py
& $py -m pytest -q
```

On Linux or macOS, use the following commands.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements/review.txt
python -m pip install --no-build-isolation --no-deps .
python -I scripts/reproduce.py
python -m pytest -q
```

The driver uses the bundled observations, rebuilds and checks the market witness and both hedges, and regenerates the two synthetic control rows in Table 1. It prints the results and writes `summary.json`, the audit and the full certificate under `runs/review-…/`. The synthetic nine-month and one-year calls have respective error bounds of 0.009797 and 0.008310. [Expected Table 1](examples/txo_optimization/table1.json) · [Synthetic gamma method](docs/GARANTIE_POSITIVE.md)

## Download fresh copies from both exchanges

With the environment active, the following commands request the original reports, check the study inputs against their SHA-256 digests, and use them to reproduce Table 1. In PowerShell replace `python` with `& $py`. Supply a new output directory to `reproduce.py` for each run.

```sh
python -I scripts/download_data.py
python -I scripts/reproduce.py --inputs data/downloaded/taifex_txo --out runs/txo-redownload
```

For the complete original responses and a self-contained market audit, use `replay` and `verify`. Both commands check the same reference case. The first stores the TAIFEX CSV and TWSE JSON as received, and the second repeats the calculation offline.

```sh
python -I scripts/taifex_txo.py replay --case examples/txo_optimization/panels/20261007_202610F3.json --out runs/txo-official
python -I scripts/taifex_txo.py verify --run runs/txo-official
```

## Replay the 126 session–expiry combinations

```sh
python -I scripts/optimise_txo.py scan --start 2026-10-01 --end 2026-11-19 --jobs 4 --published examples/txo_optimization/panels --cache data/taifex_txo --offline --out runs/txo-scan
```

The scan enumerates the series from the observations, checks the individual references and regenerates `panels.csv` and `summary.json`. To reacquire every input, run `python -I scripts/download_data.py --history --out runs/fresh-history` and pass `runs/fresh-history` as the scan cache. [Input fields, sources and attribution](docs/DATA.md)

## Synthetic frontier and R3 campaign

The driver regenerates the 27-call and 14-call synthetic chains from their analytical parameters. The separate R3 supplement contains the numerical surfaces, policies and proofs for the four-region campaign. Install the supplement from its local path to replay that experiment.

```sh
python scripts/install_supplement.py DupireIdentify-supplement-20261118.zip
python -I scripts/rejouer_r3.py tout --workers 4 --out runs/r3
python -I scripts/rejouer_r3.py comparer --out runs/r3
```

At tolerance 0.01, the reference R3 campaign certifies the minimum throughout all four regions. Joint coverage of both endpoints is 90.0450%, 95.9124%, 87.6048% and 92.3993%, respectively. The supplementary experiment records `request_completed = false`. [R3 methods and reference hashes](rejeu_r3/REJEU.md) · [Continuous frontier examples](docs/GENERAL_FRONTIER.md)

The MIT license covers the software. The market observations are attributed to TAIFEX and TWSE under the Open Government Data License, version 1.0. [Software license](LICENSE) · [Data attribution](docs/DATA.md#attribution)
