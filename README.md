# What do option bid–ask quotes identify about local variance?

This repository reproduces a bound on a gamma-weighted average of annual local variance using European S&P 500 index options. The market experiment starts from the Yahoo Finance capture of 26 October 2026 and retains five call–put pairs at the 9 November expiry. The verifier checks a common diffusion against all ten raw spreads and recomputes the outer interval with rational arithmetic. The synthetic experiments use their own frozen quotes.

## Download the data used in the paper

From the repository root, run the following command with Python 3.12. This step uses the Python standard library.

```console
python scripts/download_data.py
```

It downloads **attempt_01** from a fixed Git commit, verifies all **919,067 bytes** and the SHA-256 below, then places `raw_snapshot.json` and the versioned session-close table in `examples/yahoo_spxw/`. The latter fixes the expiry time used in the calculation. Use `--offline` to check the local input after its first download. An alternate `--out` copies both inputs together. Existing files with different bytes cause a verification error. [Raw JSON at the pinned commit](https://raw.githubusercontent.com/malikberrada/DupireLocalVolatilityForward/423b81938ac4661811119497b03ee3487acaacc5/data/raw/yahoo/2026-10-26/attempt_01.json) · [Input hashes and provenance](data/provenance.json)

```text
0c44b9b66a46b238d891ffbeb21fcaf65299f4987a9cf8b9f51993921e60db41
```

## Reproduce the results on CPU

Clone the public code and enter the repository root.

```console
git clone --single-branch --branch main https://github.com/malikberrada/ReallyIdentifiableLocalVarianceDupire.git
cd ReallyIdentifiableLocalVarianceDupire
```

Create an isolated environment. In PowerShell on Windows run:

```powershell
py -3.12 -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install -r requirements/review.txt
& $py -m pip install --no-build-isolation --no-deps .
& $py scripts/download_data.py
& $py scripts/reproduce.py
& $py -m pytest -q
```

On Linux or macOS run:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements/review.txt
python -m pip install --no-build-isolation --no-deps .
python scripts/download_data.py
python scripts/reproduce.py
python -m pytest -q
```

The driver writes fresh witnesses, audits, certificates and `summary.json` under `runs/review-<UTC timestamp>/`. It checks the archived proof, reconstructs both market estimates from the downloaded inputs, verifies each new witness, proves the all-pairs contradiction and compares the two synthetic gamma certificates byte for byte. A failed check returns a nonzero exit code. To verify just the original archived market proof, run `python -I scripts/yahoo_spxw.py verify --run examples/yahoo_spxw`.

| Experiment | Estimate | Guaranteed absolute error | Outer interval (outward rounded) |
| --- | ---: | ---: | --- |
| SPXW, fixed reference variance 0.04 | 0.018988 | 0.009501 | [0.009487, 0.028488] |
| SPXW, reference selected by the archived rule | 0.019929 | 0.000662 | [0.019267, 0.020590] |
| 27 synthetic calls, nine-month gamma | 0.039810 | 0.009797 | See exact endpoints in the certificate |
| 27 synthetic calls, one-year gamma | 0.039420 | 0.008310 | See exact endpoints in the certificate |

The two market rows use different gamma weights for call `SPXW261109C07680000`. Both use the normalized price corridor [0.65, 1.50] and annual variance bounds [0.0025, 0.25]. The guarantee is conditional on treating the selected bid–ask quotes as simultaneous. Yahoo reports the most recent trade time and gives no timestamp for each side of the book. The five pairs are chosen from 78 valid pairs by a fixed spot-relative strike rule. Imposing parity on all 78 yields the incompatible requirements `D ≥ 53/50` and `D ≤ 49/50`. The positive result therefore pertains to the ten selected contracts. Sharp endpoints for pointwise local variance are a separate question. [Market proof and assumptions](docs/YAHOO_SPXW.md) · [Synthetic gamma method](docs/GARANTIE_POSITIVE.md)

## Collect a new observation with yfinance

During a regular US cash session, run these commands using the same environment. Use a new directory for each capture.

```console
python -m pip install -r requirements/live.txt
python -I scripts/yahoo_spxw.py collect --min-days 14 --max-days 180 --max-expiries 6 --out runs/live-spxw
```

The project collector queries `yf.Ticker("^SPX").options` and then `ticker.option_chain(expiry)`. It records the first eligible expiries, all returned calls and puts, request times, errors and session closes. A call made today records a new observation. In the yfinance API the date selects the option expiry. The original October collector spread its six requested expiries across the available dates, before inspecting prices. The archived JSON records its actual choices. [Original acquisition procedure and commands](docs/DATA.md) · [yfinance implementation](https://github.com/ranaroussi/yfinance/blob/main/yfinance/ticker.py)

## Synthetic frontier and R3 campaign

The driver regenerates the 27-call and 14-call synthetic chains from their analytical parameters, then checks both CSV hashes. The supplementary archive accompanying the research submission contains the original numerical surfaces, stored policies, historical proofs and reference outputs. To install it and run the earlier, larger R3 experiment, pass its local path to the installer. The public SPXW experiment and the two synthetic gamma estimates run independently of the supplement.

```console
python scripts/install_supplement.py DupireIdentify-supplement-20261118.zip
python -I scripts/rejouer_r3.py tout --workers 4 --out runs/r3
python -I scripts/rejouer_r3.py comparer --out runs/r3
python -m pytest -q
```

At absolute tolerance 0.01 the reference campaign certifies the minimum throughout all four regions. Joint coverage of both endpoints is 90.0450%, 95.9124%, 87.6048% and 92.3993%, respectively. A successful replay preserves the published scientific status `request_completed = false`. The installed supplement also enables two endpoint tests that use the full numerical surface. The campaign describes which decisions are regenerated and which are replayed from the archive. [R3 methods, counts and reference hashes](rejeu_r3/REJEU.md) · [Continuous frontier examples](docs/GENERAL_FRONTIER.md)

The MIT license covers the software. The frozen market file records third-party Yahoo Finance observations. [Software license](LICENSE) · [Market data provenance](docs/DATA.md)
