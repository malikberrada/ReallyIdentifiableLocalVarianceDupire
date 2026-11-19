"""Use the checked-out sources, including when launched with python -I."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dupire_identify.yahoo_spxw import main

if __name__ == '__main__':
    raise SystemExit(main())
