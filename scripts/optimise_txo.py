"""Run the TXO historical optimization and counterexample verifier from this checkout."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dupire_identify.taifex_optimization import main

if __name__=="__main__":
    raise SystemExit(main())
