import json
import sys
import unittest
from datetime import date
from fractions import Fraction as F
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/"src"))
from dupire_identify.taifex_optimization import case_name, result_row, trading_dates


class ScanTests(unittest.TestCase):
    def test_objective_requires_both_certificates(self):
        p=root/"examples/txo_optimization/panels/20261007_202610F3.json"
        summary=json.loads(p.read_bytes())["summary"]
        row=result_row(summary)
        self.assertEqual(row["objective"],"0.003334")
        self.assertTrue(row["both"])
        del summary["gamma"]["auto"]
        self.assertFalse(result_row(summary)["both"])
        self.assertEqual(result_row(summary)["objective"],"")

    def test_month_and_days_are_validated(self):
        sample={"stat":"OK","fields":["日期"],"data":[["115/11/19"],["115/11/18"]]}
        month=date(2026,11,1)
        self.assertEqual(trading_dates(json.dumps(sample).encode(),month),
                         [date(2026,11,18),date(2026,11,19)])
        sample["data"].append(["115/11/18"])
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            trading_dates(json.dumps(sample).encode(),month)
        with self.assertRaises(ValueError):
            case_name("2026-11-19","../202612")


try:
    import flint
except ImportError:
    flint=None


@unittest.skipUnless(flint is not None,"python-flint is required for the attainable-model verifier")
class AttainableModelTests(unittest.TestCase):
    def test_constant_diffusion_matches_black_and_constant_weighted_variance(self):
        from dupire_identify.taifex_separation import ball, black, endpoints, inspect_model, integrals
        lo,hi,t,r=map(F,[".0025",".25",".07",".04"])
        spec=dict(forward="100",discount="1",edges=["0"],
                  regimes=[dict(time_fraction="1/2",a0=str(hi),mass="1",high=["1","1"])])
        model=inspect_model(spec,lo,hi)
        with flint.ctx.workprec(128):
            for k in [F("0.96"),F("1.04")]:
                d,n,g=integrals(k,t,r,model,lo,hi,128)
                a,b=endpoints(n/d)
                self.assertLessEqual(a,hi)
                self.assertGreaterEqual(b,hi)
                reconstructed=black(ball(k),ball(r*t),ball(2).sqrt())+g/2
                wanted=black(ball(k),ball(hi*t),ball(2).sqrt())
                self.assertTrue(reconstructed.contains(wanted))

    def test_model_with_invalid_probabilities_is_rejected(self):
        from dupire_identify.taifex_separation import inspect_model
        spec=dict(forward="100",discount="1",edges=["0"],
                  regimes=[dict(time_fraction="1/2",a0="1/4",mass="1",high=["-1/5","1"])])
        with self.assertRaisesRegex(ValueError,"conditional probability"):
            inspect_model(spec,F("0.0025"),F("0.25"))
        spec["regimes"][0].update(mass="0.5",high=["0","1"])
        with self.assertRaisesRegex(ValueError,"sum to one"):
            inspect_model(spec,F("0.0025"),F("0.25"))

    def test_corridor_tail_bound_conditions(self):
        from dupire_identify.taifex_separation import tail_ceiling
        with flint.ctx.workprec(128):
            self.assertGreater(tail_ceiling(F(1),F("0.07"),F("0.21"),[F("0.65"),F("1.5")]),0)
            with self.assertRaisesRegex(ValueError,"does not apply"):
                tail_ceiling(F("1.8"),F("0.07"),F("0.21"),[F("0.65"),F("1.5")])


if __name__=="__main__":
    unittest.main()
