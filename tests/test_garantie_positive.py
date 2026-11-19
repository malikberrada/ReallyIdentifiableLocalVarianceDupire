"""Reproducible checks for the positive identification certificates."""
from dataclasses import replace
from fractions import Fraction as F
import importlib.util
import json
from math import ceil, exp, log as ln, pi as circle, sqrt as root
from pathlib import Path
import sys
import tempfile
import unittest
from scipy.integrate import quad
from scipy.special import ndtr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dupire_identify.families import Family
from dupire_identify.interval import I, black_call, exp_point, log, pi, sqrt
from dupire_identify.io import read_quotes
from dupire_identify.positive_quotes import bounds, check_witness, gamma_bounds, gamma_lower

spec = importlib.util.spec_from_file_location("garantie_positive", ROOT / "scripts" / "garantie_positive.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)
LOW, HIGH = F(1, 400), F(1, 4)


class GarantiePositive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.family = Family.make(json.loads((ROOT / "examples/general_informative_domain.json").read_bytes()))
        sys.path.insert(0, str(ROOT / "scripts"))
        from synthetic_inputs import informative_quotes, synthetic_quotes
        cls.quotes = informative_quotes()
        with tempfile.TemporaryDirectory() as tmp:
            cls.r3 = synthetic_quotes(Path(tmp) / "quotes.csv")
        cls.beta = bounds(cls.quotes, cls.family, LOW, HIGH, None, None)

    def test_midpoint_rounding_uses_both_endpoints(self):
        est = cli.estimate(F(0), F(1, 4), F(1, 8), 2)
        self.assertTrue(est["precision_cible_garantie"])
        self.assertEqual(est["decimal"], {"estimation": "0.12", "erreur_absolue_au_plus": "0.13",
                                           "precision_cible_garantie": False})
        self.assertFalse(cli.estimate(LOW, HIGH, F(1, 100), 10)["precision_cible_garantie"])

    def test_parameter_tree_inherits_parent_bounds(self):
        root = dict(status="split", lower="1/20", upper="3/20", children=[
            dict(status="open", lower="0", upper="2/20", path="r0", family={"i": 0}),
            dict(status="open", lower="2/20", upper="1/4", path="r1", family={"i": 1})])
        rows = list(cli.verified_leaves(root, LOW, HIGH))
        result = cli.report_for({}, rows, {"variance_a_priori": [str(LOW), str(HIGH)]}, F(1, 40), 4)
        self.assertFalse(result["garantie_uniforme"]["precision_cible_garantie"])
        self.assertEqual(result["garantie_uniforme"]["intervalle_exterieur"], ["1/20", "3/20"])
        self.assertTrue(all(x["precision_cible_garantie"] for x in result["garanties_par_feuille"]))

    def test_beta_portfolio_dominates_on_finer_space_time_partition(self):
        lo, hi, source = self.beta
        self.assertEqual(hi, HIGH)
        self.assertLessEqual((hi - lo) / 2, F(1, 10000))
        d = F(source["denominateur_au_moins"])
        band = source["bande_de_probabilite"]
        t0, t1 = map(F, band["temps"])
        s0, s1 = map(F, band["prix"])
        prob = 1 - (exp_point(HIGH * t1).hi - 1) / (1 - s0)**2
        self.assertEqual((t1 - t0) * self.family.weights(t0, t1, s0, s1).lo * s0**2 * prob, d)
        weights = {k: F(v) for k, v in source["plafond"]["poids_des_calls"].items()}
        calls = [q for q in self.quotes if q.id in weights]
        cost = 2 * sum(weights[q.id] * (black_call(q.k, HIGH * q.t).hi - q.bid) for q in calls)
        self.assertEqual(lo, HIGH - cost / d)
        t0, t1, s0, s1 = self.family.support()
        for i in range(4):
            ta, tb = t0 + i * (t1 - t0) / 4, t0 + (i + 1) * (t1 - t0) / 4
            for j in range(32):
                left, right = s0 + j * (s1 - s0) / 32, s0 + (j + 1) * (s1 - s0) / 32
                hedge = F(0)
                for q in calls:
                    v = HIGH * I(q.t - tb, q.t - ta)
                    d1 = (log(I(left, right) / q.k) + v / 2) / sqrt(v)
                    energy = F(ceil(32 * max(abs(d1.lo), abs(d1.hi))**2 / 2), 32)
                    if energy <= 100:
                        hedge += weights[q.id] * exp_point(-energy).lo / (right * sqrt(2 * pi() * v).hi)
                self.assertGreaterEqual(hedge, self.family.weights(ta, tb, left, right).hi)

    def test_beta_lower_variance_endpoint_is_symmetric(self):
        floor = F(1, 5)
        quotes = []
        for q in self.quotes:
            p = black_call(q.k, floor * q.t)
            quotes.append(replace(q, bid=max(F(0), 1 - q.k, p.lo - F(1, 10**8)), ask=p.hi + F(1, 10**8)))
        lo, hi, _ = bounds(quotes, self.family, floor, HIGH, floor, None)
        self.assertEqual(lo, floor)
        self.assertLess(hi - lo, F(1, 100))

    def test_no_later_quote_and_extreme_gamma(self):
        early = [q for q in self.quotes if q.t == F(1, 4)]
        self.assertEqual(bounds(early, self.family, LOW, HIGH, HIGH, None)[:2], (LOW, HIGH))
        q = self.quotes[-1]
        self.assertEqual(gamma_lower(q, LOW, F(0), q.t - F(1, 2**400), F(1), F(2)), 0)

    def test_r3_corridors_and_all_constant_witnesses(self):
        qmap = {q.id: q for q in self.r3}
        corridor = (F(13, 20), F(3, 2))
        for name, decimal in (("synthetic_1_4", ("0.039810", "0.009797")),
                              ("synthetic_2_4", ("0.039420", "0.008310"))):
            _, lo, hi, proof = gamma_bounds(self.r3, name, F(1, 25), LOW, HIGH, corridor=corridor)
            est = cli.estimate(lo, hi, F(1, 100), 6)["decimal"]
            self.assertTrue(est["precision_cible_garantie"])
            self.assertEqual((est["estimation"], est["erreur_absolue_au_plus"]), decimal)
            for rate in (F("0.039"), F("0.04"), F("0.041")):
                check_witness(self.r3, LOW, HIGH, rate, None)
                self.assertLessEqual(lo, rate)
                self.assertLessEqual(rate, hi)
            d = F(proof["denominateur_au_moins"])
            outside = F(proof["masse_gamma_hors_corridor_au_plus"])
            self.assertGreater(d, 0)
            self.assertGreater(outside, 0)
            bid, ask = qmap[name].bid, qmap[name].ask
            b = F(1, 25)
            b_lo, b_hi = map(F, proof["prix_Black"])
            self.assertEqual(lo, b + (2 * (bid - b_hi) - (HIGH - b) * outside) / d)
            self.assertEqual(hi, b + (2 * (ask - b_lo) + (b - LOW) * outside) / d)
            # Constant Black variance b gives E[S_t² Gamma_t] = Gamma_0
            # and the gamma-weighted log-price law N(0, bt*(T-t)/T) at K=1.
            # The latter follows by multiplying the lognormal density by S² Gamma.
            T = float(qmap[name].t)
            rate = float(b)
            g0 = exp(-rate * T / 8) / root(2 * circle * rate * T)
            def outside_prob(t):
                sd = root(rate * t * (T - t) / T)
                return ndtr(ln(float(corridor[0])) / sd) + ndtr(-ln(float(corridor[1])) / sd)
            outside_ref = g0 * quad(outside_prob, 0, T, epsabs=1e-12)[0]
            self.assertLessEqual(outside_ref, float(outside))
            self.assertLessEqual(float(d), T * g0 - outside_ref)

    def test_wider_spreads_and_incompatible_witness(self):
        wide = [replace(q, bid=max(F(0), 1 - q.k), ask=F(1)) for q in self.r3]
        _, lo, hi, _ = gamma_bounds(wide, "synthetic_2_4", F(1, 25), LOW, HIGH)
        self.assertEqual((lo, hi), (LOW, HIGH))
        self.assertFalse(cli.estimate(lo, hi, F(1, 100), 6)["precision_cible_garantie"])
        with self.assertRaisesRegex(ValueError, "hors spread"):
            bounds(self.quotes, self.family, LOW, HIGH, F(1, 25), None)
        with self.assertRaisesRegex(ValueError, "corridor"):
            gamma_bounds(self.r3, "synthetic_2_4", F(1, 25), LOW, HIGH, corridor=(F(1), F(2)))

    def test_output_does_not_replace_inputs(self):
        with tempfile.TemporaryDirectory() as t:
            target = Path(t) / "domain.json"
            target.write_text('{"k":1}', encoding="utf-8")
            with self.assertRaises(ValueError):
                cli.write_report(target, {"schema": cli.SCHEMA})
            self.assertEqual(target.read_text(encoding="utf-8"), '{"k":1}')


if __name__ == "__main__":
    unittest.main()
