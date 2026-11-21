"""Run with: python -I tests/test_yahoo_spxw.py (from the project directory)."""
import copy
from datetime import datetime, timezone
from fractions import Fraction as F
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dupire_identify import spxw_capture
from dupire_identify.interval import I, black_call
from dupire_identify.spxw_books import (Book, Pair, clip, parity, parity_conflict,
                                       parse_book, settlement, span, utc)
from dupire_identify.spxw_gamma import normalized, price_gap, prove_gamma, rounded_estimate
from dupire_identify.spxw_witness import verify_model
from dupire_identify.yahoo_spxw import certify, inputs, verify

EX=ROOT/'examples'/'yahoo_spxw'


def settings(**changes):
    args=dict(expiry='2026-11-09',selection='anchors',levels=['0.98','0.99','1','1.01','1.02'],
              call_id=None,reference='0.04',amin='1/400',amax='1/4',discount=None,epsilon='.01',digits=6,
              corridor=['0.65','1.50'])
    return dict(args,**changes)


@unittest.skipUnless((EX / "raw_snapshot.json").is_file(), "Optional historical SPXW snapshot")
class SPXWTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=(EX/'raw_snapshot.json').read_bytes()
        cls.closes=(EX/'session_closes.json').read_bytes()
        cls.pairs,cls.t,cls.audit,cls.target=inputs(cls.raw,cls.closes,settings())

    def test_archive_identities_and_temporal_limits(self):
        self.assertEqual(sha256(self.raw).hexdigest(),'0c44b9b66a46b238d891ffbeb21fcaf65299f4987a9cf8b9f51993921e60db41')
        self.assertEqual([p.k for p in self.pairs],[7525,7605,7680,7755,7830])
        self.assertEqual(self.target.call.id,'SPXW261109C07680000')
        self.assertEqual(self.audit['raw_rows'],len(self.audit['rejected'])+len(self.audit['retained']))
        self.assertEqual(len(self.audit['retained_last_trade_utc']),10)
        self.assertFalse(self.audit['bid_ask_synchronization_verified'])
        self.assertEqual(self.t,F(607455446909,15768000000000))

    def test_product_currency_strike_side_and_time(self):
        start=utc(self.audit['snapshot_start'])
        q=dict(contractSymbol='SPXW261109C07680000',strike=7680,bid=89.6,ask=90,currency='USD',contractSize='REGULAR')
        self.assertEqual(parse_book(q,start,'2026-11-09','C').bid,F('89.6'))
        for changes in [dict(contractSymbol=q['contractSymbol'].replace('SPXW',r)) for r in ['SPX','SPXO','SPY','XSP']]+[
            dict(currency='EUR'),dict(contractSize='MINI'),dict(strike=7685),dict(exerciseStyle='American'),
            dict(settlement='AM'),dict(contractSymbol='SPXW261109P07680000'),dict(bid=0),dict(ask='NaN'),
            dict(bid=95),dict(contractSymbol='SPXW261110C07680000'),dict(lastTradeDate='2026-11-09T15:00:00Z')]:
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                parse_book(dict(q,**changes),start,'2026-11-09','C')
        bad=json.loads(self.raw)
        bad['chains'][0]['underlying']['symbol']='SPY'
        with self.assertRaises(ValueError):
            inputs(json.dumps(bad).encode(),self.closes,settings())

    def test_audit_rejects_duplicated_pair_and_bad_timestamps(self):
        snap=json.loads(self.raw)
        c=snap['chains'][0]
        target=next(q for q in c['calls'] if q['contractSymbol']==self.target.call.id)
        c['calls'].append(copy.deepcopy(target))
        raw=json.dumps(snap).encode()
        with self.assertRaisesRegex(ValueError,'explicitly requested'):
            inputs(raw,self.closes,settings(selection='strikes',levels=['7525','7680']))
        c['retrieval_ended_utc']=snap['started_utc']
        with self.assertRaisesRegex(ValueError,'chronology'):
            inputs(json.dumps(snap).encode(),self.closes,settings())
        self.assertEqual(settlement('2026-11-27','2026-11-27T18:00:00Z').hour,18)
        with self.assertRaises(ValueError):
            settlement('2026-11-27','2026-11-27T00:00:00Z')
        with self.assertRaises(ValueError):
            utc('2026-11-09T16:00:00')

    def test_parity_edges_and_curve_dependence(self):
        poly=parity(self.pairs)
        d=span(poly,lambda d,a:d)
        self.assertEqual((d.lo,d.hi),(F(1487,1500),F(377,375)))
        # An interior point and every vertex must be covered by the downstream normalizations.
        sample=poly+[(sum(v[0] for v in poly)/len(poly),sum(v[1] for v in poly)/len(poly))]
        for p in self.pairs:
            strike,cost,put=normalized(poly,p)
            gap=price_gap(poly,p,self.t,F('.04'))
            for disc,a in sample:
                self.assertTrue(p.strip[0]<=a-disc*p.k<=p.strip[1])
                k=p.k*disc/a
                self.assertTrue(strike.lo<=k<=strike.hi)
                bid=max(p.call.bid/a,p.put.bid/a+1-k)
                ask=min(p.call.ask/a,p.put.ask/a+1-k)
                self.assertTrue(cost.lo<=bid<=ask<=cost.hi)
                self.assertTrue(put.lo<=bid+k-1<=ask+k-1<=put.hi)
                black=black_call(k,F('.04')*self.t)
                self.assertLessEqual(gap.lo,bid-black.hi)
                self.assertGreaterEqual(gap.hi,ask-black.lo)
        fixed=parity(self.pairs,['1','1'])
        self.assertTrue(fixed and all(d==1 for d,a in fixed))
        point=clip(fixed,lambda v:min(a for d,a in fixed)-v[1])
        self.assertEqual(len(point),1)

    def test_full_slice_has_constructive_parity_conflict(self):
        pairs,*_=inputs(self.raw,self.closes,settings(selection='all',levels=[]))
        self.assertEqual(len(pairs),78)
        self.assertEqual(parity(pairs),[])
        conflict=parity_conflict(pairs)
        self.assertEqual(conflict['discount_must_be_at_least']['value'],'53/50')
        self.assertEqual(conflict['discount_must_be_at_most']['value'],'49/50')
        with TemporaryDirectory() as tmp:
            run=Path(tmp)/'run'
            with self.assertRaisesRegex(ValueError,'Empty exact'):
                certify(EX/'raw_snapshot.json',run,settings(selection='all',levels=[]),EX/'session_closes.json')
            self.assertTrue(verify(run)['certified_infeasible'])
            self.assertFalse((run/'certificate.json').exists())
            with self.assertRaises(FileExistsError):
                certify(EX/'raw_snapshot.json',run,settings())

    def test_common_model_in_raw_spreads_and_rejects_modified_witness(self):
        model=json.loads((EX/'model.json').read_text())
        result=verify_model(model,self.pairs,self.t,F('0.0025'),F('0.25'))
        self.assertEqual(result['checked_contracts'],10)
        self.assertGreater(F(result['minimum_slack_points']),0)
        self.assertEqual(model['kind'],'gaussian_terminal_loglinear')
        with self.assertRaises(ValueError):
            verify_model(dict(model,median_spot='8000'),self.pairs,self.t,F('.0025'),F('.25'))
        bad=dict(model,slopes=['1']*len(model['slopes']))
        with self.assertRaises(ValueError):
            verify_model(bad,self.pairs,self.t,F('.0025'),F('.25'))

    def test_replay_and_model_independent_certificate(self):
        # A replay must never re-run the parameter fit or the hedge LP.
        with patch('dupire_identify.yahoo_spxw.propose',side_effect=AssertionError), \
             patch('dupire_identify.spxw_subhedge.Subhedges.propose',side_effect=AssertionError):
            cert=verify(EX)
        self.assertTrue(cert['request_completed'])
        self.assertTrue(cert['parity']['curve_uncertainty_propagated'])
        g=cert['gamma']
        self.assertGreater(F(g['denominator_bounds'][0]),F('.30'))
        self.assertTrue(0<F(g['outside_gamma_mass_upper'])<F('1e-15'))
        self.assertEqual(g['estimate']['display_outer'],['0.009487','0.028488'])
        self.assertEqual(g['estimate']['absolute_error_at_most'],'0.009501')

    def test_tampering_with_raw_or_rehashed_assertions(self):
        with TemporaryDirectory() as tmp:
            run=Path(tmp)/'run'
            shutil.copytree(EX,run)
            raw=run/'raw_snapshot.json'
            raw.write_bytes(raw.read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError,'SHA-256'):
                verify(run)
            raw.write_bytes(self.raw)
            path=run/'certificate.json'
            cert=json.loads(path.read_text())
            cert['gamma']['time_slabs'][0]['subhedge']['short']={'0':'-1'}
            path.write_text(json.dumps(cert))
            m=json.loads((run/'manifest.json').read_text())
            m['certificate.json']=sha256(path.read_bytes()).hexdigest()
            (run/'manifest.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'positions'):
                verify(run)

    def test_zero_spread_at_unrepresentable_midpoint_and_rounding(self):
        est=rounded_estimate(I(F(1,3)),F('0.0000001'),6)
        self.assertTrue(est['exact_target_met'])
        self.assertFalse(est['displayed_target_met'])
        self.assertEqual(est['absolute_error_at_most'],'0.000001')
        with self.assertRaises(ValueError):
            prove_gamma(parity(self.pairs),self.pairs,self.target,self.t,settings(corridor=['1','1.1']))

class IndependentSPXWTests(unittest.TestCase):
    def test_known_constant_diffusion_in_full_space(self):
        t=F(1,12)
        variance=F('0.04')
        discount=F(1)
        forward=F(100)
        pairs=[]
        for k in (F(95),F(100),F(105)):
            call=100*black_call(k/100,variance*t)
            put=call+k-forward
            def book(kind,b):
                return Book(kind+str(k),k,b.lo-F('.02'),b.hi+F('.02'))
            pairs.append(Pair(k,book('C',call),book('P',put)))
        poly=parity(pairs,['1','1'])
        model=dict(kind='constant_variance',discount='1',forward='100',variance='.04')
        self.assertTrue(verify_model(model,pairs,t,F('.0025'),F('.25'))['verified'])
        result=prove_gamma(poly,pairs,pairs[1],t,settings(reference='.06',corridor=None,epsilon='0'))
        a,b=map(F,result['estimate']['outer'])
        self.assertLessEqual(a,variance)
        self.assertGreaterEqual(b,variance)
        self.assertFalse(result['estimate']['displayed_target_met'])
        self.assertEqual(F(result['outside_gamma_mass_upper']),0)

    def test_yfinance_worker_keeps_vendor_fields_and_trade_times(self):
        import pandas as pd
        from queue import SimpleQueue
        messages = SimpleQueue()
        connection = SimpleNamespace(send=messages.put, close=lambda: None)
        frame = pd.DataFrame([{'contractSymbol':'SPXW261109C07680000', 'bid':89.6, 'ask':90.0,
                               'lastTradeDate': pd.Timestamp('2026-10-25T11:00:00Z'), 'volume':float('nan')}])
        ticker = Mock()
        ticker.options = ('2026-11-09',)
        ticker.option_chain.return_value = SimpleNamespace(calls=frame, puts=frame.copy(),
                                                          underlying={'symbol':'^SPX'})
        adapter = SimpleNamespace(Ticker=Mock(return_value=ticker))
        with patch.dict(sys.modules, {'yfinance':adapter}), patch.object(spxw_capture,'version',return_value='fixture'):
            spxw_capture._worker(None,connection)
            self.assertEqual(messages.get(),(True, {'expiries':['2026-11-09'],'version':'fixture'}))
            spxw_capture._worker('2026-11-09',connection)
            ok, payload = messages.get()
            self.assertTrue(ok)
            self.assertEqual(payload['calls'][0]['lastTradeDate'], '2026-10-25T11:00:00+00:00')
            self.assertIsNone(payload['calls'][0]['volume'])
            self.assertEqual(payload['underlying']['symbol'], '^SPX')
            self.assertEqual(payload['puts'][0]['bid'],89.6)
            ticker.option_chain.assert_called_once_with('2026-11-09')
            adapter.Ticker.assert_called_with('^SPX')

    def test_capture_persists_full_responses_and_partial_failures(self):
        import pandas as pd
        now=datetime(2026,10,26,19,30,tzinfo=timezone.utc)
        c = {'underlying': {'symbol': '^SPX'},
             'calls': [{'strike': 7680, 'bid': 89.6, 'ask': 90.0}],
             'puts': [{'strike': 7680, 'bid': 86.0, 'ask': 87.0}]}
        def fetch(expiry,timeout):
            if not expiry:
                return dict(expiries=['2026-11-09','2026-11-10'],version='fixture')
            if expiry=='2026-11-10':
                raise RuntimeError('fixture request failure')
            return {k:c[k] for k in ('underlying','calls','puts')}
        sc=SimpleNamespace(is_open_on_minute=lambda x:True,is_session=lambda x:True,
                           session_close=lambda x:pd.Timestamp(x+'T21:00:00Z'))
        with TemporaryDirectory() as tmp,patch.dict(sys.modules,{'exchange_calendars':SimpleNamespace(get_calendar=lambda *a,**k:sc)}), \
             patch.object(spxw_capture,'version',return_value='fixture'):
            o=Path(tmp)/'success'
            raw=spxw_capture.capture(o,7,45,1,10,0,False,fetch,lambda:now)
            self.assertTrue(raw['complete'])
            self.assertEqual(raw['chains'][0]['calls'],c['calls'])
            self.assertFalse(raw['quote_timestamps_available'])
            self.assertEqual(len(raw['listed_expiries']),2)
            o=Path(tmp)/'partial'
            with self.assertRaisesRegex(RuntimeError,'Partial'):
                spxw_capture.capture(o,7,45,2,10,0,False,fetch,lambda:now)
            archived=json.loads((o/'raw_snapshot.json').read_bytes())
            self.assertEqual(len(archived['chains']),1)
            self.assertFalse(archived['complete'])
            self.assertEqual(archived['errors'][0]['expiry'],'2026-11-10')


if __name__=='__main__':
    unittest.main(verbosity=2)
