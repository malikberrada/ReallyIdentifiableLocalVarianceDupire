"""Archive, certify and replay an identified slice of European Yahoo SPXW options."""
import argparse
from fractions import Fraction as F
from hashlib import sha256
import json
from pathlib import Path
import sys

from .io import save_json
from .spxw_books import decode, endpoints, expiry_book, parity, parity_conflict, select_pairs, span
from .spxw_capture import capture
from .spxw_gamma import normalized, prove_gamma
from .spxw_witness import propose, verify_model

SCHEMA='dupire_identify.yahoo_spxw_gamma.v1'


def digest(raw):
    return sha256(raw).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def validate(settings):
    amin,amax,eps=map(F,(settings['amin'],settings['amax'],settings['epsilon']))
    if not 0<amin<=amax or eps<0 or not 0<=settings['digits']<=16:
        raise ValueError('Invalid variance bounds, precision or decimal digits')
    if settings['reference']!='auto' and F(settings['reference'])<=0:
        raise ValueError('Reference variance must be positive')
    band=settings['discount']
    if band and (len(band)!=2 or not 0<F(band[0])<=F(band[1])):
        raise ValueError('Invalid positive discount bounds')
    corridor=settings['corridor']
    if corridor and (len(corridor)!=2 or not 0<F(corridor[0])<F(corridor[1])):
        raise ValueError('Invalid normalized price corridor')
    if settings['selection'] not in ('strikes','all','anchors') or any(F(v)<=0 for v in settings['levels']):
        raise ValueError('Invalid strike rule')
    if settings['selection']!='all' and len(settings['levels'])<2:
        raise ValueError('At least two distinct strike levels are required')


def inputs(raw,calendar,settings,audit_path=None):
    validate(settings)
    snap=decode(raw)
    chain,spot,t,audit=expiry_book(snap,settings,decode(calendar) if calendar is not None else None)
    try:
        pairs=select_pairs(chain,spot,settings,audit)
    finally:
        if audit_path is not None:
            save_json(audit_path,audit)
    if settings['call_id']:
        target=next((x for x in pairs if x.call.id==settings['call_id']),None)
        if target is None:
            raise ValueError('Target call ID is not among the retained pairs')
    else:
        target=min(pairs,key=lambda x:(abs(x.k-spot),x.k))
    return pairs,t,audit,target


def build(raw,calendar,settings,pairs,t,audit,target,poly,model,positions=None):
    amin,amax=map(F,(settings['amin'],settings['amax']))
    witness=verify_model(model,pairs,t,amin,amax,settings['discount'])
    gamma=prove_gamma(poly,pairs,target,t,settings,positions)
    rows=[]
    for p in pairs:
        k,c,put=normalized(poly,p)
        rows.append(dict(strike=str(p.k),call_id=p.call.id,put_id=p.put.id,k=endpoints(k),
                         normalized_call=endpoints(c),normalized_put=endpoints(put)))
    curves=dict(coordinates=['D','A=D*F'],vertices=[[str(d),str(a)] for d,a in poly],
                D=endpoints(span(poly,lambda d,a:d)),A=endpoints(span(poly,lambda d,a:a)),
                F=endpoints(span(poly,lambda d,a:a/d)),discount_prior=settings['discount'],normalized_quotes=rows,
                curve_uncertainty_propagated=True)
    local_variance=([str(F(model['variance']))]*2 if model['kind']=='constant_variance' else
                    [str(op(F(x)**2/t for x in model['slopes'])) for op in (min,max)])
    witness.update(model_kind=model['kind'],annual_local_variance_range=local_variance)
    est=gamma['estimate']
    return dict(schema=SCHEMA,request_completed=est['exact_target_met'] and est['displayed_target_met'],
                certified_outer_interval=True,witness_verified=True,settings=settings,
                scope='All bounded-variance continuous normalized martingale diffusions and positive deterministic curves compatible with ALL retained bid/asks at this expiry',
                units='annual local variance',valuation_assumption='archived bid/asks treated as simultaneous constraints at snapshot start',
                market_synchronization_verified=False,optimal_extrema_certified=False,expiry=audit['expiry'],
                data_source='yfinance',raw_sha256=digest(raw),calendar_sha256=digest(calendar) if calendar is not None else None,
                retained_contracts=len(audit['retained']),parity=curves,witness=witness,gamma=gamma)


def certify(snapshot,out,settings,calendar_file=None):
    raw=Path(snapshot).read_bytes()
    supplied=Path(calendar_file).read_bytes() if calendar_file else None
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    (out/'raw_snapshot.json').write_bytes(raw)
    if supplied is not None:
        (out/'session_closes.json').write_bytes(supplied)
    try:
        pairs,t,audit,target=inputs(raw,supplied,settings,out/'audit.json')
        if supplied is None:
            # Freeze the date actually used, including calendar lookups for older captures.
            table=dict(source=audit['calendar_source'],sessions={audit['expiry']:audit['session_close_utc']})
            save_json(out/'session_closes.json',table)
            supplied=(out/'session_closes.json').read_bytes()
        poly=parity(pairs,settings['discount'])
        if not poly:
            conflict=negative(raw,supplied,settings,audit,pairs)
            save_json(out/'infeasibility.json',conflict)
            seal(out,['raw_snapshot.json','session_closes.json','audit.json','infeasibility.json'])
            raise ValueError('Empty exact put-call parity polygon; see infeasibility.json for the two conflicting discount bounds')
        model=propose(pairs,t,F(settings['amin']),F(settings['amax']),poly,settings['discount'])
        cert=build(raw,supplied,settings,pairs,t,audit,target,poly,model)
        save_json(out/'model.json',model)
        save_json(out/'certificate.json',cert)
        names=['raw_snapshot.json','audit.json','model.json','certificate.json']
        if supplied is not None:
            names.append('session_closes.json')
        seal(out,names)
        return cert
    except Exception as exc:
        save_json(out/'failure.json',dict(request_completed=False,settings=settings,raw_sha256=digest(raw),
                                          reason=str(exc),error_type=type(exc).__name__))
        raise


def seal(out,names):
    save_json(out/'manifest.json',{name:digest((out/name).read_bytes()) for name in names})


def negative(raw,calendar,settings,audit,pairs):
    return dict(schema='dupire_identify.spxw_empty_parity.v1',request_completed=False,
                certified_infeasible=True,witness_verified=False,expiry=audit['expiry'],
                retained_contracts=len(audit['retained']),settings=settings,raw_sha256=digest(raw),
                calendar_sha256=digest(calendar),proof=parity_conflict(pairs,settings['discount']))


def verify(out):
    out=Path(out)
    manifest=read_json(out/'manifest.json')
    required={'raw_snapshot.json','audit.json','model.json','certificate.json'}
    negative_names={'raw_snapshot.json','session_closes.json','audit.json','infeasibility.json'}
    if set(manifest) not in (required|{'session_closes.json'},negative_names):
        raise ValueError('Invalid certificate manifest')
    data={name:(out/name).read_bytes() for name in manifest}
    if any(digest(value)!=manifest[name] for name,value in data.items()):
        raise ValueError('A bound evidence file has a different SHA-256')
    raw=data['raw_snapshot.json']
    calendar=data.get('session_closes.json')
    is_empty=set(manifest)==negative_names
    cert=json.loads(data['infeasibility.json' if is_empty else 'certificate.json'])
    settings=cert['settings']
    pairs,t,audit,target=inputs(raw,calendar,settings)
    poly=parity(pairs,settings['discount'])
    if is_empty:
        if poly or audit!=json.loads(data['audit.json']) or cert!=negative(raw,calendar,settings,audit,pairs):
            raise ValueError('Invalid infeasibility proof')
        return cert
    if not poly or audit!=json.loads(data['audit.json']):
        raise ValueError('Inconsistent audit or empty parity set')
    positions=[s['subhedge'] for s in cert['gamma']['time_slabs']]
    replay=build(raw,calendar,settings,pairs,t,audit,target,poly,json.loads(data['model.json']),positions)
    if cert!=replay:
        raise ValueError('Rational replay differs from the saved certificate')
    return replay


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='cmd',required=True)
    c=sub.add_parser('collect',help='Archive ^SPX including all raw calls and puts for the requested expiries')
    c.add_argument('--out',required=True,type=Path,help='New directory')
    c.add_argument('--min-days',type=int,default=7)
    c.add_argument('--max-days',type=int,default=365)
    c.add_argument('--max-expiries',type=int,default=3)
    c.add_argument('--timeout',type=int,default=35)
    c.add_argument('--retries',type=int,default=1)
    c.add_argument('--allow-closed',action='store_true')
    g=sub.add_parser('certify',help='Check one common diffusion and bound a listed call gamma')
    g.add_argument('--snapshot',required=True,type=Path)
    g.add_argument('--out',required=True,type=Path,help='New directory')
    g.add_argument('--calendar',type=Path,help='Frozen close table for a legacy raw snapshot')
    g.add_argument('--expiry',help='YYYY-MM-DD (default: earliest archived ^SPX expiry)')
    selection=g.add_mutually_exclusive_group()
    selection.add_argument('--anchors',nargs='+',help='Nearest matched strikes to these multiples of spot')
    selection.add_argument('--strikes',nargs='+',help='Require these exact listed strikes')
    selection.add_argument('--all-pairs',action='store_true',help='Use every valid matched pair at this expiry')
    g.add_argument('--call-id')
    g.add_argument('--variance-reference',default='0.04',help='Positive annual variance, or auto')
    g.add_argument('--amin',default='1/400')
    g.add_argument('--amax',default='1/4')
    g.add_argument('--discount',nargs=2,metavar=('D_MIN','D_MAX'),help='Optional additional prior, recorded in certificate')
    corridor=g.add_mutually_exclusive_group()
    corridor.add_argument('--corridor',nargs=2,default=['0.65','1.50'])
    corridor.add_argument('--full-space',action='store_true')
    g.add_argument('--epsilon',default='0.01')
    g.add_argument('--decimales',type=int,default=6)
    v=sub.add_parser('verify',help='Recompute the proof without a network request or a new optimization')
    v.add_argument('--run',type=Path,required=True)
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    try:
        if args.cmd=='collect':
            raw=capture(args.out,args.min_days,args.max_days,args.max_expiries,args.timeout,args.retries,args.allow_closed)
            print(f'{len(raw["chains"])} expiry slices | {args.out / "raw_snapshot.json"}')
            return 0
        if args.cmd=='verify':
            cert=verify(args.run)
        else:
            mode='all' if args.all_pairs else 'strikes' if args.strikes else 'anchors'
            levels=[] if args.all_pairs else args.strikes or args.anchors or ['0.98','0.99','1','1.01','1.02']
            s=dict(expiry=args.expiry,selection=mode,levels=levels,call_id=args.call_id,reference=args.variance_reference,
                   amin=args.amin,amax=args.amax,discount=args.discount,epsilon=args.epsilon,digits=args.decimales,
                   corridor=None if args.full_space else args.corridor)
            cert=certify(args.snapshot,args.out,s,args.calendar)
        if cert.get('certified_infeasible'):
            print(json.dumps(dict(verified=True,certified_infeasible=True,request_completed=False,proof=cert['proof']),indent=2))
            return 0
        g=cert['gamma']
        e=g['estimate']
        print(json.dumps(dict(verified=True,request_completed=cert['request_completed'],contracts=cert['retained_contracts'],
              expiry=cert['expiry'],call=g['call_id'],estimate=e['estimate'],absolute_error_at_most=e['absolute_error_at_most'],
              outer=e['display_outer'],market_synchronization_verified=False),indent=2))
        return 0 if args.cmd=='verify' or cert['request_completed'] else 2
    except (ArithmeticError,ImportError,KeyError,OSError,RuntimeError,TypeError,ValueError) as exc:
        print(f'SPXW: {exc}',file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
