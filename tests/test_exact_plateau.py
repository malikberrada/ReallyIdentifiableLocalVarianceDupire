"""Independent arithmetic, PDE residual, compatibility, and continuum checks."""
from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path
import json
import shutil
import numpy as np
import pytest
from dupire_identify import binary_intervals as bi, plateau, exact_endpoint, universal_frontier
from dupire_identify.continuous import Window
from dupire_identify.io import read_quotes,sha256,save_json,Quote
from dupire_identify.interval import black_call

W=Window.make('3/8',1,'1/16','1/10')
ROOT=Path(__file__).resolve().parents[1]


def test_binary_interval_operations_enclose_exact_rationals():
    assert bi.self_check()
    rng=np.random.default_rng(23)
    for unused in range(200):
        a,b=sorted(F(float(x)) for x in rng.normal(size=2));c,d=sorted(F(float(x)) for x in rng.normal(size=2))
        x,y=bi.B(float(a),float(b)),bi.B(float(c),float(d))
        for z,lo,hi in [(x+y,a+c,b+d),(x-y,a-d,b-c),(x*y,min(a*c,a*d,b*c,b*d),max(a*c,a*d,b*c,b*d))]:
            assert F(float(z.lo))<=lo<=hi<=F(float(z.hi))
        if c*d>0:
            z=x/y;vals=[a/c,a/d,b/c,b/d]
            assert F(float(z.lo))<=min(vals)<=max(vals)<=F(float(z.hi))
    with pytest.raises(ValueError):bi.B(np.nan)
    with pytest.raises(ZeroDivisionError):bi.B(1)/bi.B(-1,1)


def rational_bernstein(c,u):
    from math import comb
    return sum(v*comb(len(c)-1,j)*u**j*(1-u)**(len(c)-1-j) for j,v in enumerate(c))


def test_bernstein_product_elevation_and_restriction_against_exact_values():
    a=[F(2,7),F(-3,11),F(1,3)];b=[F(-2,5),F(4,9)]
    aa=bi.stack([bi.B.rational(v) for v in a],-1);bb=bi.stack([bi.B.rational(v) for v in b],-1)
    for doc,truth in [(bi.product(aa,bb),lambda u:rational_bernstein(a,u)*rational_bernstein(b,u)),
                      (bi.elevate(aa,6),lambda u:rational_bernstein(a,u)),
                      (bi.restrict(aa,bi.B.rational(F(1,5)),bi.B.rational(F(4,5))),lambda u:rational_bernstein(a,F(1,5)+u*F(3,5)))]:
        for j in range(21):
            u=F(j,20);lower=rational_bernstein(list(map(lambda x:F(float(x)),doc.lo)),u)
            upper=rational_bernstein(list(map(lambda x:F(float(x)),doc.hi)),u)
            assert lower<=truth(u)<=upper


def simple_config():
    return dict(kind='piecewise_quintic_plateau_diffusion_v1',amin='1/400',amax='1/4',
        prefix_end='1/4',prefix_variance='1/25',strike_knots=['1/2','3/2'],
        phases=[dict(left='1/4',right='1/2',variance=['1/25','1/25'])])


def polynomial_payload():
    config=simple_config();grid=[F(j,4) for j in range(2,7)];times=[F(1,4),F(3,8),F(1,2)]
    z=np.empty((len(times),4,len(grid)))
    for it,t in enumerate(times):
        beta=t-F(1,4)
        for ik,k in enumerate(grid):
            g=(k-F(1,2))*(F(3,2)-k)
            z[it,:,ik]=list(map(float,[F(3,4)-k/2+beta*g,-F(1,2)+beta*(2-2*k),g,2-2*k]))
    return config,grid,[list(map(str,times))],{'phase_0':z}


def test_entire_polynomial_pde_residual_is_enclosed_and_tampering_rejected():
    c,k,t,z=polynomial_payload();rho=plateau.residual_bounds(c,k,t,z)
    # Exact residual for V=3/4-k/2+(t-1/4)*(k-1/2)*(3/2-k):
    # R=(k-1/2)*(3/2-k)+(t-1/4)*a*k^2.
    for j,(ta,tb) in enumerate(zip(map(F,t[0][:-1]),map(F,t[0][1:]))):
        for u in range(17):
            tt=ta+(tb-ta)*F(u,16)
            for v in range(33):
                kk=k[0]+(k[-1]-k[0])*F(v,32)
                exact=(kk-F(1,2))*(F(3,2)-kk)+(tt-F(1,4))*F(1,25)*kk**2
                assert abs(exact)<=F(rho[0][j])
    bad=deepcopy(t);bad[0][0]='1/8'
    with pytest.raises(ValueError,match='time mesh'):plateau.residual_bounds(c,k,bad,z)
    z['phase_0'][1,0,0]+=1
    with pytest.raises(ValueError,match='boundary'):plateau.residual_bounds(c,k,t,z)


def test_global_diffusion_bounds_and_exact_plateau_geometry(tmp_path):
    from dupire_identify.workflow import synthetic_quotes
    q=synthetic_quotes(tmp_path/'quotes.csv');c=plateau.default_config(q,W)
    plateau.validate_config(c)
    rectangles=plateau.plateau_rectangles(c,F(1,4))
    assert any(exact_endpoint.contains(r,W) for r in rectangles)
    x=np.linspace(.9,1.1,57);assert np.all(plateau.a_values(c,1,x)==.25)
    bad=deepcopy(c);bad['phases'][0]['variance'][0]='-1'
    with pytest.raises(ValueError):plateau.validate_config(bad)
    # Saturation can never be inferred from mere numerical proximity.
    bad=deepcopy(c);bad['phases'][1]['variance'][5]='249999999/1000000000'
    assert not any(exact_endpoint.contains(r,W) for r in plateau.plateau_rectangles(bad,F(1,4)))


def test_independent_black_prices_inside_constant_diffusion_certificates():
    c=simple_config();c['strike_knots']=['1/8','13/20','7/5','8'];c['phases'][0]['variance']=['1/25']*4
    grid=plateau.make_grid(c,32,4);z,t=plateau.solve_numerical(c,grid,128)
    q=[]
    for k in (F(4,5),F(1),F(6,5)):
        q.append(Quote(str(k),F(1,2),k,F(0),F(1)))
    cert=plateau.certify_prices(c,grid,t,z,q)
    for quote,row in zip(q,cert['prices']):
        truth=black_call(quote.k,F(1,25)*quote.t)
        assert F(row['price']['lo'])<=truth.lo<=truth.hi<=F(row['price']['hi'])


def atlas_fixture():
    return dict(rectangles=[dict(t_left='5/16',t_right='7/16',k_left='9/10',k_right='11/10',
        identification_interval=['1/400','1/4'])],temporal_strips=None)


def test_continuum_certificate_covers_four_dimensions_not_samples():
    d=dict(t=['23/64','25/64'],k=['39/40','41/40'],ht=['1/64','1/32'],hk=['1/40','3/40'])
    b=universal_frontier.parse_domain(d);a=atlas_fixture();tree=universal_frontier.cover(b,a)
    assert universal_frontier.check_tree(tree,b,a)==(1,0)
    d['hk'][1]='1/10';b2=universal_frontier.parse_domain(d)
    tree2=universal_frontier.cover(b2,a,max_leaves=8,max_depth=3)
    assert universal_frontier.check_tree(tree2,b2,a)[1]>0
    forged=deepcopy(tree);forged['box']['hk'][1]='1/5'
    with pytest.raises(ValueError):universal_frontier.check_tree(forged,b,a)
    bad={'box':universal_frontier.payload(b2),'status':'exact','proof':tree['proof']}
    with pytest.raises(ValueError):universal_frontier.check_tree(bad,b2,a)


def test_partition_holes_and_unsupported_domains_are_rejected():
    a=atlas_fixture();d=dict(t=['3/8','3/8'],k=['1','1'],ht=['1/64','1/16'],hk=['1/20','1/5'])
    b=universal_frontier.parse_domain(d);tree=universal_frontier.cover(b,a,max_leaves=4,max_depth=2)
    assert tree['status']=='split'
    tree['children'][0]['box']['hk'][1]='1/20'
    with pytest.raises(ValueError):universal_frontier.check_tree(tree,b,a)
    with pytest.raises(ValueError):universal_frontier.parse_domain(dict(d,ht=['0','1/16']))


def test_full_supplied_endpoint_certificate_recomputed_without_numerical_solver(monkeypatch,tmp_path):
    source=ROOT/'examples/exact_initial'
    if not (source/'exact_manifest.json').exists():pytest.skip('Full source example not present in this installation')
    out=tmp_path/'exact';shutil.copytree(source,out)
    def forbidden(*a,**k):raise AssertionError('A verifier must not rerun the PDE solver')
    monkeypatch.setattr(plateau,'solve_numerical',forbidden)
    checked=exact_endpoint.verify(out)
    assert checked['maximum_exact_proved'] and checked['minimum_exact_proved']
    assert checked['maximum']['lo']==checked['maximum']['hi']=='1/4'
    # Recompute with a changed assertion and UPDATED hashes: the mathematics,
    # rather than a checksum mismatch, must reject the fabricated result.
    result=json.loads((out/'result.json').read_text());result['maximum']['lo']='251/1000'
    save_json(out/'result.json',result)
    m=json.loads((out/'exact_manifest.json').read_text());m['artifacts']['result.json']=sha256(out/'result.json')
    save_json(out/'exact_manifest.json',m)
    with pytest.raises(ValueError,match='endpoint claim'):exact_endpoint.verify(out)


def test_native_tridiagonal_matches_independent_dense_solve():
    try:from dupire_identify._native import tridiagonal
    except ImportError:pytest.skip('Optional native extension unavailable')
    n=37;A=np.zeros((3,n));A[0,1:]=-.2;A[1]=1.5;A[2,:-1]=-.3;r=np.linspace(-1,1,n)
    exact=np.linalg.solve(np.diag(A[1])+np.diag(A[0,1:],1)+np.diag(A[2,:-1],-1),r)
    assert np.max(abs(tridiagonal(A,r)-exact))<2e-15


def test_incompatible_prices_never_become_an_exact_maximum(tmp_path):
    from dupire_identify.workflow import synthetic_quotes
    q=synthetic_quotes(tmp_path/'quotes.csv');c=plateau.default_config(q,W)
    result=exact_endpoint.global_claim(c,q,W,{'all_quotes_compatible':False},F(1,8192),None)
    assert not result['maximum_exact_proved'] and result['maximum'] is None


def test_checkpoint_resume_reuses_surface_and_rejects_changed_mesh(monkeypatch,tmp_path):
    source=ROOT/'examples/exact_initial'
    if not (source/'exact_manifest.json').exists():pytest.skip('Source example absent')
    out=tmp_path/'resume';shutil.copytree(source,out);(out/'exact_manifest.json').unlink()
    settings=json.loads((source/'exact_manifest.json').read_text())['settings']
    # Reuse the legacy surface as a payload, but simulate an interrupted run
    # of the currently installed version. Completed 0.4 proofs are checked
    # above; cross-version checkpoint migration is intentionally rejected.
    from dupire_identify import __version__
    settings['version']=__version__
    save_json(out/'checkpoint.json',dict(settings=settings,level=0,
        hashes={name:sha256(out/name) for name in ('surface.npz','mesh.json','diffusion.json')}))
    truth=json.loads((source/'pricing.json').read_text())
    def forbidden(*a,**k):raise AssertionError('Resume unnecessarily reran the PDE')
    monkeypatch.setattr(plateau,'solve_numerical',forbidden)
    # The arithmetic proof is exercised separately; isolate the checkpoint
    # control flow here using the independently recomputed stored proof.
    monkeypatch.setattr(plateau,'certify_prices',lambda *a,**k:truth)
    r=exact_endpoint.run(source/'normalized.csv',out,W,progress_mode='off',resume=True)
    assert r['maximum_exact_proved']
    (out/'exact_manifest.json').unlink()
    (out/'mesh.json').write_text((out/'mesh.json').read_text()+'\n')
    with pytest.raises(ValueError,match='checkpoint'):
        exact_endpoint.run(source/'normalized.csv',out,W,progress_mode='off',resume=True)
