import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import sys
import time
import numpy as np
from . import __version__
from .backend import Backend
from .continuous import Window
from .data import collect_yahoo,normalize,estimate_curves
from .io import save_json,read_quotes
from .p2 import certify_polynomial
from .p3 import common_american_models
from .progress import Progress
from .workflow import synthetic_quotes,run_identification,verify_run,frontier


def _csv(value):return [s.strip() for s in value.split(',') if s.strip()]


def _window(value):
    try:return Window.make(*_csv(value))
    except Exception as e:raise argparse.ArgumentTypeError("Expected T,K,hT,hK; compact support in T,K>0") from e


def parser():
    p=argparse.ArgumentParser(prog='dupire-id',description='Dupire P1/P2/P3: portee des garanties explicite')
    p.add_argument('--version',action='version',version=__version__)
    sub=p.add_subparsers(dest='command',required=True)
    def output(cmd,default):cmd.add_argument('--out',default=default)
    def progress(cmd):
        cmd.add_argument('--progress',choices=['plain','json','off'],default='plain')
        cmd.add_argument('--heartbeat',type=float,default=2.)
    def engine(cmd):
        cmd.add_argument('--backend',choices=['auto','numpy','cpp','cuda'],default='auto')
        cmd.add_argument('--threads',type=int,default=None)
    def identify(cmd):
        output(cmd,'runs/identify');progress(cmd);engine(cmd)
        cmd.add_argument('--amin',default='1/400',help='Borne de VARIANCE annuelle, pas de volatilite')
        cmd.add_argument('--amax',default='1/4')
        cmd.add_argument('--levels',default='8,16',help='Nombres de cellules par axe pour les encadrements continus')
        cmd.add_argument('--nt',type=int,default=24);cmd.add_argument('--nk',type=int,default=45)
        cmd.add_argument('--window',type=_window,default=None)
        cmd.add_argument('--epsilon',default='1/100',help='Erreur absolue cible sur la VARIANCE moyenne')
        cmd.add_argument('--a0',default='1/25');cmd.add_argument('--witness-epsilon',default='1/50')
        cmd.add_argument('--lp-time-limit',type=float,default=120)
        cmd.add_argument('--no-discrete',action='store_true')
        cmd.add_argument('--resume',action='store_true');cmd.add_argument('--overwrite',action='store_true')
    cmd=sub.add_parser('doctor',help='Verifie import, pilote CUDA, compilation NVRTC et kernel reel')
    engine(cmd);progress(cmd)
    cmd=sub.add_parser('demo',help='Cas synthetique reproductible avec temoins continus verifiables')
    identify(cmd);cmd.set_defaults(out='runs/demo')
    cmd.add_argument('--spread',default='1/500',help='Demi-spread normalise')
    cmd=sub.add_parser('run',help='Identification sur calls europeens normalises')
    identify(cmd);cmd.add_argument('--quotes',required=True)
    cmd=sub.add_parser('verify',help='Recalcule les preuves continues et verifie les duals exacts sans HiGHS')
    cmd.add_argument('--run',required=True);progress(cmd)
    cmd=sub.add_parser('collect-yahoo',help='SPXW PM europeen, controle calendrier, delais et erreurs')
    output(cmd,'runs/yahoo');progress(cmd)
    cmd.add_argument('--symbol',default='^SPX');cmd.add_argument('--root',default='SPXW')
    cmd.add_argument('--min-days',type=int,default=14);cmd.add_argument('--max-days',type=int,default=365)
    cmd.add_argument('--max-expiries',type=int,default=12);cmd.add_argument('--timeout',type=float,default=35)
    cmd.add_argument('--retries',type=int,default=2);cmd.add_argument('--allow-closed',action='store_true')
    cmd=sub.add_parser('estimate-curves',help='Scenario F,D numerique compatible avec les intervalles de parite')
    cmd.add_argument('--raw',required=True);output(cmd,'curves.csv');progress(cmd)
    cmd.add_argument('--discount-min',type=float,default=.8);cmd.add_argument('--discount-max',type=float,default=1.2)
    cmd.add_argument('--min-pairs',type=int,default=5)
    cmd=sub.add_parser('normalize',help='Normalisation rationnelle exacte conditionnelle a F,D fixes')
    cmd.add_argument('--raw',required=True);cmd.add_argument('--curves',required=True)
    output(cmd,'normalized.csv');progress(cmd)
    cmd=sub.add_parser('frontier',help='Cartographie des encadrements continus selon hT et hK')
    cmd.add_argument('--quotes',required=True);output(cmd,'runs/frontier');progress(cmd)
    for arg in ['t','k','ht','hk']:cmd.add_argument('--'+arg,required=True)
    cmd.add_argument('--amin',default='1/400');cmd.add_argument('--amax',default='1/4')
    cmd.add_argument('--epsilon',default='1/100');cmd.add_argument('--cells',type=int,default=16)
    cmd=sub.add_parser('p2',help='Certificat continu par residu polynomial et principe du maximum')
    cmd.add_argument('--candidate',required=True);cmd.add_argument('--quotes')
    output(cmd,'runs/p2.json');progress(cmd)
    cmd=sub.add_parser('p3',help='Ensemble de vecteurs europeens issu de memes modeles americains candidats')
    cmd.add_argument('--quotes',required=True);cmd.add_argument('--dividends',required=True)
    cmd.add_argument('--spot',type=float,required=True);cmd.add_argument('--rate',type=float,default=.03)
    cmd.add_argument('--yield',dest='yield_',type=float,default=0.)
    cmd.add_argument('--variance-grid',default='.02,.03,.04,.06,.09')
    cmd.add_argument('--skew-grid',default='-.4,-.2,0,.2,.4')
    cmd.add_argument('--nx',type=int,default=257);cmd.add_argument('--steps',type=int,default=252)
    cmd.add_argument('--batch',type=int,default=32);cmd.add_argument('--no-refine',action='store_true')
    output(cmd,'runs/p3');progress(cmd);engine(cmd)
    cmd=sub.add_parser('benchmark',help='Mesure reelle et parite numerique CPU/C++/CUDA, sans promesse de vitesse')
    engine(cmd);progress(cmd);output(cmd,'runs/benchmark.json')
    cmd.add_argument('--models',type=int,default=512);cmd.add_argument('--contracts',type=int,default=256)
    cmd.add_argument('--repeats',type=int,default=5)
    cmd=sub.add_parser('benchmark-fd',help='Temps et parite CPU/CUDA du nouveau pricer de calibration par lots')
    engine(cmd);progress(cmd);output(cmd,'runs/benchmark_fd.json')
    cmd.add_argument('--models',type=int,default=64);cmd.add_argument('--nodes',type=int,default=320)
    cmd.add_argument('--repeats',type=int,default=3)
    for name,description in [('tighten','Bornes continues couplees et modeles de diffusion certifies'),
                             ('strong-demo','Demonstration des nouveaux certificats continus'),
                             ('certify','Minorant positif global et recherche des extremites'),
                             ('certify-demo','Demonstration du minorant positif et du raffinement global')]:
        cmd=sub.add_parser(name,help=description)
        certified=name.startswith('certify')
        output(cmd,'runs/certified' if certified else 'runs/strong');progress(cmd);engine(cmd)
        if name in ('tighten','certify'):cmd.add_argument('--quotes',required=True)
        else:
            cmd.add_argument('--a0',default='1/25');cmd.add_argument('--spread',default='1/500')
        cmd.add_argument('--window',type=_window)
        cmd.add_argument('--amin',default='1/400');cmd.add_argument('--amax',default='1/4')
        cmd.add_argument('--outer-levels',default='8')
        cmd.add_argument('--spline-levels',default='8,16' if certified else '12,16',
            help='Certify: cellules locales sur le noyau; tighten: cellules globales par axe')
        cmd.add_argument('--subdivisions',type=int,default=2)
        cmd.add_argument('--quadrature-cells',type=int,default=32 if certified else 24)
        cmd.add_argument('--endpoint-tolerance',default='1/100')
        cmd.add_argument('--base-variances',default=None)
        cmd.add_argument('--density-floor',default='1/100')
        cmd.add_argument('--lp-time-limit',type=float,default=120)
        cmd.add_argument('--denominator-levels',default='8')
        cmd.add_argument('--initial-run',help='Previous verified model run, or JSON model file; models are reverified')
        cmd.add_argument('--expand-marginals',action='store_true',help='Allow spline corrections across observed maturity dates (certify mode)')
        cmd.add_argument('--proposal-solver',choices=('auto','cuts','full'),default='full' if certified else 'auto')
        cmd.add_argument('--resume',action='store_true');cmd.add_argument('--overwrite',action='store_true')
    cmd=sub.add_parser('verify-tighten',help='Recalcule les preuves continues et les modeles, sans reoptimiser')
    cmd.add_argument('--run',required=True);progress(cmd)
    cmd=sub.add_parser('certified-frontier',help='Frontiere avec bornes et temoins globaux sur toutes les fenetres demandees')
    cmd.add_argument('--quotes',required=True);output(cmd,'runs/certified_frontier');progress(cmd);engine(cmd)
    for name in ('t','k','ht','hk'):cmd.add_argument('--'+name,required=True,help='Valeurs rationnelles separees par des virgules')
    cmd.add_argument('--amin',default='1/400');cmd.add_argument('--amax',default='1/4')
    cmd.add_argument('--outer-levels',default='8');cmd.add_argument('--spline-levels',default='8,16')
    cmd.add_argument('--denominator-levels',default='8');cmd.add_argument('--subdivisions',type=int,default=2)
    cmd.add_argument('--quadrature-cells',type=int,default=32);cmd.add_argument('--endpoint-tolerance',default='1/100')
    cmd.add_argument('--base-variances',default='1/25');cmd.add_argument('--lp-time-limit',type=float,default=120)
    cmd.add_argument('--proposal-solver',choices=('auto','cuts','full'),default='full')
    cmd.add_argument('--expand-marginals',action='store_true')
    cmd.add_argument('--initial-run',help='Reuse certified spline models, with a new certified average at each kernel')
    cmd.add_argument('--resume',action='store_true')
    cmd=sub.add_parser('verify-frontier',help='Recalcule les certificats de chaque fenetre sans reoptimisation')
    cmd.add_argument('--run',required=True);progress(cmd)
    cmd=sub.add_parser('saturate',help='Maximum exact par diffusion a plateau et certification PDE des prix')
    cmd.add_argument('--quotes',required=True);output(cmd,'runs/exact');progress(cmd)
    cmd.add_argument('--window',type=_window)
    cmd.add_argument('--config',help='Diffusion piecewise_quintic_plateau_diffusion_v1; otherwise use the supplied synthetic geometry')
    cmd.add_argument('--initial-run',help='Reuse and reverify an existing global denominator certificate')
    cmd.add_argument('--spatial-cells',type=int,default=192)
    cmd.add_argument('--pulse-cells',type=int,default=32)
    cmd.add_argument('--time-steps',type=int,default=512)
    cmd.add_argument('--refinements',type=int,default=2)
    cmd.add_argument('--resume',action='store_true')
    cmd=sub.add_parser('verify-exact',help='Recalcule residus, prix, plateau et extremites sans solveur PDE')
    cmd.add_argument('--run',required=True);progress(cmd)
    cmd=sub.add_parser('universal-frontier',help='Certifie un continuum de fenetres avec couverture de boites rationnelles')
    cmd.add_argument('--certificate',required=True);output(cmd,'runs/universal_frontier');progress(cmd)
    cmd.add_argument('--domain',help='JSON with finite closed intervals t,k,ht,hk; otherwise the documented initial continuum')
    cmd.add_argument('--max-leaves',type=int,default=128);cmd.add_argument('--max-depth',type=int,default=8)
    cmd=sub.add_parser('verify-universal',help='Recalcule la preuve continue et la couverture complete du domaine')
    cmd.add_argument('--run',required=True);progress(cmd)
    def general_options(cmd):
        cmd.add_argument('--quotes',required=True);progress(cmd)
        cmd.add_argument('--model-run',action='append',default=[],help='Verified fixed-model or 0.4 exact certificate; may be repeated')
        cmd.add_argument('--amin',default='1/400');cmd.add_argument('--amax',default='1/4')
        cmd.add_argument('--tolerance',default='1/100');cmd.add_argument('--max-leaves',type=int,default=8)
        cmd.add_argument('--max-depth',type=int,default=3);cmd.add_argument('--strike-cells',type=int,default=6)
        cmd.add_argument('--target-cells',type=int,default=6);cmd.add_argument('--outer-steps',type=int,default=6)
        cmd.add_argument('--mesh-refinements',type=int,default=1)
        cmd.add_argument('--lp-seconds',type=float,default=30);cmd.add_argument('--fit-models',type=int,default=1)
        cmd.add_argument('--fit-evaluations',type=int,default=1500);cmd.add_argument('--fit-seconds',type=float,default=180)
        cmd.add_argument('--threads',type=int,default=1);cmd.add_argument('--resume',action='store_true')
        cmd.add_argument('--backend',choices=('auto','numpy','cpp','cuda'),default='auto',help='Numerical proposal batches only; proofs and LPs stay on CPU')
    cmd=sub.add_parser('solve-general',help='Frontiere conditionnelle aux vraies cotations, avec bornes uniformes sur chaque boite')
    general_options(cmd);output(cmd,'runs/general');cmd.add_argument('--domain',required=True)
    cmd=sub.add_parser('verify-general',help='Recalcule modeles, theoremes, LP signes et couverture sans optimiseurs')
    cmd.add_argument('--run',required=True);progress(cmd)
    cmd=sub.add_parser('exhaust-general',help='Prefixe fini de domaines qui epuisent les parametres positifs')
    general_options(cmd);output(cmd,'runs/exhaustion');cmd.add_argument('--indices',default='0,1')
    cmd=sub.add_parser('verify-exhaust',help='Verifie le prefixe; ne le confond jamais avec une preuve sur une infinite de domaines')
    cmd.add_argument('--run',required=True);progress(cmd)
    cmd=sub.add_parser('fit-model',help='Calibre un coefficient libre au smile fourni puis certifie ses prix continus')
    cmd.add_argument('--quotes',required=True);cmd.add_argument('--domain',required=True)
    cmd.add_argument('--amin',default='1/400');cmd.add_argument('--amax',default='1/4')
    cmd.add_argument('--controls',type=int,default=6);cmd.add_argument('--threads',type=int,default=1)
    cmd.add_argument('--fit-evaluations',type=int,default=2000);cmd.add_argument('--fit-seconds',type=float,default=180)
    cmd.add_argument('--endpoint',choices=('minimum','maximum'))
    cmd.add_argument('--spatial-cells',type=int,default=128);cmd.add_argument('--pulse-cells',type=int,default=16)
    cmd.add_argument('--time-steps',type=int,default=512);cmd.add_argument('--refinements',type=int,default=2)
    cmd.add_argument('--storage-stride',type=int,default=4)
    cmd.add_argument('--backend',choices=('auto','numpy','cpp','cuda'),default='auto')
    output(cmd,'runs/fitted_model');progress(cmd)
    cmd=sub.add_parser('verify-model',help='Recalcule le certificat de prix du modele libre')
    cmd.add_argument('--run',required=True);progress(cmd)
    cmd=sub.add_parser('construct-general',help='Construit deux coefficients extremes dans une region generale prouvee')
    cmd.add_argument('--model-run',required=True);cmd.add_argument('--window',type=_window,required=True)
    output(cmd,'runs/general_witnesses');progress(cmd)
    cmd=sub.add_parser('verify-general-witnesses',help='Recalcule la construction par couplage, sans nouveau pricer')
    cmd.add_argument('--run',required=True);progress(cmd)
    cmd=sub.add_parser('theorem-regions',help='Calcule les regions generales explicites depuis un modele verifie')
    cmd.add_argument('--model-run',required=True);output(cmd,'runs/theorem_regions.json');progress(cmd)
    cmd=sub.add_parser('target-general',help='Modeles extremes adaptes a chaque famille: horloges exactes puis calibration spatiale a plateau')
    cmd.add_argument('--quotes',required=True);cmd.add_argument('--targets',required=True)
    cmd.add_argument('--model-run',action='append',default=[])
    cmd.add_argument('--amin',default='1/400');cmd.add_argument('--amax',default='1/4')
    cmd.add_argument('--backend',choices=('auto','numpy','cpp','cuda'),default='auto')
    cmd.add_argument('--threads',type=int,default=1);cmd.add_argument('--controls',type=int,default=6)
    cmd.add_argument('--fit-evaluations',type=int,default=1500);cmd.add_argument('--fit-seconds',type=float,default=180)
    cmd.add_argument('--spatial-cells',type=int,default=128);cmd.add_argument('--pulse-cells',type=int,default=16)
    cmd.add_argument('--time-steps',type=int,default=512);cmd.add_argument('--refinements',type=int,default=2)
    cmd.add_argument('--tolerance',default='1/100');cmd.add_argument('--max-leaves',type=int,default=4)
    cmd.add_argument('--max-depth',type=int,default=2);cmd.add_argument('--mesh-refinements',type=int,default=1)
    cmd.add_argument('--strike-cells',type=int,default=6);cmd.add_argument('--target-cells',type=int,default=6)
    cmd.add_argument('--outer-steps',type=int,default=6);cmd.add_argument('--lp-seconds',type=float,default=30)
    cmd.add_argument('--resume',action='store_true');output(cmd,'runs/targeted');progress(cmd)
    cmd=sub.add_parser('verify-targets',help='Recalcule les preuves des familles ciblees sans solveurs')
    cmd.add_argument('--run',required=True);progress(cmd)
    cmd=sub.add_parser('recover-exhaust',help='Copie un ancien calcul termine et reprend les raffinements abandonnes')
    cmd.add_argument('--run',required=True);output(cmd,'runs/exhaustion_recovered');progress(cmd)
    cmd.add_argument('--backend',choices=('auto','numpy','cpp','cuda'),default='auto')
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    progress=Progress(args.progress,args.heartbeat)
    try:
        if args.command=='doctor':
            engine=Backend(args.backend,args.threads,progress)
            numerical=engine.black([.8,1.,1.2],[.01,.04,.09])
            reference=Backend('numpy',progress=Progress('off')).black([.8,1.,1.2],[.01,.04,.09])
            difference=float(np.max(np.abs(numerical-reference)))
            if difference>5e-12:raise RuntimeError('Backend Black parity failed')
            # Exercise the shared tridiagonal/dividend kernel as well as Black.
            grid=np.linspace(0,1,17);div=np.zeros(17);div[8]=.02
            kwargs=dict(models=[[.04,0],[.06,-.2]],strikes=[.9,1.1],iscall=[1,0],
                        maturity_index=[16,16],times=grid,dividends=div,spot=1.,
                        rate=.03,nx=49,american=True)
            price=engine.price(**kwargs)
            reference=Backend('numpy',progress=Progress('off')).price(**kwargs)
            pde_difference=float(np.max(np.abs(price-reference)))
            if pde_difference>5e-11:raise RuntimeError('Backend pricing/dividend parity failed')
            import highspy
            from .native_solver import optimize
            from .lp import exact_dual_lower
            from scipy import sparse
            probe_A=sparse.csr_matrix([[1.]])
            probe=optimize([-1.],probe_A,[.75],[0.],[1.],10,progress=progress,label='continuous_solver_probe')
            if not probe.dual_finite:raise RuntimeError('Continuous HiGHS solver has no finite dual')
            bound=exact_dual_lower(probe_A,[.75],[0.],[1.],[F(-1)],np.maximum(0,-probe.ineqlin.marginals))
            if not F(-75000001,10**8)<bound<=F(-3,4):raise RuntimeError('Exact continuous solver probe failed')
            from .denominator import brownian_floor,verify_brownian
            positive_probe=verify_brownian(brownian_floor(Window.make('3/8',1,'1/16','1/10'),F(1,400),F(1,4)))
            result=dict(version=__version__,python=sys.version,backend=engine.name,
                        threads=engine.threads,diagnostics=engine.diagnostics,
                        cpp_openmp=bool(engine.native.openmp()) if engine.native else None,
                        max_abs_black_difference=difference,
                        max_abs_pricer_difference=pde_difference,
                        certificates='CPU rational arithmetic and outward binary64 intervals; CUDA never validates certificates',
                        continuous_solver='HiGHS '+highspy.Highs().version(),
                        continuous_solver_probe_exact_bound=str(bound),positive_denominator_probe=str(positive_probe))
            if engine.cp is not None:
                from .fd_acceleration import CudaFD
                result['diffusion_batch_cuda']=CudaFD(progress).diagnostics
                props=engine.cp.cuda.runtime.getDeviceProperties(0)
                name=props['name'];result['gpu']=name.decode() if isinstance(name,bytes) else name
        elif args.command in ('demo','run'):
            if args.command=='demo':
                Path(args.out).mkdir(parents=True,exist_ok=True)
                path=Path(args.out)/'synthetic_input.csv'
                synthetic_quotes(path,args.spread,args.a0,args.backend,progress)
            else:path=args.quotes
            result=run_identification(path,args.out,args.window,args.amin,args.amax,
                tuple(int(x) for x in _csv(args.levels)),args.nt,args.nk,args.epsilon,
                args.a0,args.witness_epsilon,args.backend,args.threads,args.lp_time_limit,
                not args.no_discrete,args.progress,args.heartbeat,args.resume,args.overwrite)
            result=dict(status=result['status'],output=str(Path(args.out).resolve()),
                        decision=result.get('decision'),discrete_status=(result.get('discrete') or {}).get('status'),
                        report=str((Path(args.out)/'report.html').resolve()))
        elif args.command=='verify':
            result=verify_run(args.run,progress);save_json(Path(args.run)/'verify.json',result)
        elif args.command=='collect-yahoo':
            result=collect_yahoo(args.out,args.symbol,args.root,args.min_days,args.max_days,
                                 args.max_expiries,args.timeout,args.retries,args.allow_closed,progress)
        elif args.command=='estimate-curves':
            Path(args.out).parent.mkdir(parents=True,exist_ok=True)
            result=estimate_curves(args.raw,args.out,args.discount_min,args.discount_max,args.min_pairs,progress)
        elif args.command=='normalize':result=dict(quotes=normalize(args.raw,args.curves,args.out,progress),output=args.out)
        elif args.command=='frontier':
            rows=frontier(args.quotes,args.out,args.t,args.k,_csv(args.ht),_csv(args.hk),
                          args.amin,args.amax,args.epsilon,args.cells,progress)
            result=dict(windows=len(rows),output=args.out)
        elif args.command=='p2':
            config=json.loads(Path(args.candidate).read_text(encoding='utf-8'))
            result=certify_polynomial(config,read_quotes(args.quotes) if args.quotes else (),progress)
            save_json(args.out,result)
        elif args.command=='p3':
            result=common_american_models(args.quotes,args.dividends,args.out,args.spot,args.rate,args.yield_,
                [float(x) for x in _csv(args.variance_grid)],[float(x) for x in _csv(args.skew_grid)],
                args.nx,args.steps,args.backend,args.threads,args.batch,not args.no_refine,progress)
        elif args.command in ('tighten','strong-demo','certify','certify-demo'):
            from . import tightening
            if args.command in ('strong-demo','certify-demo'):
                Path(args.out).mkdir(parents=True,exist_ok=True)
                path=Path(args.out)/'synthetic_input.csv'
                synthetic_quotes(path,args.spread,args.a0,args.backend,progress)
            else:
                Backend(args.backend,args.threads,progress);path=args.quotes
            variances=_csv(args.base_variances) if args.base_variances else [getattr(args,'a0','1/25')]
            result=tightening.run(path,args.out,args.window,args.amin,args.amax,
                tuple(map(int,_csv(args.outer_levels))),tuple(map(int,_csv(args.spline_levels))),
                args.subdivisions,args.quadrature_cells,args.endpoint_tolerance,variances,args.density_floor,
                args.lp_time_limit,args.progress,args.heartbeat,args.resume,args.overwrite,
                positive_denominator=args.command.startswith('certify'),
                denominator_levels=tuple(map(int,_csv(args.denominator_levels))),
                expanded_models=args.command.startswith('certify'),initial_run=args.initial_run,
                expand_marginals=args.expand_marginals,proposal_solver=args.proposal_solver)
            floor=result.get('denominator_floor')
            result=dict(status=result['status'],outer=result['outer']['approximate'],
                        minimum_model=result['minimum_model']['average']['approximate'],
                        maximum_model=result['maximum_model']['average']['approximate'],
                        gap=result['gap'],denominator_lower=floor['lower_bound'] if floor else None,
                        output=str(Path(args.out).resolve()))
        elif args.command=='verify-tighten':
            from . import tightening
            result=tightening.verify(args.run,progress);save_json(Path(args.run)/'verify_strong.json',result)
        elif args.command=='certified-frontier':
            from . import certified_frontier
            Backend(args.backend,args.threads,progress)
            result=certified_frontier.run(args.quotes,args.out,_csv(args.t),_csv(args.k),_csv(args.ht),_csv(args.hk),
                args.amin,args.amax,tuple(map(int,_csv(args.outer_levels))),tuple(map(int,_csv(args.spline_levels))),
                tuple(map(int,_csv(args.denominator_levels))),args.subdivisions,args.quadrature_cells,
                args.endpoint_tolerance,_csv(args.base_variances),args.lp_time_limit,args.progress,args.heartbeat,args.resume,
                proposal_solver=args.proposal_solver,expand_marginals=args.expand_marginals,initial_run=args.initial_run)
            result=dict(status=result['status'],windows=len(result['windows']),requested_grid_certified=result['requested_grid_certified'],
                        universal_frontier_proved=False,output=str(Path(args.out).resolve()))
        elif args.command=='verify-frontier':
            from . import certified_frontier
            result=certified_frontier.verify(args.run,progress)
        elif args.command=='saturate':
            from . import exact_endpoint
            result=exact_endpoint.run(args.quotes,args.out,args.window or Window.make('3/8',1,'1/16','1/10'),
                args.config,args.initial_run,args.spatial_cells,args.pulse_cells,args.time_steps,args.refinements,
                args.progress,args.heartbeat,args.resume)
            result={key:result[key] for key in ('status','maximum_exact_proved','minimum_exact_proved','maximum','minimum','denominator_lower','maximum_gap','unbounded_parameter_frontier_solved')}
            result['output']=str(Path(args.out).resolve())
        elif args.command=='verify-exact':
            from . import exact_endpoint
            result=exact_endpoint.verify(args.run,progress)
        elif args.command=='universal-frontier':
            from . import universal_frontier
            result=universal_frontier.run(args.certificate,args.out,args.domain,args.max_leaves,args.max_depth,args.progress,args.heartbeat)
            result={key:result[key] for key in ('status','domain','exact_leaf_boxes','open_leaf_boxes','every_window_in_requested_domain_solved_exactly','unbounded_parameter_frontier_solved')}
        elif args.command=='verify-universal':
            from . import universal_frontier
            result=universal_frontier.verify(args.run,progress)
        elif args.command in ('solve-general','exhaust-general'):
            from . import general_frontier,exhaustion
            from .families import Family
            options=dict(amin=args.amin,amax=args.amax,tolerance=args.tolerance,max_leaves=args.max_leaves,
                max_depth=args.max_depth,strike_cells=args.strike_cells,target_cells=args.target_cells,
                outer_steps=args.outer_steps,lp_seconds=args.lp_seconds,fit_models=args.fit_models,
                fit_evaluations=args.fit_evaluations,fit_seconds=args.fit_seconds,threads=args.threads,mesh_refinements=args.mesh_refinements,backend=args.backend)
            if args.command=='solve-general':
                family=Family.make(json.loads(Path(args.domain).read_text(encoding='utf-8')))
                result=general_frontier.run(args.quotes,args.out,family,model_runs=args.model_run,
                    progress_mode=args.progress,heartbeat=args.heartbeat,resume=args.resume,**options)
                result={k:result[k] for k in ('status','model_class_empty_certified','verified_model_count','certified_leaf_boxes','open_leaf_boxes',
                    'all_parameters_in_requested_domain_solved','all_positive_parameters_solved','all_smiles_solved','universal_convergence_proved')}
            else:
                result=exhaustion.run(args.quotes,args.out,tuple(map(int,_csv(args.indices))),args.model_run,
                    args.progress,args.heartbeat,args.resume,**options)
            result['output']=str(Path(args.out).resolve())
        elif args.command=='verify-general':
            from . import general_frontier
            result=general_frontier.verify(args.run,progress)
        elif args.command=='verify-exhaust':
            from . import exhaustion
            result=exhaustion.verify(args.run,progress)
        elif args.command=='fit-model':
            from . import calibrate_general,fixed_model
            from .families import Family
            q=read_quotes(args.quotes);family=Family.make(json.loads(Path(args.domain).read_text(encoding='utf-8')))
            cfg=calibrate_general.template(q,family,F(args.amin),F(args.amax),args.controls)
            proposer=calibrate_general.Proposal(cfg,q,family.center(),threads=args.threads,
                max_evaluations=args.fit_evaluations,max_seconds=args.fit_seconds,progress=progress,backend=args.backend)
            cfg,diag=proposer.fit(args.endpoint)
            result=fixed_model.run(cfg,args.quotes,args.out,args.spatial_cells,args.pulse_cells,args.time_steps,args.refinements,args.progress,args.heartbeat,args.storage_stride)
            save_json(Path(args.out)/'fit.json',diag)
            result=dict(all_continuous_price_intervals_contained=result['all_quotes_compatible'],proposal=diag,
                maximum_price_error=float(max(F(p['eta']) for p in result['prices'])),output=str(Path(args.out).resolve()))
        elif args.command=='verify-model':
            from . import fixed_model
            result=fixed_model.verify(args.run,progress)
        elif args.command=='construct-general':
            from . import general_witnesses
            result=general_witnesses.run(args.model_run,args.out,args.window,progress)
            result={k:result[k] for k in ('minimum','maximum','maximum_price_change','numeric_denominator_lower','all_positive_parameters_solved','all_smiles_solved')}
        elif args.command=='verify-general-witnesses':
            from . import general_witnesses
            result=general_witnesses.verify(args.run,progress)
        elif args.command=='theorem-regions':
            from . import general_theorems,fixed_model
            fixed_model.verify(args.model_run,progress);model=fixed_model.Model(args.model_run)
            result=general_theorems.regions(model.config,model.quotes,model.pricing)
            save_json(args.out,result)
        elif args.command=='target-general':
            from . import targeted
            result=targeted.run(args.quotes,args.targets,args.out,args.model_run,args.amin,args.amax,
                args.backend,args.threads,args.fit_evaluations,args.fit_seconds,args.controls,
                args.spatial_cells,args.pulse_cells,args.time_steps,args.refinements,
                args.progress,args.heartbeat,args.resume,tolerance=args.tolerance,
                max_leaves=args.max_leaves,max_depth=args.max_depth,strike_cells=args.strike_cells,
                target_cells=args.target_cells,outer_steps=args.outer_steps,lp_seconds=args.lp_seconds,
                mesh_refinements=args.mesh_refinements)
            result={k:result[k] for k in ('targets','all_requested_targets_solved','all_positive_parameters_solved','all_smiles_solved')}
            result['output']=str(Path(args.out).resolve())
        elif args.command=='verify-targets':
            from . import targeted
            result=targeted.verify(args.run,progress)
        elif args.command=='recover-exhaust':
            from . import recovery
            result=recovery.run(args.run,args.out,args.progress,args.heartbeat,args.backend)
            result={k:result[k] for k in ('shells','all_requested_shells_solved','all_positive_parameters_solved','all_smiles_solved')}
            result['output']=str(Path(args.out).resolve())
        elif args.command=='benchmark-fd':
            from .fd_acceleration import benchmark
            result=benchmark(args.backend,args.models,args.nodes,args.repeats,args.threads or 1,progress)
            save_json(args.out,result)
        elif args.command=='benchmark':
            if args.models<1 or args.contracts<1 or args.repeats<1:raise ValueError('Positive benchmark sizes required')
            engine=Backend(args.backend,args.threads,progress)
            k=np.geomspace(.5,2,args.contracts);v=np.linspace(.005,.5,args.models)
            reference=Backend('numpy',progress=Progress('off')).black(k,v)
            engine.black(k,v)  # compilation/warmup excluded and disclosed
            timings=[]
            for i in range(args.repeats):
                t=time.perf_counter();value=engine.black(k,v);timings.append(time.perf_counter()-t)
                progress.event('benchmark','running',done=i+1,total=args.repeats)
            delta=float(np.max(np.abs(value-reference)))
            if delta>5e-12:raise RuntimeError(f'Black backend parity failed: {delta}')
            result=dict(backend=engine.name,models=args.models,contracts=args.contracts,
                        median_s=float(np.median(timings)),timings_s=timings,max_abs_error=delta,
                        includes_host_device_transfers=engine.name=='cuda',warmup_excluded=True,
                        workload='batched analytic Black calls; not an end-to-end LP benchmark')
            save_json(args.out,result)
        print(json.dumps(result,indent=2,ensure_ascii=False,default=str,allow_nan=False))
    except KeyboardInterrupt:
        progress.event('command','interrupted');raise SystemExit(130)
    except Exception as e:
        progress.event('command','failed',error=str(e))
        print(f'ERROR: {e}',file=sys.stderr,flush=True)
        raise SystemExit(2)


if __name__=='__main__':main()
