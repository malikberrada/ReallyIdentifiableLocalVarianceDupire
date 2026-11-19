"""P3 shared-model candidate search. No full inverse-set certificate is claimed."""
import csv
import json
from pathlib import Path
import numpy as np
from .io import save_json
from .backend import Backend
from .progress import QUIET


def common_american_models(quotes_file,dividends_file,output,spot,rate=.03,yield_=0.,
                           variances=None,skews=None,nx=257,steps=252,backend="auto",
                           threads=None,batch=32,refine=True,progress=QUIET):
    with open(quotes_file,encoding="utf-8-sig",newline="") as f:rows=list(csv.DictReader(f))
    if not rows:raise ValueError("Empty American chain")
    if any(r['style'].lower()!='american' for r in rows):raise ValueError("P3 input must be explicitly American")
    if any(r['kind'] not in ('call','put') for r in rows):raise ValueError("kind must be call/put")
    times=np.array([float(r['t']) for r in rows]);strikes=np.array([float(r['strike']) for r in rows])
    bids=np.array([float(r['bid']) for r in rows]);asks=np.array([float(r['ask']) for r in rows])
    if (np.any(times<=0) or np.any(strikes<=0) or np.any(bids<0) or np.any(asks<bids) or
        not all(np.all(np.isfinite(x)) for x in (times,strikes,bids,asks)) or steps<2):
        raise ValueError("Invalid American quote/time/steps")
    divs=json.loads(Path(dividends_file).read_text(encoding='utf-8'))
    dtime=np.array([float(d['t']) for d in divs]);amount=np.array([float(d['amount']) for d in divs])
    if np.any(dtime<=0) or np.any(amount<0) or not np.all(np.isfinite(dtime)) or not np.all(np.isfinite(amount)):
        raise ValueError("Dividends must be known nonnegative cash amounts at positive times")
    va=np.asarray(variances if variances is not None else np.linspace(.015,.12,24))
    sk=np.asarray(skews if skews is not None else np.linspace(-.5,.5,9))
    models=np.array([(a,s) for a in va for s in sk],dtype=float)
    engine=Backend(backend,threads,progress)
    def price_grid(factor,american):
        grid=np.unique(np.r_[np.linspace(0,times.max(),steps*factor+1),times,
                              dtime[dtime<=times.max()]])
        cash=np.zeros(len(grid))
        for dt,value in zip(dtime,amount):
            if dt<=times.max():cash[np.searchsorted(grid,dt)]+=value
        index=np.searchsorted(grid,times).astype(np.int32)
        return engine.price(models,strikes,[int(r['kind']=='call') for r in rows],index,grid,cash,
                            spot,rate,yield_,nx=1+(nx-1)*factor,american=american,batch=batch)
    with progress.phase("p3_common_model_search",models=len(models),contracts=len(rows)):
        american=price_grid(1,True)
        european=price_grid(1,False)
        coarse_inside=np.all((american>=bids)&(american<=asks),axis=1)
        change=None
        if refine:
            refined=price_grid(2,True)
            refined_european=price_grid(2,False)
            change=np.abs(refined-american)
            inside=np.all((refined>=bids)&(refined<=asks),axis=1)
            # A stability filter, explicitly NOT an error bound.
            stable=np.all((refined-change>=bids)&(refined+change<=asks),axis=1)&inside
            american,european=refined,refined_european
        else:inside=coarse_inside;stable=inside
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'common_models.npz',models=models,american=american,european=european,
                        accepted=inside,stable=stable,refinement_change=change if change is not None else [],
                        strikes=strikes,times=times,bids=bids,asks=asks,
                        ids=np.asarray([r.get('id',str(i)) for i,r in enumerate(rows)]))
    result=dict(scope="finite_candidate_models_with_discrete_exercise_and_domain",
                status="numerical_candidates_only",models_tested=len(models),
                accepted_models=int(inside.sum()),stable_models=int(stable.sum()),
                backend=engine.name,continuous_american_certified=False,
                exercise="Bermudan max on time grid, including ex-dividend dates",
                dividend_jump="V(t-,S)=V(t+,max(S-D,0)); exercise before jump",
                common_model_preserved=True,full_inverse_set_covered=False,
                european_vectors="One entire vector per accepted common model; no independent conversions",
                model_family="a(t,S)=a0*(1+skew*tanh(log(S/spot)))^2",
                refinement_difference_is_error_bound=False,
                input_assumptions=dict(spot=spot,rate=rate,continuous_yield=yield_,
                                       cash_dividends=divs,nx=nx,steps=steps),
                limitations=["Finite parameter family does not cover all admissible diffusions",
                             "No rigorous continuous optimal-stopping error bound",
                             "No rigorous spatial/time/domain error bound in this candidate engine"])
    save_json(out/'p3.json',result)
    return result
