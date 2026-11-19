"""Public HiGHS C++ interface; keep finite duals even without crossover.

Exact dual verification does not require an optimizer's 'optimal' flag. A
finite multiplier vector is only a proposal; certlp verifies its bound exactly.
"""
from types import SimpleNamespace
import numpy as np
from scipy import sparse
from .progress import QUIET


class WarmLP:
    """Retain the C++ simplex basis across added rows and objective updates."""
    def __init__(self,f,A,b,lo,hi,solver='simplex'):
        import highspy as hs
        A=sparse.csr_matrix(A)
        self.h=hs.Highs();h=self.h;self.n=A.shape[1];self.m=A.shape[0]
        for name,val in (('output_flag',False),('threads',1),('solver',solver),
                         ('run_crossover','off'),('primal_feasibility_tolerance',1e-9),
                         ('dual_feasibility_tolerance',1e-9)):
            if h.setOptionValue(name,val)==hs.HighsStatus.kError:raise RuntimeError('Unsupported HiGHS option '+name)
        model=hs.HighsLp();model.num_col_=self.n;model.num_row_=self.m
        model.col_cost_=np.asarray(f,dtype=float);model.col_lower_=np.asarray(lo,dtype=float)
        model.col_upper_=np.asarray(hi,dtype=float)
        model.row_lower_=np.full(self.m,-hs.kHighsInf);model.row_upper_=np.asarray(b,dtype=float)
        model.a_matrix_.format_=hs.MatrixFormat.kRowwise
        model.a_matrix_.start_=A.indptr;model.a_matrix_.index_=A.indices;model.a_matrix_.value_=A.data
        if h.passModel(model)==hs.HighsStatus.kError:raise RuntimeError('HiGHS rejected LP')

    def add_rows(self,A,b):
        import highspy as hs
        A=sparse.csr_matrix(A)
        if not A.shape[0]:return
        status=self.h.addRows(A.shape[0],np.full(A.shape[0],-hs.kHighsInf),np.asarray(b,dtype=float),
            A.nnz,A.indptr.astype(np.int32),A.indices.astype(np.int32),A.data)
        if status==hs.HighsStatus.kError:raise RuntimeError('HiGHS rejected new constraints')
        self.m+=A.shape[0]

    def set_primal(self,x):
        import highspy as hs
        status=self.h.setSolution(self.n,np.arange(self.n,dtype=np.int32),np.asarray(x,dtype=float))
        if status==hs.HighsStatus.kError:raise RuntimeError('HiGHS rejected the initial primal proposal')

    def solve(self,f,time_limit=120,progress=QUIET,label='HiGHS'):
        import highspy as hs
        h=self.h
        if time_limit<=0:raise ValueError('Positive LP time budget required')
        h.changeColsCost(self.n,np.arange(self.n,dtype=np.int32),np.asarray(f,dtype=float))
        h.setOptionValue('time_limit',h.getRunTime()+float(time_limit))
        with progress.phase(label,variables=self.n,constraints=self.m,solver='HiGHS '+h.version()):h.run()
        status=h.getModelStatus();solution=h.getSolution();info=h.getInfo()
        x=np.asarray(solution.col_value,dtype=float);dual=np.asarray(solution.row_dual,dtype=float)
        primal_valid=bool(solution.value_valid) and len(x)==len(f) and np.all(np.isfinite(x))
        dual_finite=len(dual)==self.m and np.all(np.isfinite(dual))
        return SimpleNamespace(success=status==hs.HighsModelStatus.kOptimal,
            status=str(status),message=h.modelStatusToString(status),
            x=x if primal_valid else None,fun=float(np.asarray(f)@x) if primal_valid else 0.,
            ineqlin=SimpleNamespace(marginals=dual if dual_finite else None),
            nit=int(info.simplex_iteration_count+info.ipm_iteration_count),
            dual_valid=bool(solution.dual_valid),dual_finite=dual_finite)


def optimize(f,A,b,lo,hi,time_limit=120,solver='ipm',progress=QUIET,label='HiGHS'):
    return WarmLP(f,A,b,lo,hi,solver).solve(f,time_limit,progress,label)
