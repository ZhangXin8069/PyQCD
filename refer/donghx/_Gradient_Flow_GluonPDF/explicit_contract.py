"""Shape-keyed immutable opt_einsum paths; no numerical-result caching."""
from time import perf_counter
import opt_einsum as oe


class ExplicitContractor:
    def __init__(self, optimize='dp'):
        if optimize not in ('dp','optimal','greedy'):
            raise ValueError(optimize)
        self.optimize=optimize
        self.expressions={}
        self.plans={}
        self.cache_hits=0

    def __call__(self,equation,*arrays):
        shapes=tuple(tuple(int(n) for n in array.shape) for array in arrays)
        key=(equation,shapes,self.optimize)
        if key not in self.expressions:
            started=perf_counter()
            path,info=oe.contract_path(equation,*shapes,shapes=True,optimize=self.optimize)
            path=tuple(tuple(int(i) for i in step) for step in path)
            expression=oe.contract_expression(equation,*shapes,optimize=path)
            self.plans[key]=dict(equation=equation,shapes=shapes,optimizer=self.optimize,
                path=path,opt_cost=float(info.opt_cost),largest_intermediate=int(info.largest_intermediate),
                planning_seconds=perf_counter()-started)
            self.expressions[key]=expression
        else:
            self.cache_hits+=1
        return self.expressions[key](*arrays)

    def metadata(self):
        return dict(optimizer=self.optimize,opt_einsum_version=oe.__version__,
            cache_hits=self.cache_hits,expressions=len(self.expressions),
            planning_seconds=sum(x['planning_seconds'] for x in self.plans.values()),
            plans=list(self.plans.values()))
