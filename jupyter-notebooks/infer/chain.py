import numpy as np
import scipy.stats as st

class Chain(object):
    def __init__(self, model, chain_id):
        
        self.varnames = model.varnames
        self.vars = model.vars
        
        self.nx = model.nx
        self.ns = model.ns
        self.ny = model.ny
        
        self.id = chain_id
        
        self.scale = model.scale[0]
        
        self.ny = model.ny
        self.ns = model.ns
        self.nx = model.nx
        
        self.map = model.map
        self.YY = model.YY
        
        self.chain = {}
        
        self.prev_loglikelihood = np.zeros(shape=self.ny) - np.inf


    def accept(self, var, slce):

        affected = set()
        for i in slce:
            affected.update(self.map[var.name][i])

        mask =  np.array(list(affected))

        t = np.ones(shape=self.ny, dtype=np.float64)
        u = np.ones(shape=self.ny, dtype=np.float64)
        
        RS = self.vars['S'].value * self.vars['R'].value
        RnS = (1. - self.vars['S'].value) * self.vars['R'].value
            
        for j in mask:
            for i, k in self.map['Y'][j]:
                t[j] *= 1. - self.vars['X'].value[i] * RS[k]
                u[j] *= 1. - self.vars['X'].value[i] * RnS[k] 

        p0 = 1. - u[mask]
        p1 = t[mask]*u[mask]
        p2 = u[mask] - p1

        pp = np.stack([p0, p1, p2], axis=1)

        loglikelihood = self.prev_loglikelihood.copy()
        loglikelihood[mask] = st.multinomial(1, pp).logpmf(self.YY[mask])
        
        logratio = loglikelihood[mask].sum() - self.prev_loglikelihood[mask].sum()
        logratio += var.lgpdf[slce].sum() - var.prev_lgpdf[slce].sum()

        accept = logratio >= 0 or logratio > -np.random.exponential()

        if accept:
            self.prev_loglikelihood = loglikelihood

        return accept


    def sample(self, N, total_sampled=None, thin=1):

        steps_until_thin = thin
        
        for variable in self.vars.values():
            self.chain[variable.name] = np.zeros(shape=variable.size)
        
        np.random.seed()
                
        updt_interval = max(1, N*0.0001)
        steps_until_updt = updt_interval

        tune_interval = updt_interval * 30
        steps_until_tune = tune_interval
        acc_rate = 0
        accepted = rejected = 0
        total_accepted = total_rejected = 0
        
        for i in range(N):
            steps_until_updt -= 1
            if not steps_until_updt:
                if total_sampled is not None:
                    total_sampled[self.id] += updt_interval
                else:
                    print("\rChain {} - Acceptance rate {: 7.2%}, ".format(self.id, acc_rate), end="")
                    print("Progress {: 7.2%}".format(i/N), end="")
                steps_until_updt = updt_interval

            steps_until_tune -= 1
            if not steps_until_tune:
                acc_rate = accepted/(accepted + rejected)

                #tune()

                steps_until_tune = tune_interval
                accepted = rejected = 0
            
            for var in self.vars.values():
                slce = var.mutate()

                if not self.accept(var, slce):
                    var.revert()
                    rejected += 1
                    total_rejected += 1
                else:
                    accepted += 1
                    total_accepted += 1
            
            steps_until_thin -= 1
            if not steps_until_thin:
                steps_until_thin = thin
                for var in self.vars.values():
                    self.chain[var.name] = np.vstack([self.chain[var.name], var.value])
            
        for variable in self.vars.values():
            self.chain[variable.name] = self.chain[variable.name][1:]

        acc_rate = total_accepted/(total_accepted + total_rejected)
        print("\rChain {} - Acceptance rate {: 7.2%}, ".format(self.id, acc_rate), end="")
        print("Sampling completed")
        if total_sampled is not None:
            total_sampled[self.id] = N
        
        return self
        