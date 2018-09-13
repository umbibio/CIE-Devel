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
        
        self.edgeMap = model.edgeMap
        self.YY = model.YY
        
        self.chain = {}
        
        self.logpp = - np.inf
        self.accepted = 0
        self.rejected = 0
        
    def accept(self):

        t = np.ones(shape=self.ny, dtype=np.float64)
        u = np.ones(shape=self.ny, dtype=np.float64)
        
        RS = self.vars['S'].value * self.vars['R'].value
        RnS = (1. - self.vars['S'].value) * self.vars['R'].value
            
        for i, k, j in self.edgeMap:
            t[j] *= 1. - self.vars['X'].value[i] * RS[k]
            u[j] *= 1. - self.vars['X'].value[i] * RnS[k] 

        t = 1. - t
        u = 1. - u

        p0 = u
        p2 = t
        p1 = 1. - p0 - p2

        pp = np.stack([p0, p1, p2], axis=1)

        loglikelihood = st.multinomial(1, pp).logpmf(self.YY).sum()

        logposterior = loglikelihood
        for variable in self.vars.values():
            logposterior += self.vars[variable.name].prior.logpdf(self.vars[variable.name].value).sum()

        logratio = logposterior - self.logpp

        accept = logratio >= 0 or logratio > -np.random.exponential()

        if accept:
            self.logpp = logposterior

            self.accepted += 1
        else:
            self.rejected += 1

        return accept
    
    def sample(self, N, total_sampled=None, thin=1):

        steps_until_thin = thin
        
        for variable in self.vars.values():
            self.chain[variable.name] = np.zeros(shape=variable.size)
        
        np.random.seed()
                
        updt_interval = 3
        steps_until_updt = updt_interval

        tune_interval = updt_interval * 30
        steps_until_tune = tune_interval
        acc_rate = 0
        
        for i in range(N):
            steps_until_updt -= 1
            if not steps_until_updt:
                print("\rChain {} - Acceptance rate {: 7.2%}, ".format(self.id, acc_rate), end="")
                print("Progress {: 7.2%}".format(i/N), end="")
                if total_sampled is not None:
                    total_sampled[self.id] += updt_interval
                steps_until_updt = updt_interval

            steps_until_tune -= 1
            if not steps_until_tune:
                acc_rate = self.accepted/(self.accepted + self.rejected)
                print("\rChain {} - Acceptance rate {: 7.2%}, ".format(self.id, acc_rate), end="")
                print("Progress {: 7.2%}".format(i/N), end="")

                #tune()

                steps_until_tune = tune_interval
                self.accepted = 0
                self.rejected = 0
            
            for variable in self.vars.values():
                var = variable.value
                sample_size = variable.sample_size

                slce = np.random.choice(list(range(len(var))), size=sample_size, replace=False)
                prev = var[slce]
                a, b = (0.001 - prev) / self.scale, (0.999 - prev) / self.scale
                var[slce] = st.truncnorm(a, b, prev, self.scale).rvs()
                if not self.accept():
                    var[slce] = prev
            
            steps_until_thin -= 1
            if not steps_until_thin:
                steps_until_thin = thin
                for variable in self.vars.values():
                    self.chain[variable.name] = np.vstack([self.chain[variable.name], variable.value])
            
        for variable in self.vars.values():
            self.chain[variable.name] = self.chain[variable.name][1:]

        print("\rChain {} - Acceptance rate {: 7.2%}, ".format(self.id, acc_rate), end="")
        print("Sampling completed")
        if total_sampled is not None:
            total_sampled[self.id] = N
        
        return self
        