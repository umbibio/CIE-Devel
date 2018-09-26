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
        self.acc_rate = 0
        
        self.ny = model.ny
        self.ns = model.ns
        self.nx = model.nx
        
        self.map = model.map
        self.YY = model.YY
        
        self.chain = {}
        
        self.prev_loglikelihood = np.zeros(shape=self.ny) - np.inf

        self.t = np.ones(shape=self.ny, dtype=np.float64)
        self.u = np.ones(shape=self.ny, dtype=np.float64)
        self.RS = self.vars['S'].value * self.vars['R'].value
        self.RnS = self.vars['R'].value - self.RS


    def accept(self, var):
        
        slce = var.slce

        Ys_affected = set()
        Ys_affected_by = {}
        for i in slce:
            lst = self.map[var.name][i]['Y']
            Ys_affected_by[i] = np.array(lst)
            Ys_affected.update(lst)

        Y_mask =  np.array(list(Ys_affected))

        self.RS = self.vars['S'].value * self.vars['R'].value
        self.RnS = self.vars['R'].value - self.RS

        self.t = np.ones(shape=self.ny, dtype=np.float64)
        self.u = np.ones(shape=self.ny, dtype=np.float64)

        for j in Y_mask:
            for i, k in self.map['Y'][j]['XS']:
                self.t[j] *= 1. - self.vars['X'].value[i] * self.RS[k]
                self.u[j] *= 1. - self.vars['X'].value[i] * self.RnS[k] 

        p0 = 1. - self.u[Y_mask]
        p1 = self.t[Y_mask]*self.u[Y_mask]
        p2 = self.u[Y_mask] - p1

        pp = np.stack([p0, p1, p2], axis=1)

        loglikelihood = self.prev_loglikelihood.copy()
        loglikelihood[Y_mask] = st.multinomial(1, pp).logpmf(self.YY[Y_mask])

        logratio_var = np.zeros(var.size) - np.inf
        logratio_lik = np.zeros(self.ny) - np.inf
        logratio_var[slce] = var.lgpdf[slce] - var.prev_lgpdf[slce]
        logratio_lik[Y_mask] = loglikelihood[Y_mask] - self.prev_loglikelihood[Y_mask]

        logratio = logratio_var
        for i in slce:
            # TODO: find better way to fill this values
            logratio[i] += logratio_lik[Ys_affected_by[i]].sum()

        slce_mask = np.zeros(var.size, dtype=bool)
        slce_mask[slce] = 1

        threshold = - np.random.exponential(size=var.size)

        accept = (logratio >= 0) + (logratio > threshold)
        reject_mask = (~accept)*slce_mask
        var.value[reject_mask] = var.prev_value[reject_mask]
        var.lgpdf[reject_mask] = var.prev_lgpdf[reject_mask]
        var.n_rejected[reject_mask] += 1
        
        slce_mask = accept * slce_mask
        slce = np.argwhere(slce_mask).T[0]
        var.slce = slce

        if slce.shape[0] == 0:
            return False

        Ys_affected = set()
        for i in slce:
            Ys_affected.update(self.map[var.name][i]['Y'])

        Y_mask =  np.array(list(Ys_affected))

        logratio = loglikelihood[Y_mask].sum() - self.prev_loglikelihood[Y_mask].sum()
        logratio += var.lgpdf[slce].sum() - var.prev_lgpdf[slce].sum()

        accept = logratio >= 0 or logratio > - np.random.exponential()

        if accept:
            self.prev_loglikelihood[Y_mask] = loglikelihood[Y_mask]

        return accept


    def sample(self, N, total_sampled=None, thin=1):

        steps_until_thin = thin
        
        for variable in self.vars.values():
            self.chain[variable.name] = np.zeros(shape=variable.size)
        
        np.random.seed()
                
        updt_interval = max(1, N*0.0001)
        steps_until_updt = updt_interval

        tune_interval = 100 #updt_interval * 30
        steps_until_tune = tune_interval
        self.acc_rate = 0
        accepted = rejected = 0
        total_accepted = total_rejected = 0
        
        for i in range(N):
            steps_until_updt -= 1
            if not steps_until_updt:
                if total_sampled is not None:
                    total_sampled[self.id] += updt_interval
                else:
                    print("\rChain {} - Acceptance rate {: 7.2%}, ".format(self.id, self.acc_rate), end="")
                    print("Progress {: 7.2%}".format(i/N), end="")
                steps_until_updt = updt_interval

            steps_until_tune -= 1
            if not steps_until_tune:
                self.acc_rate = accepted/(accepted + rejected)

                #self.tune()
                for var in self.vars.values():
                #    var.scale = self.scale
                    var.tune()

                steps_until_tune = tune_interval
                accepted = rejected = 0
            
            for var in self.vars.values():
                var.mutate()

                if not self.accept(var):
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

        self.acc_rate = total_accepted/(total_accepted + total_rejected)
        print("\rChain {} - Acceptance rate {: 7.2%}, ".format(self.id, self.acc_rate), end="")
        print("Sampling completed")
        if total_sampled is not None:
            total_sampled[self.id] = N
        
        return self

    def tune(self):
        # Switch statement
        if self.acc_rate < 0.001:
            # reduce by 90 percent
            self.scale *= 0.1
        elif self.acc_rate < 0.05:
            # reduce by 50 percent
            self.scale *= 0.5
        elif self.acc_rate < 0.2:
            # reduce by ten percent
            self.scale *= 0.9
        elif self.acc_rate > 0.95:
            # increase by factor of ten
            self.scale *= 10.0
        elif self.acc_rate > 0.75:
            # increase by double
            self.scale *= 2.0
        elif self.acc_rate > 0.5:
            # increase by ten percent
            self.scale *= 1.1

        self.scale = min(self.scale, 0.5)
        self.scale = max(self.scale, 0.01)
