import signal, time
import numpy as np
import pandas as pd
import scipy.stats as st
from multiprocessing import Pool, Manager

class Model(object):
        
    def __init__(self):
        
        self.edgeMap = None
        self.nx = None
        self.ny = None
        self.ns = None
        self.dictionaries = None
        
        self.scale = [0.5]
        self.trace = []
        
        self.burn = None
        self.gelman_rubin = {}
        self.max_gr = 0
    
    def build(self, Y, rels):
        # create several dictionaries for mapping uids to a range starting from zero
        # this is for better performance by making a single function call
        # to create the distributions in pymc3
        xs = rels['srcuid'].unique()
        ys = Y.keys()
        Dx = dict(zip(xs, range(len(xs))))
        Dy = dict(zip(ys, range(len(ys))))
        
        self.nx = len(Dx)
        self.ny = len(Dy)

        # this is the inverse of the dictionary
        # it simply goes in the reverse direction
        ADx = {}
        for src, i in Dx.items():
            ADx[i] = src

        Ds = {}
        ADs = {}
        for k, (src, trg) in enumerate(rels.index):
            Ds[(src, trg)] = k
            ADs[k] = src, trg

        self.ns = len(Ds)
        
        self.edgeMap = np.zeros((self.nx, self.ns, self.ny), dtype=np.int8)
        for k, (src, trg) in enumerate(rels.index):
            self.edgeMap[Dx[src], Ds[(src, trg)], Dy[trg]] = 1

        self.dictionaries = Dx, ADx, Ds, ADs, Dy
        
        self.YY = []
        for trg, j in Dy.items():
            tmp = [0, 0, 0]
            tmp[1 + Y[trg]] = 1
            self.YY.append(tmp)

    def set_priors(self, Xprior, Rprior, Sprior):
        self.Xprior = Xprior
        self.Rprior = Rprior
        self.Sprior = Sprior
        
    def init_chains(self, chains=2):
        for i in range(chains):
            self.trace.append(Chain(self, i))
            
    
    def sample(self, N=1000, burn=500, njobs=2):
        self.burn = burn
        if njobs > 1:
            
            chains = len(self.trace)
            
            print(f"\nSampling {chains} chains in {njobs} jobs")
        
            # Want workers to ignore Keyboard interrupt
            original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
            # Create the pool of workers
            pool = Pool(processes=njobs)
            # restore signals for the main process
            signal.signal(signal.SIGINT, original_sigint_handler)
            
            try:
                manager = Manager()
                sampled = manager.list([0]*chains)
                mres = [pool.apply_async(chain.sample, (N, sampled)) for chain in self.trace]
                pool.close()
                
                tmp = 3600 * 4 # one hour timeout
                target_total = N * chains
                while tmp:
                    time.sleep(1/4)
                    total_sampled = 0
                    for count in sampled:
                        total_sampled += count
                    progress = total_sampled / target_total
                    print("\rProgress {: 7.2%}".format(progress), end="")
                    if progress == 1:
                        break
                    tmp -= 1
                    
                if tmp <= 0:
                    raise TimeoutError

                self.trace = [res.get(timeout=3600) for res in mres]
            except KeyboardInterrupt:
                pool.terminate()
                print("\n\nCaught KeyboardInterrupt, workers have been terminated\n")
                timeseries, dataseries = [], []
                raise SystemExit

            except TimeoutError:
                pool.terminate()
                print("\n\nThe workers ran out of time. Terminating simulation.\n")
                raise SystemExit
            
            pool.join()
        else:
            for chain in self.trace:
                chain.sample(N)
            
        self.run_convergence_test()
    
    def rscore(self, x, num_samples):
        """Implementation taken from
        https://github.com/pymc-devs/pymc3/blob/master/pymc3/diagnostics.py
        
        Further reference:
        https://blog.stata.com/2016/05/26/gelman-rubin-convergence-diagnostic-using-multiple-chains/
        DOI: 10.1080/10618600.1998.10474787
        https://www.jstor.org/stable/2246093
        """
        # Calculate between-chain variance
        B = num_samples * np.var(np.mean(x, axis=1), axis=0, ddof=1)
        
        # Calculate within-chain variance
        W = np.mean(np.var(x, axis=1, ddof=1), axis=0)

        # Estimate of marginal posterior variance
        Vhat = W * (num_samples - 1) / num_samples + B / num_samples
        
        return np.sqrt(Vhat / W)
        
    def get_trace(self, varname, chain=None, combine=False):
        burn = self.burn
        trace = []
        for t in self.trace:
            trace.append(t.chain[varname][burn:, :])
        trace = np.array(trace)
        
        if combine:
            nchains, nsample, nvars = trace.shape
            return trace.reshape((-1, nvars))
        
        if chain is not None:
            return trace[chain]
        
        return trace
        
    def run_convergence_test(self):
        x = self.get_trace('X')
        self.gelman_rubin['X'] = self.rscore(x, x.shape[1])
        
        r = self.get_trace('R')
        self.gelman_rubin['R'] = self.rscore(r, r.shape[1])
        
        s = self.get_trace('S')
        self.gelman_rubin['S'] = self.rscore(s, s.shape[1])
        
    def converged(self):
        max_gr = 0
        for var, gr in self.gelman_rubin.items():
            max_gr = max(max_gr, gr.max())
        self.max_gr = max_gr
        
        if max_gr < 1.1:
            print("\nChains have converged")
            return True
        else:
            print(f"\nFailed to converge. Gelman-Rubin statistics was {max_gr: 7.4} for some parameter")
            return False

    def get_result(self, varname):
        trace = self.get_trace(varname, combine=True)
        mean = trace.mean(axis=0)
        std = trace.std(axis=0)
        return pd.DataFrame({'mean': mean, 'std': std})


class Chain(object):
    def __init__(self, model, chain_id):
        
        self.id = chain_id
        
        self.scale = model.scale[0]
        
        self.nx = model.nx
        self.ns = model.ns
        
        self.edgeMap = model.edgeMap
        self.YY = model.YY
        
        self.Xprior = model.Xprior
        self.Rprior = model.Rprior
        self.Sprior = model.Sprior
        
        self.X = self.Xprior.rvs()
        self.R = self.Rprior.rvs()
        self.S = self.Sprior.rvs()
        
        self.chain = {}
        self.chain['X'] = np.zeros(shape=(self.nx))
        self.chain['R'] = np.zeros(shape=(self.ns))
        self.chain['S'] = np.zeros(shape=(self.ns))
        
        self.logpp = - np.inf
        self.accepted = 0
        self.rejected = 0
        
    def accept(self):
        
        RS = self.Rtensor * self.Stensor
        RnS = self.Rtensor * (1 - self.Stensor)
        t = 1 - (1 - self.edgeMap * self.Xtensor * RS).prod(axis=0).prod(axis=0)
        u = 1 - (1 - self.edgeMap * self.Xtensor * RnS).prod(axis=0).prod(axis=0)

        p0 = u#/2
        p2 = t#/2
        p1 = 1 - p0 - p2

        pp = np.stack([p0, p1, p2], axis=1)

        loglikelihood = st.multinomial(1, pp).logpmf(self.YY).sum()

        logposterior = loglikelihood
        logposterior += self.Xprior.logpdf(self.X).sum()
        logposterior += self.Rprior.logpdf(self.R).sum()
        logposterior += self.Sprior.logpdf(self.S).sum()

        logratio = logposterior - self.logpp

        accept = logratio >= 0 or logratio > -np.random.exponential()

        if accept:
            self.logpp = logposterior

            self.accepted += 1
        else:
            self.rejected += 1

        return accept

    def a_sample_from(self, var, size=None):
        if not size:
            size = 1

        prev = {}
        for i in np.random.choice(list(range(len(var))), size=size, replace=False):
            prev[i] = var[i]
            a, b = (0.001 - prev[i]) / self.scale, (0.999 - prev[i]) / self.scale
            var[i] = st.truncnorm(a, b, prev[i], self.scale).rvs()
        if not self.accept():
            for i, val in prev.items():
                var[i] = val
    
    def sample(self, N, total_sampled=None):
        
        np.random.seed()
        
        self.Xtensor = self.X.view()
        self.Rtensor = self.R.view()
        self.Stensor = self.S.view()
        self.Xtensor.shape = (self.nx, 1, 1)
        self.Rtensor.shape = (1, self.ns, 1)
        self.Stensor.shape = (1, self.ns, 1)
        
        tune_interval = 43
        steps_until_tune = tune_interval
        
        s_sample_size = int(self.ns/10)
        
        for i in range(N):
            if not steps_until_tune:
                acc_rate = self.accepted/(self.accepted + self.rejected)
                print("\rChain {} - Acceptance rate {: 7.2%}, ".format(self.id, acc_rate), end="")
                print("Progress {: 7.2%}".format(i/N), end="")

                #tune()

                steps_until_tune = tune_interval
                self.accepted = 0
                self.rejected = 0
                if total_sampled is not None:
                    total_sampled[self.id] += tune_interval

            self.a_sample_from(self.X, 1)
            self.a_sample_from(self.R, s_sample_size)
            self.a_sample_from(self.S, s_sample_size)
            
            self.chain['X'] = np.vstack([self.chain['X'], self.X])
            self.chain['R'] = np.vstack([self.chain['R'], self.R])
            self.chain['S'] = np.vstack([self.chain['S'], self.S])

            steps_until_tune -= 1

        print("\rChain {} - Acceptance rate {: 7.2%}, ".format(self.id, acc_rate), end="")
        print("Sampling completed")
        if total_sampled is not None:
            total_sampled[self.id] = N
        
        return self
        