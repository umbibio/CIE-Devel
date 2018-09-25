import signal, time
import numpy as np
import pandas as pd
from multiprocessing import Pool, Manager

from infer.variable import Variable
from infer.chain import Chain

class Model(object):
        
    def __init__(self):
        
        self.edgeMap = None
        self.dictionaries = None
        
        self.scale = [0.5]
        self.trace = []
        self.chains = []
        
        self.burn = None
        self.gelman_rubin = {}
        self.max_gr = 0
        
        self.vars = {}
        self.varnames = None
    
    def build(self, Y, rels, priors):
        # create several dictionaries for mapping uids to a range starting from zero
        # this is for better performance by making a single function call
        # to create the distributions in pymc3
        xs = rels['srcuid'].unique()
        ys = Y.keys()
        Dx = dict(zip(xs, range(len(xs))))
        Dy = dict(zip(ys, range(len(ys))))
        
        nx = len(Dx)
        self.nx = nx
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

        ns = len(Ds)
        self.ns = ns
        

        self.map = {}
        self.map['X'] = {}
        for i in Dx.values():
            self.map['X'][i] = {} 
            self.map['X'][i]['Y'] = []
            self.map['X'][i]['R'] = []
            self.map['X'][i]['S'] = []
        for (src, trg), k in Ds.items():
            self.map['X'][Dx[src]]['Y'].append(Dy[trg])
            self.map['X'][Dx[src]]['R'].append(k)
            self.map['X'][Dx[src]]['S'].append(k)

        self.map['Y'] = {}
        for j in Dy.values():
            self.map['Y'][j] = {}
            self.map['Y'][j]['XS'] = []
            self.map['Y'][j]['S'] = []
        for (src, trg), k in Ds.items():
            self.map['Y'][Dy[trg]]['XS'].append((Dx[src], k))
            self.map['Y'][Dy[trg]]['S'].append(k)

        self.map['S'] = {}
        for (src, trg), k in Ds.items():
            self.map['S'][k] = {}
            self.map['S'][k]['Y'] = [Dy[trg]]

        self.map['R'] = self.map['S']
        
        self.dictionaries = Dx, ADx, Ds, ADs, Dy

        self.YY = []
        for trg in Dy.keys():
            tmp = [0, 0, 0]
            tmp[1 + Y[trg]] = 1
            self.YY.append(tmp)

        self.YY = np.array(self.YY)

        self.varnames = ['X', 'R', 'S']

        for varname, prior in zip(self.varnames, priors):
            self.vars[varname] = Variable(varname, prior, self.scale[0])

        
    def init_chains(self, chains=2):
        for i in range(chains):
            self.chains.append(Chain(self, i))
            self.trace.append(dict(zip(self.varnames, [None]*len(self.varnames))))
            for variable in self.vars.values():
                self.trace[i][variable.name] = np.zeros(shape=variable.size)


    def sample(self, N=1000, burn=None, thin=1, njobs=2):
        if burn is None:
            if self.burn is None:
                self.burn = 500
        else:
            self.burn = burn
            
        if njobs > 1:
            
            chains = len(self.chains)
            
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
                mres = [pool.apply_async(chain.sample, (N, sampled, thin)) for chain in self.chains]
                pool.close()
                
                timer = 90 * 24 * 60 * 60 * 4 # 90 days timeout
                target_total = N * chains
                while timer:
                    time.sleep(1/4)
                    total_sampled = 0
                    for count in sampled:
                        total_sampled += count
                    progress = total_sampled / target_total
                    print("\rProgress {: 7.2%}".format(progress), end="")
                    if progress == 1:
                        break
                    timer -= 1
                print("\rProgress {: 7.2%}".format(progress))
                
                if timer <= 0:
                    raise TimeoutError

                self.chains = [res.get(timeout=3600) for res in mres]
            except KeyboardInterrupt:
                pool.terminate()
                print("\n\nCaught KeyboardInterrupt, workers have been terminated\n")
                raise SystemExit

            except TimeoutError:
                pool.terminate()
                print("\n\nThe workers ran out of time. Terminating simulation.\n")
                raise SystemExit
            
            pool.join()
        else:
            for chain in self.chains:
                chain.sample(N, thin=thin)
        
        for i, trace in enumerate(self.trace):
            for varname in self.varnames:
                trace[varname] = np.vstack([trace[varname], self.chains[i].chain[varname]])

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
        trace_length = self.trace[0][varname].shape[0]
        if self.burn < 1:
            burn = int(trace_length * self.burn)
        else:
            burn = self.burn
        trace = []
        for t in self.trace:
            trace.append(t[varname][burn:, :])
        trace = np.array(trace)
        
        if combine:
            nvars = trace.shape[2]
            return trace.reshape((-1, nvars))
        
        if chain is not None:
            return trace[chain]
        
        return trace


    def run_convergence_test(self):
        
        if len(self.trace) < 2:
            print('Need at least two chains for the convergence test')
            return
        
        x = self.get_trace('X')
        self.gelman_rubin['X'] = self.rscore(x, x.shape[1])
        
        r = self.get_trace('R')
        self.gelman_rubin['R'] = self.rscore(r, r.shape[1])
        
        s = self.get_trace('S')
        self.gelman_rubin['S'] = self.rscore(s, s.shape[1])


    def converged(self):

        if len(self.gelman_rubin) == 0:
            return False

        max_gr = 0
        for gr in self.gelman_rubin.values():
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


    def set_subsample_size(self, factor):
        for var in self.vars.values():
            var.sample_size = max(1, int(var.size*factor))
            print(f"{var.name}: new sample_size={var.sample_size}")
        for chain in self.chains:
            for var in chain.vars.values():
                var.sample_size = max(1, int(var.size*factor))
                print(f"chain{chain.id}.{var.name}: new sample_size={var.sample_size}")