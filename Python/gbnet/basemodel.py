import signal, time
import numpy as np
import pandas as pd
from multiprocessing import Pool, Manager
from gbnet.aux import Reporter

class BaseModel(object):

    __slots__ = [
        'trace',
        'burn',
        'gelman_rubin',
        'max_gr',
        'vars',
        '_trace_keys',
        'rp',
        'rels',
        'DEG',
        'nchains',
        'manager',
        'stats',
    ]

    def __init__(self, rels, DEG, nchains=2):

        self.rels = rels
        self.DEG = DEG
        self.nchains = nchains
        
        self.gelman_rubin = {}
        
        self.manager = Manager()
        self.vars = self.manager.list()
        self.stats = self.manager.list()

        self._trace_keys = None
        self.rp = Reporter()


        for ch in range(nchains):
            variables = self.build_variables(rels, DEG)
            self.vars.append(variables)
        
        stats = {}
        for key in self.trace_keys:
            stats[key] = { 'sum1': 0, 'sum2': 0, 'N': 0 }

        for ch in range(nchains):
            statistics = stats.copy()
            self.stats.append(statistics)

    def build_variables(self, rels, DEG):
        return {}

    def burn_stats(self):
        for ch in range(self.nchains):
            for key in self.trace_keys:
                self.stats[ch][key] = { 'sum1': 0, 'sum2': 0, 'N': 0 }


    def get_trace_stats(self, combine=False):

        dfs = []
        for ch in range(self.nchains):
            df = pd.DataFrame(self.stats[ch]).transpose()
            df.index.name = 'name'
            dfs.append(df)

        num_samples = dfs[0]['N'][0]

        if combine:
            stats = dfs[0]
            for df in dfs[1:]:
                stats += df

            if num_samples > 0:

                stats = stats.assign(mean=stats.apply(lambda r: r.sum1/r.N, axis=1))
                stats = stats.assign(var=stats.apply(lambda r: r.sum2/r.N - r['mean']**2, axis=1))
                stats = stats.assign(std=stats.apply(lambda r: np.sqrt(r['var']), axis=1))

        else:
            stats = []
            for df in dfs:
                if num_samples > 0:
                    df = df.assign(mean=df.apply(lambda r: r.sum1/r.N, axis=1))
                    df = df.assign(var=df.apply(lambda r: r.sum2/r.N - r['mean']**2, axis=1))
                    df = df.assign(std=df.apply(lambda r: np.sqrt(r['var']), axis=1))
                stats.append(df)

        return stats


    @property
    def trace_keys(self):
        if self._trace_keys is None:
            self._trace_keys = []
            for vardict in self.vars[0].values():
                for node in vardict.values():
                    try:
                        # if node is multinomial, value will be a numpy array
                        # have to set a list for each element in 'value'
                        for i in range(len(node.value)):
                            self._trace_keys.append(f"{node.id}_{i}")
                    except TypeError:
                        # value is no array, it won't have Attribute 'size'
                        self._trace_keys.append(node.id)
        return self._trace_keys


    def get_gelman_rubin(self):
        
        if self.nchains < 2:
            print('Need at least two chains for the convergence test')
            return
        
        trace_stats = self.get_trace_stats()

        num_samples = trace_stats[0]['N'][0]

        if num_samples == 0:
            return []

        # Calculate between-chain variance
        B = num_samples * pd.DataFrame([df['mean'] for df in trace_stats]).var(ddof=1)

        # Calculate within-chain variance
        W = pd.DataFrame([df['var'] for df in trace_stats]).mean()

        # Estimate of marginal posterior variance
        Vhat = W * (num_samples - 1) / num_samples + B / num_samples

        var_table = pd.DataFrame({'W':W, 'Vhat':Vhat})

        gelman_rubin = var_table.apply(lambda r: np.sqrt(r.Vhat/r.W) if r.W > 0 else 1., axis=1)
        self.gelman_rubin = gelman_rubin

        return gelman_rubin


    def converged(self):

        gelman_rubin = self.get_gelman_rubin()

        if len(gelman_rubin) == 0:
            return False

        max_gr = gelman_rubin.max()
        
        if max_gr < 1.1:
            print("\nChains have converged")
            return True
        else:
            print(f"\nFailed to converge. "
                  f"Gelman-Rubin statistics was {max_gr: 7.4} for some parameter")
            return False


    def sample(self, N=200, thin=1, njobs=2):
            
        if njobs > 1:
            
            chains = self.nchains
            
            print(f"\nSampling {chains} chains in {njobs} jobs")
        
            # Want workers to ignore Keyboard interrupt
            original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
            # Create the pool of workers
            pool = Pool(processes=njobs)
            # restore signals for the main process
            signal.signal(signal.SIGINT, original_sigint_handler)
            
            try:
                #manager = Manager()
                sampled = self.manager.list([0]*chains)
                for ch in range(self.nchains):
                    pool.apply_async(sample_chain, (N, self.vars, self.stats, ch, sampled, thin))
                pool.close()
                
                timer = 90 * 24 * 60 * 60 * 4 # 90 days timeout
                target_total = N * chains
                while timer:
                    time.sleep(1/4)
                    total_sampled = 0
                    for count in sampled:
                        total_sampled += count
                    progress = total_sampled / target_total
                    self.rp.report(f"Progress {progress: 7.2%}", schar='\r', lchar='', showlast=False)
                    if progress == 1:
                        break
                    timer -= 1
                self.rp.report(f"Progress {progress: 7.2%}", schar='\r')
                
                if timer <= 0:
                    raise TimeoutError

                #self.chains = [res.get(timeout=3600) for res in mres]
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
            for ch in range(self.nchains):
                sample_chain(N, self.vars, self.stats, ch, thin=thin)


def sample_chain(N, modelvars, modelstats, chain_id, run_sampled_count=None, thin=1):

    mvars = modelvars[chain_id]
    mstats = modelstats[chain_id]

    steps_until_thin = thin

    # this will run in multiprocessing job, so we need to reset seed
    np.random.seed()
            
    updt_interval = max(1, N*0.0001)
    steps_until_updt = updt_interval

    for i in range(N):
        steps_until_updt -= 1
        if not steps_until_updt:
            if run_sampled_count is not None:
                run_sampled_count[chain_id] += updt_interval
            else:
                print("\rChain {} - Progress {: 7.2%}".format(chain_id, i/N), end="")
            steps_until_updt = updt_interval

        for vardict in mvars.values():
            for node in vardict.values():
                node.sample()

        steps_until_thin -= 1
        if not steps_until_thin:
            steps_until_thin = thin

            for vardict in mvars.values():
                for node in vardict.values():
                    try:
                        # if node is multinomial, value will be a numpy array
                        # have to set a list for each element in 'value'
                        for i, val in enumerate(node.value):
                            mstats[f"{node.id}_{i}"]['sum1'] += val
                            mstats[f"{node.id}_{i}"]['sum2'] += val**2
                            mstats[f"{node.id}_{i}"]['N'] += 1
                    except TypeError:
                        # value is no array, it won't be iterable
                        mstats[node.id]['sum1'] += node.value
                        mstats[node.id]['sum2'] += node.value**2
                        mstats[node.id]['N'] += 1

    modelvars[chain_id] = mvars
    modelstats[chain_id] = mstats

    print(f"\rChain {chain_id} - Sampling completed")
    if run_sampled_count is not None:
        run_sampled_count[chain_id] = N

