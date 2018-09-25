import numpy as np
import scipy.stats as st

class Variable(object):

    def __init__(self, name, prior, scale):
        self.name = name
        self.prior = prior
        self.args = np.array(list(prior.args))
        self.size = self.args.shape[1]

        self.value = np.ones(self.size)/2
        try:
            self.lgpdf = self.prior.logpdf(self.value)
        except AttributeError:
            # in case we try to use a discrete RV with PMF and no PDF
            self.lgpdf = self.prior.logpmf(self.value)
            self.prior.logpdf =self.prior.logpmf

        self.prev_value = None
        self.prev_lgpdf = np.zeros(self.size) - np.inf

        self.sample_size = max(1, int(self.size*0.05))
        self.scale = scale
        
        self.slce = None
        self.n_proposed = np.zeros(self.size)
        self.n_rejected = np.zeros(self.size)
    

    def mutate(self):
        self.prev_value = self.value.copy()
        self.prev_lgpdf = self.lgpdf.copy()

        slce = np.random.choice(self.size, size=self.sample_size, replace=False)
        prev = self.value[slce]

        a, b = (0.01 - prev) / self.scale, (0.99 - prev) / self.scale
        self.value[slce] = st.truncnorm(a, b, prev, self.scale).rvs()
        #self.value[slce] = self.prior.dist(*list(self.args[:, slce])).rvs()
        self.lgpdf = self.prior.logpdf(self.value)
        
        self.n_proposed[slce] += 1

        self.slce = slce


    def revert(self):
        self.value = self.prev_value
        self.lgpdf = self.prev_lgpdf
        self.n_rejected[self.slce] += 1


    def tune(self):

        acc_rate = (self.n_proposed - self.n_rejected)/np.maximum(self.n_proposed, 1)

        mask = (acc_rate < 0.001) & (self.n_proposed > 0)
        # reduce by 90 percent
        self.scale[mask] *= 0.1

        mask = (acc_rate >= 0.001) & (acc_rate <0.05)
        # reduce by 50 percent
        self.scale[mask] *= 0.5

        mask = (acc_rate >= 0.050) & (acc_rate <0.20)
        # reduce by ten percent
        self.scale[mask] *= 0.9

        mask = (acc_rate >= 0.500) & (acc_rate <0.75)
        # increase by ten percent
        self.scale[mask] *= 1.1

        mask = (acc_rate >= 0.750) & (acc_rate <0.95)
        # increase by double
        self.scale[mask] *= 2.0

        mask = (acc_rate >= 0.950)
        # increase by factor of ten
        self.scale[mask] *= 10.0

        self.scale = np.minimum(self.scale, 0.5)
        self.scale = np.maximum(self.scale, 0.05)

        self.n_proposed *= 0
        self.n_rejected *= 0

