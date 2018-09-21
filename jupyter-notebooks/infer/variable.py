import numpy as np
import scipy.stats as st

class Variable(object):
    def __init__(self, name, prior):
        self.name = name
        self.prior = prior
        self.size = prior.args[0].shape[0]

        self.value = np.ones(shape=self.size)/2
        self.lgpdf = self.prior.logpdf(self.value)

        self.prev_value = None
        self.prev_lgpdf = None

        self.scale = None
        self.sample_size = max(1, int(self.size*0.05))
    

    def mutate(self):
        self.prev_value = self.value
        self.prev_lgpdf = self.lgpdf

        slce = np.random.choice(self.size, size=self.sample_size, replace=False)
        prev = self.value[slce]
        a, b = (0.01 - prev) / self.scale, (0.99 - prev) / self.scale
        self.value[slce] = st.truncnorm(a, b, prev, self.scale).rvs()
        self.lgpdf = self.prior.logpdf(self.value)
