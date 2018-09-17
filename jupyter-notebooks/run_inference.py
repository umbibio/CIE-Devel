import numpy as np
import scipy.stats as st
from infer.aux import genData, Reporter, mutate_data
from infer.model import Model
import pickle

Xgt, Y, rels = genData(60, 10, 20000)

# this is for adding noise to input data
Y = mutate_data(Y, 0.05)

njobs = 2
chains = 2

model = Model()
model.scale = [0.5]
model.build(Y, rels)

model.set_subsample_size(0.0001)

alph, beta = 0.4, 0.7
x_alph = np.ones(shape=model.nx)*alph
x_beta = np.ones(shape=model.nx)*beta

r_alph = np.ones(shape=model.ns)*alph
r_beta = np.ones(shape=model.ns)*beta

alph, beta = 0.5, 0.5
s_alph = np.ones(shape=model.ns)*alph
s_beta = np.ones(shape=model.ns)*beta

Xprior = st.beta(x_alph, x_beta)
Rprior = st.beta(r_alph, r_beta)
Sprior = st.beta(s_alph, s_beta)

model.set_priors(Xprior, Rprior, Sprior)
model.init_chains(chains=chains)

reporter = Reporter()
reporter.report("Start")
while not model.converged():
    model.sample(N=2000, burn=0.1, thin=100, njobs=njobs)
    reporter.report("Completed current sample")

pickle.dump( model, open( "model.p", "wb" ) )
