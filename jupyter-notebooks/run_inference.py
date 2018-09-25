import pickle
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from infer.aux import genData, processTrace, updateRes, Reporter, mutate_data, mutate_data2
from infer.model import Model

Xgt, Y, rels = genData(2, 1, 50)

# this is for adding noise to input data
#Y = mutate_data2(Y, 0.05)

alph, beta = 0.5, 1
x_alph = np.ones(shape=len(Xgt))*alph
x_beta = np.ones(shape=len(Xgt))*beta

r_alph = np.ones(shape=len(rels))*alph
r_beta = np.ones(shape=len(rels))*beta

alph = beta = 0.5
s_alph = np.ones(shape=len(rels))*alph
s_beta = np.ones(shape=len(rels))*beta

Xprior = st.beta(x_alph, x_beta)
Rprior = st.beta(r_alph, r_beta)
Sprior = st.beta(s_alph, s_beta)

njobs = 1
chains = 2

model = Model()
model.scale = [0.5]
model.build(Y, rels, [Xprior, Rprior, Sprior])

model.set_subsample_size(0.1)

model.init_chains(chains=chains)

reporter = Reporter()
reporter.report("Start")
while not model.converged():
    model.sample(N=2000, burn=0.5, thin=10, njobs=njobs)
    reporter.report("Completed current sample")

pickle.dump( model, open( "model.p", "wb" ) )
