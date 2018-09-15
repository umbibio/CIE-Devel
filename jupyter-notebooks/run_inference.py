import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from infer.aux import genData, processTrace, updateRes, Reporter, mutate_data
from infer.model import Model

Xgt, Y, rels = genData(60, 10, 50)

# this is for adding noise to input data
#Y = mutate_data(Y, 1)

njobs = 1
chains = 1

model = Model()
model.scale = [0.5]
model.build(Y, rels)

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
model.sample(N=2000, burn=0.0, thin=1, njobs=njobs)
reporter.report("Completed current sample")
while not model.converged():
    model.sample(N=2000, burn=0.0, thin=1, njobs=njobs)
    reporter.report("Completed current sample")

model.trace[0]['X'].shape

y = model.get_trace('X', chain=0)
for i in range(y.shape[1]):
    plt.plot(range(y.shape[0]), y[:, i], alpha=0.4)
plt.ylim(0,1)
plt.show()

Xres, Rres, Sres = processTrace(model, Xgt, rels)
Xres, Rres, Sres = updateRes(Xres, Rres, Sres, final=True)

Xres

Rres

Sres