import numpy as np
import scipy.stats as st
import pandas as pd
from gbnet.cmodels import ORNORModel
from gbnet.aux import genData

import pstats, cProfile

NX, NActvX, NY = 60, 3, 200
Xgt, DEG, rels = genData(NX, NActvX, NY, AvgNTF=12)
print(len(rels), 'edges in rels')

model = ORNORModel(rels, DEG, nchains=2)
model.sample(N=1, njobs=1, quiet=True)

cProfile.runctx("model.sample(N=100, njobs=1, quiet=True)", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()