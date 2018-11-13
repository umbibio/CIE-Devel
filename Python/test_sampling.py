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

model.sample(N=100, njobs=1, quiet=True)

