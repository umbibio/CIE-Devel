import pickle
from gbnet.aux import genData
from gbnet.models import ORNORModel

NX, NActvX, NY = 10, 3, 200
Xgt, DEG, rels = genData(NX, NActvX, NY)
model = ORNORModel(rels, DEG)

while not model.converged():
    model.sample(N=500, burn=0.5, njobs=2)

with open('results.p', 'wb') as file:
    pickle.dump(model.result(Xgt=Xgt), file)
