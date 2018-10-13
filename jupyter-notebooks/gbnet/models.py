import numpy as np
import pandas as pd
from gbnet.basemodel import BaseModel
from gbnet.nodes import Beta, Multinomial


class ORNOR_YLikelihood(Multinomial):
    __slots__ = []

    def get_loglikelihood(self):
        if self.value[0]:
            pr0 = 1.
            for x, t, s in self.in_edges:
                if s.value[0]:
                    pr0 *= 1. - t.value * x.value[1]
            pr0 = (1. - pr0)
            likelihood = pr0

        elif self.value[2]:
            pr0 = 1.
            pr2 = 1.
            for x, t, s in self.in_edges:
                if s.value[2]:
                    pr2 *= 1. - t.value * x.value[1]
                elif s.value[0]:
                    pr0 *= 1. - t.value * x.value[1]
            pr2 = (pr0 - pr2*pr0)
            likelihood = pr2

        else:
            pr1 = 1.
            for x, t, s in self.in_edges:
                if not s.value[1]:
                    pr1 *= 1. - t.value * x.value[1]
            likelihood = pr1

        if likelihood > 0:
            return np.log(likelihood)
        else:
            return -np.inf


    def sample(self):
        self.value = self.dist.rvs(*self.params)


class ORNORModel(BaseModel):

    def __init__(self, rels, DEG):
        BaseModel.__init__(self)

        # identify TF considered in current network
        X_list = rels['srcuid'].unique()

        # define a conditional probability table for the observed values
        # H is hidden true value of gene, Z is observed value in DEG
        a = 0.05
        b = 0.02

        PHZTable = np.empty(shape=(3,3), dtype=np.float64)
        PHZTable[0] = [1. - a - b,        a,          b]
        PHZTable[1] = [         a, 1. - 2*a,          a]
        PHZTable[2] = [         b,        a, 1. - a - b]

        Ynodes = {}
        for trg, val in DEG.items():
            obs_val = [0, 0, 0]
            obs_val[1 + val] = 1
            y_prob = PHZTable[:, np.argwhere(obs_val)[0, 0]]

            Ynodes[trg] = ORNOR_YLikelihood('Y', trg, y_prob, value=obs_val)

        Xnodes, Tnodes = {}, {}
        for src in X_list:
            Xnodes[src] = Multinomial('X', src, [0.99, 0.01])
            Tnodes[src] = Beta('T', src, 5, 5)

        Snodes = {}
        for edg in rels.index:
            
            src, trg = edg
            
            Snodes[edg] = Multinomial('S', edg, [0.1, 0.8, 0.1])
            Snodes[edg].children.append(Ynodes[trg])
            
            Xnodes[src].children.append(Ynodes[trg])
            Tnodes[src].children.append(Ynodes[trg])
            
            Ynodes[trg].parents.append(Xnodes[src])
            Ynodes[trg].parents.append(Tnodes[src])
            Ynodes[trg].parents.append(Snodes[edg])
            
            Ynodes[trg].in_edges.append([Xnodes[src], Tnodes[src], Snodes[edg]])

        self.vars['X'] = Xnodes
        self.vars['T'] = Tnodes
        self.vars['S'] = Snodes

