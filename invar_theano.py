#!/usr/bin/env python

import theano as T
import theano.tensor as M

import cProfile

import numpy.matlib as np
from time import time

from collections import defaultdict as dd

rng = np.random.RandomState(int(time()))

def rand_init(fan_in, fan_out):
    ret = np.asarray(rng.uniform(
        low = -np.sqrt(3. / fan_in),
        high = np.sqrt(3. / fan_in),
        size = (fan_in, fan_out)), dtype = T.config.floatX)
    return ret

inp_dim = 96 * 96
hidden_dim = 20

class DimLayer(object):
    inp1 = M.matrix(name = 'input_0')
    inp2 = M.matrix(name = 'input_1')

    arrdict = dd(lambda : [
                T.shared(rand_init(inp_dim, hidden_dim)),
                T.shared(np.zeros((hidden_dim, 1))),
                T.shared(rand_init(hidden_dim, 1)),
                T.shared(np.zeros((1, 1)))
                ])

    def __init__(self, name, inp):
        eqvars = self.arrdict[name]
        w_hidden, b_hidden, w_output, b_output = eqvars

        hidden = T.dot(w_hidden.T, inp) + b_hidden
        hidden_act = M.tanh(hidden)
        output = (T.dot(w_output.T, hidden_act) + b_output)
        self.proj = output.sum()
        self.proj_fcn = T.function([inp], self.proj)
        
class NNet(object):
    def __init__(self):
        x1 = DimLayer('x', DimLayer.inp1)
        x2 = DimLayer('x', DimLayer.inp2)
        y1 = DimLayer('y', DimLayer.inp1)
        y2 = DimLayer('y', DimLayer.inp2)
        z1 = DimLayer('z', DimLayer.inp1)
        z2 = DimLayer('z', DimLayer.inp2)

        dw = M.sqrt((x1.proj - x2.proj) ** 2 + \
                (y1.proj - y2.proj) ** 2 + \
                (z1.proj - z2.proj) ** 2)

        m = 0.05
        simparmupdates = {}
        dissimparmupdates = {}
        for lis in DimLayer.arrdict.values():
            for parm in lis:
                simparmupdates[parm] = parm - 0.1 * dw * M.grad(dw, parm)
                dissimparmupdates[parm] = parm + 0.1 * (m - dw) * M.grad(dw, parm)

        self.simupdatefcn = T.function([DimLayer.inp1, DimLayer.inp2], updates = simparmupdates)
        self.dissimupdatefcn = T.function([DimLayer.inp1, DimLayer.inp2], updates = dissimparmupdates)
        self.pfcn = T.function([DimLayer.inp1, DimLayer.inp2], dw)

    def sim(self, img1, img2):
        self.simupdatefcn(img1, img2)

    def dissim(self, img1, img2):
        self.dissimupdatefcn(img1, img2)

    def dist(self, img1, img2):
        return self.pfcn(img1, img2)

img1 = np.random.random((96 * 96, 1))
img2 = np.random.random((96 * 96, 1))
img1 /= np.absolute(img1).sum()
img2 /= np.absolute(img2).sum()

nnet = NNet()

def iter():
    nnet.dissim(img1, img2)
    print nnet.dist(img1, img2)

def main():
    for i in xrange(100):
        iter()

if __name__=='__main__':
    main()
    #cProfile.run('main()', 'invar_prof')
