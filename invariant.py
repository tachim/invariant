#!/usr/bin/env python2.6

import numpy.matlib as m
import numpy as np
from math import exp, sqrt
from time import time
from struct import unpack
from collections import defaultdict
from random import sample, shuffle

def logistic(p):
    return 1. / (1 + exp(-p))

def logistic_p(p):
    v = logistic(p)
    return v * (1-v)

sigma = np.vectorize(logistic)
sigma_p = np.vectorize(logistic_p)

class PlaneNNet(object):
    def __init__(self, imgdim, n_hidden, n_out):
        self.imgdim = imgdim
        self.n_hidden = n_hidden
        self.n_out = n_out

        self.W1 = np.zeros((imgdim + 1, n_hidden))
        self.W2 = np.zeros((n_hidden + 1, n_out))

class EvalNNet(object):
    def __init__(self, inp, sh, xh, so, xo):
        self.inp = inp
        self.sh = sh
        self.xh = xh
        self.so = so
        self.xo = xo

def fwd_prop(nnet, inp):
    ''' PlaneNNet -> Input -> PlaneNNet'''
    # add constant feature
    n_rows = inp.shape[0]
    inp.resize((n_rows + 1, 1))
    inp[n_rows, 0] = 1

    sh = nnet.W1.T * inp
    xh = sigma(sh)
    n_xh_rows = xh.shape[0]
    xh.resize((n_xh_rows + 1, 1))
    xh[n_xh_rows, 0] = 1

    so = nnet.W2.T * xh
    xo = sigma(so)

    return EvalNNet(inp, sh, xh, so, xo)

def backprop_hidden(nnet, evaluated):
    common = evaluated.inp * sigma_p(evaluated.sh).T
    ret = []
    for out_ind in xrange(nnet.n_out):
        w_mat = repmat(nnet.W2[:nnet.n_hidden, out_ind].T, common.shape[0], 1)
        coeff = sigma_p(evaluated.so)[out_ind, 0]
        ret.append(np.dot(common, w_mat) * coeff)

    return ret

def backprop_output(nnet, evaluated):
    ret = []
    for out_ind in xrange(nnet.n_out):
        to_add = m.zeros((nnet.n_hidden + 1, nnet.n_out))
        to_add[:, out_ind] = nnet.W2[:, out_ind]
        ret.append(to_add * sigma_p(evaluated.xo[out_ind, 0]))
    return ret

def load_images():
    dat = open('smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat', 'rb')
    cat = open('smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat', 'rb')
    info = open('smallnorb-5x46789x9x18x6x2x96x96-training-info.mat', 'rb')

    dat.read(24)
    cat.read(20)
    info.read(20)

    azimuths = defaultdict(lambda : defaultdict(list))
    elevations = defaultdict(lambda : defaultdict(list))

    retlis = []

    # process an individual airplane image
    def _procnext():
        tmp = np.fromfile(dat, np.uint8, 96 * 96).astype(np.float64) / 255.0
        if tmp.shape[0] == 0:
            raise IOError("out of shit")
        next_im = np.asmatrix(tmp.reshape((96 * 96, 1)))
        # skip the paired image
        dat.read(96 * 96)

        # read the info
        meta = unpack('iiii', info.read(16))
        instance, elevation, azimuth, lighting = meta

        # read the category
        category = unpack('i', cat.read(4))[0]

        if category != 2 or instance != 7:
            return False

        retlis.append(next_im)

        azimuths[azimuth][elevation].append(len(retlis) - 1)
        elevations[elevation][azimuth].append(len(retlis) - 1)

        return True

    # load all the matrices
    ct = 0
    while True:
        try:
            result = _procnext()
            if result:
                ct += 1
        except IOError:
            break
    print ct, 'pics'

    dat.close()
    cat.close()
    info.close()

    return (elevations, azimuths, retlis)

NONADJ = 0
ADJ = 1
M = 1.25

if __name__=='__main__':
    imgdim = 96 * 96
    n_hidden = 20
    n_out = 3
#    W1 = np.asmatrix(np.random.rand(imgdim + 1, n_hidden).astype(np.float64))
#    W2 = np.asmatrix(np.random.rand(n_hidden + 1, n_out).astype(np.float64))
    W1 = np.asmatrix(np.zeros((imgdim + 1, n_hidden)).astype(np.float64))
    W2 = np.asmatrix(np.zeros((n_hidden + 1, n_out)).astype(np.float64))
    net1 = PlaneNNET(W1, W2, imgdim = imgdim)
    net2 = PlaneNNET(W1, W2, imgdim = imgdim)

    # actually process the pairs :)
    elevations, azimuths, imglis = load_images()

    n_e, n_a = len(elevations), len(azimuths)

    pairs = []
    nonadj = []

    for e in xrange(8):
        for a in xrange(0, 34, 2):
            pairs.append((e, a, (e+1)%n_e, a))
            pairs.append((e, a, e, (a+1)%n_a))

    for e in xrange(9):
        for a in xrange(0, 34, 2):
            for e1 in xrange(9):
                for a1 in xrange(0, 34, 2):
                    if abs(e - e1) > 1 or abs(a - a1) > 1:
                        nonadj.append((e, a, e1, a1))
    nonadj = sample(nonadj, 5000)
    pairs = [(ADJ, e) for e in pairs]
    nonadj = [(NONADJ, e) for e in nonadj]

    train_set = pairs + nonadj
    shuffle(train_set)

    n_divzero = 0
    n_total = 0

    for (i, (cat, (e1, a1, e2, a2))) in enumerate(train_set[:300]):
        print i, '/', len(train_set)
        imgset1 = elevations[e1][a1]
        imgset2 = elevations[e2][a2]
        if len(imgset1) != 6 or len(imgset2) != 6:
            continue
        for img1 in imgset1:
            for img2 in imgset2:
                n_total += 1
                im1data = imglis[img1]
                im2data = imglis[img2]

                print 'DIFFERENCE:', np.abs(im1data - im2data).sum()

                # add the constant feature
                nrows_im1 = im1data.shape[0]
                im1data = np.resize(im1data, (nrows_im1 + 1, 1))
                im1data[nrows_im1,0] = 1
                nrows_im2 = im2data.shape[0]
                im2data = np.resize(im2data, (nrows_im2 + 1, 1))
                im2data[nrows_im2,0] = 1

                g1 = net1.fwdprop(im1data)
                g2 = net2.fwdprop(im2data)
                print g1, g2
                delta = g1 - g2
                dw2 = delta.T * delta
                if dw2 == 0:
                    n_divzero += 1
                    continue
                net1.calc_derivs()
                net2.calc_derivs()
                dw = sqrt(dw2)
                if cat == NONADJ and dw > M:
                    continue
                dx, dy, dz = delta[0,0], delta[1,0], delta[2,0]

                up1 = net1.calc_update(dx, dy, dz, dw)
                up2 = net2.calc_update(dx, dy, dz, dw)
                inner_grad = up1[0] - up2[0]
                outer_grad = up1[1] - up2[1]

                if cat == NONADJ:
                    outer_grad *= -(M - dw)
                    inner_grad *= -(M - dw)
                else:
                    outer_grad *= dw
                    inner_grad *= dw

                net1.W1 -= inner_grad * PlaneNNET.ALPHA
                net1.W2 -= outer_grad * PlaneNNET.ALPHA
        print n_divzero, 'zero out of', n_total
