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

class Derivatives(object):
    def __init__(self, n_out, hidden, output):
        self.n_out = n_out
        self.hidden = hidden
        self.output = output

class PlaneNNet(object):
    def __init__(self, imgdim, n_hidden, n_out):
        self.imgdim = imgdim
        self.n_hidden = n_hidden
        self.n_out = n_out

        self.W1 = m.random.random((imgdim + 1, n_hidden)) / 1e3
        self.W2 = m.random.random((n_hidden + 1, n_out)) / 1e3

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
    inp = np.asmatrix(m.resize(inp, (n_rows + 1, 1)))
    inp[n_rows, 0] = 1

    sh = nnet.W1.T * inp
    xh = sigma(sh)
    n_xh_rows = xh.shape[0]
    xh = np.asmatrix(m.resize(xh, (n_xh_rows + 1, 1)))
    xh[n_xh_rows, 0] = 1

    so = nnet.W2.T * xh
    xo = sigma(so)

    return EvalNNet(inp, sh, xh, so, xo)

def back_prop(nnet, evaluated):
    ''' PlaneNNet -> EvalNNet -> Derivatives '''

    def _backprop_hidden(nnet, evaluated):
        common = np.asmatrix(evaluated.inp * sigma_p(evaluated.sh).T)
        ret = []
        for out_ind in xrange(nnet.n_out):
            w_mat = np.asmatrix(m.repmat(nnet.W2[:nnet.n_hidden, out_ind].T, common.shape[0], 1))
            coeff = np.asmatrix(sigma_p(evaluated.so))[out_ind, 0]
            to_append = m.multiply(common, w_mat)
            ret.append(to_append * coeff)

        return ret

    def _backprop_output(nnet, evaluated):
        ret = []
        for out_ind in xrange(nnet.n_out):
            to_add = m.zeros((nnet.n_hidden + 1, nnet.n_out))
            to_add[:, out_ind] = np.asmatrix(nnet.W2)[:, out_ind]
            ret.append(to_add * sigma_p(evaluated.xo)[out_ind, 0])
        return ret

    return Derivatives(nnet.n_out, 
            _backprop_hidden(nnet, evaluated),
            _backprop_output(nnet, evaluated))

def eval_dist(e1, e2):
    delta = e1.xo - e2.xo
    assert (delta.T * delta).shape == (1, 1), str((delta.T * delta).shape)
    return sqrt((delta.T * delta)[0,0])

def update_nnet(nnet, e1, e2, d1, d2, are_similar):
    ''' PlaneNNet -> EvalNNet -> EvalNNet -> Derivatives -> Derivatives -> bool -> PlaneNNet '''
    M = 0.01
    ALPHA = 10.
    dist = eval_dist(e1, e2)
    if are_similar and dist <= M:
        print 'close enough'
        return nnet
    if not are_similar and dist >= M:
        print 'far enough apart'
        return nnet
    if dist < 1e-8:
        print 'distance too small:', dist, 'similar:', are_similar
        return nnet

    if not are_similar and dist <= M:
        print 'NOT SIMILAR, d=%f' % dist

    dx = e1.xo[0,0] - e2.xo[0,0]
    dy = e1.xo[1,0] - e2.xo[1,0]
    dz = e1.xo[2,0] - e2.xo[2,0]

    def _calc_deriv(g_mats, h_mats):
        return dx * (g_mats[0] - h_mats[0]) + dy * (g_mats[1] - h_mats[1]) + \
                dz * (g_mats[2] - h_mats[2])

    dDw_dhidden = _calc_deriv(d1.hidden, d2.hidden)
    dDw_dout = _calc_deriv(d1.output, d2.output)

    ret_nnet = PlaneNNet(nnet.imgdim, nnet.n_hidden, nnet.n_out)
    if are_similar:
        ret_nnet.W1 = nnet.W1 - ALPHA * dDw_dhidden
        ret_nnet.W2 = nnet.W2 - ALPHA * dDw_dout
    else:
        ret_nnet.W1 = nnet.W1 + ALPHA * (M / dist - 1) * dDw_dhidden
        ret_nnet.W2 = nnet.W2 + ALPHA * (M / dist - 1) * dDw_dout

    return ret_nnet

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

    nnet = PlaneNNet(96 * 96, 20, 3)

    for (i, (cat, (e1, a1, e2, a2))) in enumerate(train_set[:300]):
        print i, '/', len(train_set)
        imgset1 = elevations[e1][a1]
        imgset2 = elevations[e2][a2]
        if len(imgset1) != 6 or len(imgset2) != 6:
            continue
        for img1 in imgset1:
            for img2 in imgset2:
                n_total += 1
                im1data = np.asmatrix(imglis[img1])
                im2data = np.asmatrix(imglis[img2])

                print np.abs(nnet.W1).sum()
                e1 = fwd_prop(nnet, im1data)
                e2 = fwd_prop(nnet, im2data)
                d1 = back_prop(nnet, e1)
                d2 = back_prop(nnet, e2)

                nnet = update_nnet(nnet, e1, e2, d1, d2, cat == ADJ)

        print n_divzero, 'zero out of', n_total
