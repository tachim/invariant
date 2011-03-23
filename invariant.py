#!/usr/bin/env python2.6

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

class PlaneNNET(object):
    ALPHA = 0.001

    def __init__(self, W1, W2, imgdim = 96, n_hidden = 20, n_out = 3):
        self.n_hidden = n_hidden
        self.imgdim = imgdim
        self.n_out = n_out

        self.W1 = W1
        self.W2 = W2

    def fwdprop(self, x0):
        self.x0 = x0

        self.s1 = self.W1.T * self.x0
        self.x1 = sigma(self.s1)

        nrows_x1 = self.x1.shape[0]
        assert self.x1.shape[1] == 1

        self.x1.resize((nrows_x1 + 1, 1))
        self.x1[nrows_x1,0] = 1

        self.s2 = self.W2.T * self.x1
        self.x2 = sigma(self.s2)

        return self.x2

    def calc_derivs(self):
        sp_1 = np.asmatrix(sigma_p(self.s1))
        sp_2 = np.asmatrix(sigma_p(self.s2))

        tmp = self.x1 * sp_2.T
        self.outer = [np.asmatrix(np.zeros((self.n_hidden, self.n_out))) for i in xrange(self.n_out)]
        for i in xrange(self.n_out):
            self.outer[i][:,i] = tmp[:,i]

        lprods = [sp_2[i,0] * np.multiply(self.W2[:,i], sp_1) for i in xrange(self.n_out)]
        self.inner = [self.x0 * e.T for e in lprods]

    def calc_update(self, dx, dy, dz, dw):
        ret = (dx * self.inner[0] + dy * self.inner[1] + dz * self.inner[2],
                dx * self.outer[0] + dy * self.outer[1] + dz * self.outer[2])
        return (ret[0] / dw, ret[1] / dw)


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
