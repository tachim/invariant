#!/usr/bin/env python

import numpy as np
import math

class NetLayer(object):
    GRAD_CONST = 0.01
    def __init__(self, W, sigma, sigma_p, prev = None, next = None, fixed = False):
        self.W = W
        self.prev = prev
        self.next = next
        self.sigma = sigma
        self.sigma_p = sigma_p
        self.fixed = fixed

    def fwdprop(self, inp):
        ''' returns the result of forward propagating the input from this
        layer to the final one '''
        self.sn = self.W * inp
        self.xn = self.sigma(self.sn)
        if self.next:
            return self.next.fwdprop(self.xn)
        else:
            return self.xn

    def backprop(self, loss_deriv = None):
        if loss_deriv:
            # we're the output
            dl_dx = loss_deriv(self.xn)
            self.dl_ds = dl_dx.multiply(self.sigma_p(self.sn))

            if self.prev:
                self.prev.backprop()

            if not self.fixed and self.prev:
                dl_dw = self.dl_ds * self.prev.xn.T
                self.W -= GRAD_CONST * dl_dw

        else:
            # we're in a hidden layer
            dl_dx = self.next.W.T * self.next.dl_ds
            self.dl_ds = dl_dx.multiply(self.sigma_p(self.sn))
            if not self.prev:
                # we are the input layer
                self.prev.backprop()
            if not self.fixed and not self.prev:
                dl_dw = self.dl_ds * self.prev.xn.T
                self.W -= GRAD_CONST * dl_dw

class AgglomLayer(NetLayer):
    def __init__(self, n_dims):
        self.n_dims = n_dims
        W = np.identity(self.n_dims * 2)
        NetLayer.__init__(self, W, self._sigma, self._sigma_p, fixed = True)

    def _sigma(self, inp):
        gw1 = inp[: self.n_dims, 0]
        gw2 = inp[self.n_dims :, 0]

        dw = math.sqrt(((gw1 - gw2).T * (gw1 - gw2))[0,0])
        return dw

    def _sigma_p(

    def backprop(self):
        return NetLayer.backprop(self, self._lossderiv)
