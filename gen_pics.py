#!/usr/bin/env python2.6

import numpy.matlib as m
import numpy as np
from math import exp, sqrt
from time import time
from struct import unpack
from collections import defaultdict
from random import sample, shuffle

from PIL import Image

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
        tmp = np.fromfile(dat, np.uint8, 96 * 96)#.astype(np.float64) / 255.0
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

        if category != 2 or instance != 7 or lighting != 3:
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

    for e in elevations:
        for a in elevations[e]:
            for (i, imgind) in enumerate(elevations[e][a]):
                fname = '%d_%d_%d.jpg' % (e, a, i)
                im = Image.fromarray(imglis[imgind].reshape((96, 96)), mode = 'L')
                im.save(fname, 'JPEG')

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
    
