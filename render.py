#!/usr/bin/env python2.6

from enthought.mayavi import mlab 
from numpy import loadtxt

import sys

rate = float(sys.argv[1])
bfname = 'before_train_%f.txt' % rate
afname = 'after_train_final_%f.txt' % rate

def read_proj(fname):
    a = loadtxt(fname)
    return a[:,0], a[:,1], a[:,2], a[:,3], a[:,4]

bx, by, bz, be, ba = read_proj(bfname)
ax, ay, az, ae, aa = read_proj(afname)

print az.shape, aa.shape, aa

#mlab.points3d(bx, by, bz, ba, colormap = 'jet', scale_factor = 2.5e-2)#, color = (1, 0, 0))
mlab.points3d(ax, ay, az, aa, colormap = 'jet', scale_factor = 8e-1, scale_mode = 'none')#, color = (1, 0, 0))

mlab.show()
