#!/usr/bin/env python

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
