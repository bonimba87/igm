from __future__ import division, print_function

import numpy as np
import h5py

from .restraint import Restraint
from ..model.forces import HarmonicLowerBound, EllipticEnvelope
from ..model import Particle

try:
    UNICODE_EXISTS = bool(type(unicode))
except NameError:
    unicode = lambda s: str(s)


# radial position of of the bead surface in radius units
# on 1 the surface contacts the envelope
def snormsq_sphere(x, R, r):
    # return np.square(
    #     (np.linalg.norm(x, axis=1) + r) / R**2
    # )
    return np.sum(np.square(x), axis=1) / (R-r)**2


def snormsq_ellipsoid(x, semiaxes, r):
    a, b, c = np.array(semiaxes) - r
    sq = np.square(x)
    return sq[0]/(a**2) + sq[1]/(b**2) + sq[2]/(c**2)


def snorm(x, shape=u"sphere", **kwargs):
    if shape == u"sphere":
        return np.linalg.norm(x) / kwargs['a']
    elif shape == u"ellipsoid":
        a, b, c = kwargs['a'], kwargs['b'], kwargs['c']
        return np.sqrt( (x[0]/a)**2 + (x[1]/b)**2 + (x[2]/c)**2 )


class Damid(Restraint):
    """
    Object handles Hi-C restraint

    Parameters
    ----------
    damid_file : activation distance file for damid

    contact_range : int
        defining contact range between 2 particles as contactRange*(r1+r2)


    """
    def __init__(self, damid_file, contact_range=0.05, nuclear_radius=5000.0, shape="sphere",
                 semiaxes=(5000, 5000, 5000), k=1.0):

        self.shape = unicode(shape)

        if self.shape == u"sphere":
            self.a = self.b = self.c = nuclear_radius
        elif self.shape == u"ellipsoid":
            self.a, self.b, self.c = semiaxes

        self.contact_range = contact_range
        self.nuclear_radius = nuclear_radius
        self.k = k
        self.forceID = []
        self._load_actdist(damid_file)
    #-


    def _load_actdist(self,damid_actdist):
        self.damid_actdist = DamidActivationDistanceDB(damid_actdist)


    def _apply_envelope(self, model):

        center = model.addParticle([0., 0., 0.], 0., Particle.DUMMY_STATIC)
        cutoff = 1 - self.contact_range

        affected_particles = [
            i for i, d in self.damid_actdist
            if snormsq_ellipsoid(
                model.particles[i].pos,
                np.array([self.a, self.b, self.c]) * cutoff,
                model.particles[i].r
            ) >= d**2
        ]

        f = model.addForce(
            EllipticEnvelope(
                affected_particles,
                center,
                np.array([self.a, self.b, self.c])*cutoff,
                -self.k,
                scale=cutoff*np.mean([self.a, self.b, self.c])
            )
        )

        self.forceID.append(f)


    def _apply(self, model):
        return self._apply_envelope(model)

#==

class DamidActivationDistanceDB(object):
    """
    HDF5 activation distance iterator
    """

    def __init__(self, damid_file, chunk_size = 10000):
        self.h5f = h5py.File(damid_file,'r')
        self._n = len(self.h5f['loc'])

        self.chunk_size = min(chunk_size, self._n)

        self._i = 0
        self._chk_i = chunk_size
        self._chk_end = 0

    def __len__(self):
        return self._n

    def __iter__(self):
        return self

    def __next__(self):
        if self._i < self._n:
            self._i += 1
            self._chk_i += 1

            if self._chk_i >= self.chunk_size:
                self._load_next_chunk()

            return (int(self._chk_row[self._chk_i]),
                    self._chk_data[self._chk_i])
        else:
            self._i = 0
            self._chk_i = self.chunk_size
            self._chk_end = 0
            raise StopIteration()

    def next(self):
        return self.__next__()

    def _load_next_chunk(self):
        self._chk_row = self.h5f['loc'][self._chk_end : self._chk_end + self.chunk_size]
        self._chk_data = self.h5f['dist'][self._chk_end : self._chk_end + self.chunk_size]
        self._chk_end += self.chunk_size
        self._chk_i = 0

    def __del__(self):
        try:
            self.h5f.close()
        except:
            pass

