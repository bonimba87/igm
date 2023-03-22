from __future__ import division, print_function

import numpy as np
import h5py

from .restraint import Restraint
from ..model.forces import ExpEnvelope    # restraint forces associated with damid
from ..model import Particle
from ..utils.log import logger
from ..utils import HmsFile, VolumeFile

try:
    UNICODE_EXISTS = bool(type(unicode))
except NameError:
    unicode = lambda s: str(s)

def snormsq_exp(x, vol):

    """
    Compute the distance of a bead to a lamina that is defined via a grid

    INPUT
      * x, bead/locus coordinates
      * grid, center, origin (np.array, 3): voxel sizes, map center and map origin (lowest, leftmost voxel)
      * matrice (np.array, (nx, ny, nz, 3)),   matrice[i,j,k] = EDT([i,j,k]) (Euclidean Distance Transform)
 
    OUTPUT 
       * distanza
    """

    voxel_xyz = np.round(np.array((x - vol.origin)/vol.grid)).astype(int)  # voxel coordinates are integer numbers

    # if voxel is inside the grid...
    if (voxel_xyz >= np.zeros(3)).all()  and (voxel_xyz < vol.nvoxel).all():

            # compute distance squared
            distanza = np.dot((voxel_xyz - vol.matrice[tuple(voxel_xyz)]) * vol.grid,\
                              (voxel_xyz - vol.matrice[tuple(voxel_xyz)]) * vol.grid)   # distance bead 
                                                                                                  # surface - surface
    else:
            distanza = np.dot((voxel_xyz - vol.center) * vol.grid, \
                              (voxel_xyz - vol.center) * vol.grid)     # distance to the center of the grid (if nucleoli aut similia)

    return distanza
#-



class GenDamid(Restraint):

    """
    Object handle Extended DamID restraint: enforces proximity of genome particles to ANY laminas from experiments, either nuclear lamina or nucleolar or nuclear body-style laminas. If any of those can be represented with built-in ellipsoidal shapes, than we don't need to overcomplicate things with grids, but we can use the standard "DamID" restraint.

    Parameters
    ----------
    * damid_file : activation distance file for damid

    * contact_range : float,  defines tolerance of contact with the lamina. The grid voxels are enlarged or reduced by this factor, to define a hollow volume shell inside which we say a contact with lamina is expressed.
    * volume_file: file containing the complete information about the grid/lamina/volume (see format in INPUTS example)
    * k: float
      see also "GenEnvelope" restraint. This is a force amplitude factor, it does not retain any "spring constant" intepretation       when experimental envelopes are used. It is crucial to observe that the parameter is positive in the config file BUT its ne      gative value is passed as a parameter to the "ExpEnvelope" force. Negative k always indicates that a lamina restraint of some kind is being applied
     
    """

    def __init__(self, damid_file, shape, volume_file, k =1.0, contact_range=0.05):

        self.shape = unicode(shape)

        # recapitulate parameters
        self.k             = k
        self.volume_file   = volume_file
        self.contact_range = contact_range
        self._load_actdist(damid_file)    # load lamina DamiD restraint activation file

        self.forceID = []


    def _load_actdist(self,damid_actdist):

        """ Read in file containing current DAMID activation distances """
        self.damid_actdist = DamidActivationDistanceDB(damid_actdist)


    def _apply_envelope(self, model):

        """ Effectively apply damid restraints to the different beads, if distance is smaller than activation distance """

        #--- load volume file ---
        vol = VolumeFile(self.volume_file)
        vol.load_file()
     
        # here: load volume file, compute distance to lamina of only those particles that are listed in damid_actdist
        # that is a teeny tiny fraction of all the voxels that are listed in he volume file

        affected_particles = [
            i for i, d in self.damid_actdist
            if snormsq_exp(
                model.particles[i].pos, vol     # there is no n_struct here, models (aka, structures) are run indepedently
                          ) <= d**2    # where does the cutoff enter here?
        ]

        #logger.info('---------------------------------')
        #logger.info('Number of affected particles = ')
        #logger.info(len(affected_particles))
        #logger.info('---------------------------------')

        # apply force
        f = model.addForce(
            ExpEnvelope(
                affected_particles,
                self.volume_file,
                -self.k,    ## we don't need to push this anymore  negative k to inform Lammps we are "pushing" toward the lamina voxels (~ enforce a lamina contact)
                self.contact_range, note = Restraint.DAMID
            )
        )

        self.forceID.append(f)


    def _apply(self, model):
        return self._apply_envelope(model)


    def __repr__(self):
        return 'LaminaDamID[shape={},map={},k={}, cr={}]'.format(self.shape, self.volume_file, self.k, self.contact_range)
#==

class DamidActivationDistanceDB(object):
    """
    HDF5 activation distance iterator: read in damid activation distance file in chunks
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

