from __future__ import division, absolute_import, print_function
import numpy as np

from ..utils import VolumeFile

# Define "Force" class, and then all the force instances


class Force(object):
    """
    Define Basic Force type
    """

    EXCLUDED_VOLUME = 0
    HARMONIC_UPPER_BOUND = 1
    HARMONIC_LOWER_BOUND = 2
    ENVELOPE = 3
    GENERAL_ENVELOPE = 4

    FTYPES = ["EXCLUDED_VOLUME","HARMONIC_UPPER_BOUND","HARMONIC_LOWER_BOUND", "ENVELOPE", "GENERAL_ENVELOPE"]

    def __init__(self, ftype, particles, para, note=""):
        self.ftype = ftype
        self.particles = particles
        self.parameters = para
        self.note = note
        self.rnum = 1 # rnum is used in case of "collective" forces

    def __str__(self):
        return "FORCE: {} {}".format(Force.FTYPES[self.ftype],
                                        self.note)

    def __repr__(self):
        return self.__str__()

    def getScore(self, particles):
        return 0

    def getViolationRatio(self, particles):
        return 0

class ExcludedVolume(Force):

    """
    Exluded Volume Restraint, for a pair of particles check if they overlap (violation) or not

    e = ri + rj - d_ij if ri + rj - d_ij >0, otherwise 0

    Parameters
    ----------
    particles : tuple(int, int)
        Two particle indexes
    note : str
        additional information
    """

    ftype = Force.EXCLUDED_VOLUME

    def __init__(self, particles, k=1.0, note=""):
        self.particles = particles
        self.k = k
        self.note = note
        self.rnum = len(particles) * len(particles)

    def __str__(self):
        return "FORCE: {} (NATOMS: {}) {}".format(Force.FTYPES[self.ftype],
                                                  len(self.particles),
                                                  self.note)

    def getScore(self, particles):
        return self.getScores(particles).sum()

    def getScores(self, particles):

        # scipy.spatial.distance takes an array X (m, n); m = observations, n = space dimensionality
        from scipy.spatial import distance

        crd = np.array([particles[i].pos for i in self.particles])
        rad = np.array([[particles[i].r for i in self.particles]]).T

        dist = distance.pdist(crd)

        # distance is computed by plain sum
        cap = distance.pdist(rad, lambda u, v: u + v)

        # if (r_i + r_j) - d_ij < 0, then no penalty, so clip value to 0, otherwise, keep number
        s = (cap - dist).clip(min=0)

        return s.ravel()

#-

class HarmonicUpperBound(Force):
    """
    Harmonic upper bound force

    e = 1/2*k*(x-d)^2 if x > d; otherwise 0

    Parameters
    ----------
    particles : tuple(int, int)
        Two particle indexes
    d : float
        mean distance
    k : float
        spring constant
    note : str
        additional information
    """

    ftype = Force.HARMONIC_UPPER_BOUND

    def __init__(self, particles, d=0.0, k=1.0, note=""):
        if len(particles) != 2:
            raise ValueError("Two particles required")
        else:
            self.i, self.j = particles

        self.d = d
        self.k = k
        self.note = note
        self.rnum = 1

    def __str__(self):
        return "FORCE: {} {} {} {}".format(Force.FTYPES[self.ftype],
                                        self.i, self.j,
                                        self.note)

    def getScore(self, particles):

        dist = particles[self.i] - particles[self.j]

        return 0 if dist <= self.d else self.k*(dist - self.d)

    def getViolationRatio(self, particles):

        return 0 if self.d == 0 else self.getScore(particles) / (self.k * self.d)
#-

class HarmonicLowerBound(Force):
    """
    Harmonic lower bound force

    e = 1/2*k*(x-d)^2 if x < d; otherwise 0

    Parameters
    ----------
    particles : tuple(int, int)
        Two particle indexes
    d : float
        mean distance
    k : float
        spring constant
    note : str
        additional information
    """
    ftype = Force.HARMONIC_LOWER_BOUND

    def __init__(self, particles, d=0.0, k=1.0, note=""):
        if len(particles) != 2:
            raise ValueError("Two particles required")
        else:
            self.i, self.j = particles


        self.d = d
        self.k = k
        self.note = note
        self.rnum = 1

    def __str__(self):
        return "FORCE: {} {} {} {}".format(Force.FTYPES[self.ftype],
                                        self.i, self.j,
                                        self.note)

    def getScore(self, particles):

        dist = particles[self.i] - particles[self.j]

        return 0 if dist >= self.d else self.k*(self.d - dist)

    def getViolationRatio(self, particles):

        return 0 if self.d == 0 else self.getScore(particles) / (self.k * self.d)
#-

class EllipticEnvelope(Force):

    ftype = Force.ENVELOPE

    def __init__(self, particle_ids,
                 center=(0, 0, 0),
                 semiaxes=(5000.0, 5000.0, 5000.0),
                 k=1.0, note="", scale=100.0):

        self.shape = 'ellipsoid'
        self.center = np.array(center)
        self.semiaxes = np.array(semiaxes)
        self.particle_ids = particle_ids
        self.k = k
        self.scale = scale  # pff. This is to actually give a
                            # "relative  measure" for violation ratios.
        self.note = note
        self.rnum = len(particle_ids)

    def getScore(self, particles):

        E = 0
        for i in self.particle_ids:
            p = particles[i]
            s2 = np.square(self.semiaxes - p.r)
            x = p.pos
            x2 = x**2
            k2 = np.sum(x2 / s2)
            if k2 > 1:
                t = ( 1.0 - 1.0/np.sqrt(k2) )*np.linalg.norm(x)
                E += 0.5 * (t**2) * self.k
        return E

    def getScores(self, particles):

        scores = np.zeros(len(self.particle_ids))

        for q, i in enumerate(self.particle_ids):

            p = particles[i]
            s2 = np.square(self.semiaxes - p.r)
            x = p.pos
            x2 = x**2

            k2 = np.sqrt(np.sum(x2 / s2))

            # note that those scores are somewhat approximate
            if k2 > 1 and self.k > 0:
                t = ( 1.0 - 1.0/np.sqrt(k2) )*np.linalg.norm(x)/self.scale

            elif k2 < 1 and self.k < 0:
                t = ( 1.0 - np.sqrt(k2))

            else:
                t = 0

            scores[q] = max(0, t)

        return scores

    def getViolationRatio(self, particles):
        ave_t = np.sqrt(2 * self.getScore(particles) / self.k)
        ave_ax = np.sqrt(np.sum(np.square(self.semiaxes)))
        return ave_t/ave_ax

    def getViolationRatios(self, particles):
        return self.getScores(particles)


class ExpEnvelope(Force):

    """
    Envelope/Nuclear body restraint from experimental map.

    We use Euclidean Distance Transform (EDT) to find the closest lamina voxel for each voxel.
    Their distance is used as a single restraint residual.


    Parameters
    ---------
    particles : tuple(int, int, ...)
        N particle indexes
    volume_file: string
        filename, contains all the information about the imaging map (voxels, grid spacings, occupancy)
    k: float
        elastic constant
    note : str
        additional information
    contact_range: float (<1)
        tolerance, this defines a concentric smaller lamina; anything in the outer shell is assumed to be in contact with lamina (even if no physical contact actually occurs)

    Returns
    -------
    scores: array of floats
        scores[k] gives the single restraint for particle k in the simulation 
    """

    ftype = Force.GENERAL_ENVELOPE

    def __init__(self, particle_ids,
                 volume_file = "",
                 k=1.0, contact_range = 1.0, note = ""):

        self.shape = 'exp_map'
        self.volume_file = volume_file
        self.particle_ids = particle_ids
        self.k = k
        self.contact_range = contact_range  # pff. This is to actually give a
        self.note = note
        self.rnum = len(particle_ids)

    def __str__(self):
        return "FORCE: {} {} {}".format(Force.FTYPES[self.ftype],
                                        self.rnum,
                                        self.note)

    #  fill in the 'scores' array (len(scores) = n_particles) with the fraction of violations or violations
    def getScores(self, particles):

        scores = np.zeros(len(self.particle_ids))

        # load file with volumetric information and parameters
        vol = VolumeFile(self.volume_file)
        vol.load_file()

        center = vol.center
        origin = vol.origin
        grid   = vol.grid

        # lamina for nucleus
        if (vol.body_idx == 0) and (self.k <0):

                center = center * self.contact_range
                origin = origin * self.contact_range
                grid   = grid   * self.contact_range

        # lamian for nucleolus 
        if (vol.body_idx == 1) and (self.k >0):

                center = center / self.contact_range
                origin = origin / self.contact_range
                grid   = grid   / self.contact_range
               
        nvoxel = vol.nvoxel

        # discriminate between nucleolus and nucleus
        if vol.body_idx == False:

          for m, i in enumerate(self.particle_ids):

              idx = np.zeros(3)
              id_int = np.zeros(3).astype('int')

              p = particles[i]

              # find voxel the particle is in, e.g. (i,j,k)
              idx = (p.pos - origin)/grid
	    
              # are indexes outside of the three d volume spanned by the map box? If yes, set index to -1
              id_int[0] = -1.0 if ((idx[0] < 0) or (idx[0] >= nvoxel[0])) else  round(idx[0])
              id_int[1] = -1.0 if ((idx[1] < 0) or (idx[1] >= nvoxel[1])) else  round(idx[1])
              id_int[2] = -1.0 if ((idx[2] < 0) or (idx[2] >= nvoxel[2])) else  round(idx[2])

              # if voxel is inside the grid AND outside of the envelope

              if (id_int.all() >=0):

                   # pixel is outside the lamina and volume confinement   OR   pixel inside the lamina and lamina DamID
                   if ((vol.int_matrix[tuple(id_int)] == 0) and (self.k > 0)) or ((vol.int_matrix[tuple(id_int)] != 0) and (self.k < 0)): 
    
                      # this is the vector (point to envelope), then the vector (center to envelope)
                      scores[m] = np.linalg.norm(grid * (vol.matrice[tuple(id_int)] - id_int))/np.linalg.norm(grid)

                   # else, in the grid and inside the lamina, no violation, no issue

              # if voxel is outside the grid
              else:
    
                   if self.k > 0:    # if lamina DamID, no problem
                      print('We are somehow outside of the volumetric grid, so we could use a radial thing to be coded, max distance in grid')
                      scores[m] = np.linalg.norm(grid * idx)


        #THIS IS FOR THE NUCLEOLUS OR WHATEVER
        if vol.body_idx == True:

          for m, i in enumerate(self.particle_ids):

              idx = np.zeros(3)
              id_int = np.zeros(3).astype('int')

              p = particles[i]

              # find indexes
              idx = (p.pos - origin)/grid

              # are indexes outside of the three d volume spanned by the map box? If yes, set index to -1
              id_int[0] = -1.0 if ((idx[0] < 0) or (idx[0] >= nvoxel[0])) else round(idx[0])
              id_int[1] = -1.0 if ((idx[1] < 0) or (idx[1] >= nvoxel[1])) else round(idx[1])
              id_int[2] = -1.0 if ((idx[2] < 0) or (idx[2] >= nvoxel[2])) else round(idx[2])

              # if inside the volume grid
              if (id_int.all()>=0):

                  # if inside the body and looking for volume confinemt  OR  outsid the nucleolus but with lamina DaMID
                  if ((vol.int_matrix[tuple(id_int)] != 0) and (self.k > 0)) or ((vol.int_matrix[tuple(id_int)] == 0) and (self.k < 0)): 

                     # this is the vector (point to envelope), then the vector (center to envelope)
                     scores[m] = np.linalg.norm(grid * (vol.matrice[tuple(id_int)] - id_int))/np.linalg.norm(grid)   # p.pos - center)  # the closest to the envelope

              # if outside the grid: 
              else:
                if self.k < 0:
                    # technically there is no violation here, since the stuff is outside the nucleolus. 
                    # if the lamina DAmID case, that would be pathological, with a particle outside of the boundary which should be instead close to the lamina
                    # should that occur, let's use the actual particle coordiantes and the center of mass of the nucleolus to compute the violation score
                    scores[m] = np.linalg.norm(p.pos - center)/np.linalg.norm(grid)

        return scores

    def getViolationRatios(self, particles):
        return self.getScores(particles)
#---------

