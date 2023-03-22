from __future__ import division, print_function

import numpy as np
import h5py

from ..model.particle import Particle
from .restraint import Restraint
from ..model.forces import HarmonicUpperBound

class Imaging(Restraint):
    """
    Add IMAGING restraint to the MODEL class: the Assignment/input_file gave us a list of "target" (x,y,z) positions for a set of (phased) i loci and the structure index. 
    We'll use a SPRITE/radial FISH restraint, which uses an HarmonicUpperBound, with a tolerance to be decreased after each iteration
   
    Parameters
    ----------
    assignment_file : IMAGING activation position file
        
    radial_tolerance : float
        defining tolerance within which the position from imaging data is defined

    k (float): elastic constant for restraining
    """
    
    def __init__(self, assignment_file, radial_tolerance, struct_id, k):
        
        """ Initialize IMAGING restraint parameters and input file """

        self.radial_tolerance = radial_tolerance
        self.struct_id        = struct_id
        self.k                = k
        self.forceID          = []
        self.assignment_file  = h5py.File(assignment_file, 'r')
    #-
    
    def _apply(self, model):

        """ Apply IMAGING restraints """ 

        locus          = self.assignment_file['locus'][()]    # list of (phased) loci to restrain
        target         = self.assignment_file['target'][()]   # list of associated target distances
        assignment     = self.assignment_file['assignment'][()]   # list of structure indexes thye have to be restrained in

        here_loci      = np.where(assignment == self.struct_id)[0]   # positions in list of loci to be restrained in current structure
        #print(locus[here_loci])

        #radii          = model.getRadii()
        #coord          = model.getCoordinates()

        # loop over those loci in the Activation File that need to be restrained in structure 'struct_id' (see ImagingActivationStep.py)

        for i in here_loci:

            # add a centroid in the target position
            centroid_pos = target[i]
            #print(centroid_pos)

            centroid     = model.addParticle(centroid_pos, 0, Particle.DUMMY_STATIC) # no excluded volume

            f = model.addForce(HarmonicUpperBound( (centroid, locus[i]), float(self.radial_tolerance), self.k, note = Restraint.IMAGING))
            self.forceID.append(f)

            #print(model.forces[-5:])

