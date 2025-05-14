from __future__ import division, print_function
import numpy as np
import os, os.path

from math import acos, sin, cos, pi

from ..core import Step
from ..utils import HmsFile
from alabtools.analysis import HssFile

from shutil import copyfile
from tqdm import tqdm

from ..core.job_tracking import StepDB
from ..utils.log import print_progress, logger
from ..utils.files import make_absolute_path
from hashlib import md5

import h5py

class RandomInit(Step):

    def __init__(self, cfg):
        super(RandomInit, self).__init__(cfg)

        self.argument_list = list(range(self.cfg["model"]["population_size"]))

        self.tmp_extensions.append(".hms")
        self.keep_temporary_files = cfg["optimization"]["keep_temporary_files"]
        self.keep_intermediate_structures = cfg["optimization"]["keep_intermediate_structures"]

    def setup(self):
        self.tmp_file_prefix = "random"
        self.argument_list = range(self.cfg["model"]["population_size"])

    @staticmethod
    def task(struct_id, cfg, tmp_dir):
        """
        generate one random structure with territories
        """

        # the next two lines of commands are needed to initialize the radnom seed, so random numbers are reproducible at each run
        # if we woudl like to generate replicates, e.g. to generate multiple populations with different initial conditions, then the fwolloing
        # two lines must be commented out .

        #k = np.random.randint(0, 2**32)
        #np.random.seed( (k*struct_id) % (2**32) )
        
        hssfilename    = cfg["optimization"]["structure_output"]

        logger.info(hssfilename)

        logger.info(' Strcuture id = ')
        logger.info(struct_id)
        logger.info('\n')

        nucleus_radius = cfg.get("model/init_radius")
        logger.info('Radius of initialization sphere enclosing the genome = ')
        logger.info(nucleus_radius)
        
        with HssFile(hssfilename,'r') as hss:
            index = hss.index
            logger.info(hss.radii)

        crd = generate_territories(index, nucleus_radius)

        # if some spots are traced and we know their location, override the "territories" initialization
        # and interpolate the others

        if "tracing" in cfg['restraints']:

             logger.info('Interpolate between tracing data (if applicable)')

             tracing_input = h5py.File(cfg['restraints']['tracing']['assignment_file'], 'r')

             assignment     = tracing_input['assignment'][()]
             target         = tracing_input['target'][()]
             locus          = tracing_input['locus'][()]

             # identify loci that are assigned to this structure
             here_loci      = np.where(assignment == struct_id)[0]

             # rewrite coordinates of those loci that have been imaged
             for j in here_loci:         
                   crd[locus[j], :] = target[j]

             # consider all the other loci and interpolate: interpolation only works if there is something to interpolate
             for i in range(len(here_loci) - 1):

                  pin_1 = locus[here_loci[i]]
                  pin_2 = locus[here_loci[i+1]]
    
                  if pin_2 - pin_1 > 1:
                      for m, k in enumerate(range(pin_1 +1, pin_2)):
                           crd[k, :] = crd[pin_1,:] + (m+1) * (crd[pin_2,:] - crd[pin_1,:])/(pin_2 - pin_1) 

             if 'init_noise' in cfg['model']:

                  logger.info(cfg['model']['init_noise'])

                  # apply simulation noise to the initial coordinates, assuming it is applicable
                  crd = crd + cfg['model']['init_noise'] * np.random.randn(crd.shape[0], 3)
                      
        else:
             logger.info('Use random initialization of the territories')
 
        ofname = os.path.join(tmp_dir, 'random_%d.hms' % struct_id)

        with HmsFile(ofname, 'w') as hms:
            hms.saveCoordinates(struct_id, crd)

    def intermediate_name(self):
        return '.'.join([
            self.cfg["optimization"]["structure_output"],
            'randomInit'
        ])
    #-
    def reduce(self):
        """
        Collect all hms structure coordinates together to assemble a hssFile
        """

        hssfilename = self.cfg["optimization"]["structure_output"] + '.T'
        logger.info(hssfilename)

        # bonimba: using changes as Nan
        with HssFile(hssfilename, 'r+') as hss:

            n_struct = hss.nstruct
            n_beads = hss.nbead

            #iterate all structure files and
            total_restraints = 0.0
            total_violations = 0.0

            # extract coordinates and put them in matrix
            master = hss.coordinates
            logger.info('Collecting all the coordinates from all configurations....')

            for i in tqdm(range(n_struct), desc='(REDUCE)'):
                # extract info from each single .hms file
                fname = "{}_{}.hms".format(self.tmp_file_prefix, i)

                hms = HmsFile( os.path.join( self.tmp_dir, fname ) )
                crd = hms.get_coordinates()
                total_restraints += hms.get_total_restraints()
                total_violations += hms.get_total_violations()

                # replace coordinates in master matrix with coordinates from hms file
                master[:,i,:] = crd

            # master was updated, now use those to define the coordinates in the hss file
            hss.set_coordinates(master)
            #-
            if (total_violations == 0) and (total_restraints == 0):
                hss.set_violation(np.nan)
            else:
                hss.set_violation(total_violations / total_restraints)

        hss.close()

        # repack 
        PACK_SIZE = 1e6
        pack_beads = max(1, int( PACK_SIZE / n_struct / 3 ) )
        pack_beads = min(pack_beads, n_beads)

        logger.info('repacking...')
        cmd = 'h5repack -l coordinates:CHUNK={:d}x{:d}x3 {:s} {:s}'.format(
            pack_beads, n_struct, hssfilename, hssfilename + '.swap'
        )
        os.system(cmd)
        logger.info('done.')
        os.rename(hssfilename + '.swap', self.cfg.get("optimization/structure_output"))

        if self.keep_intermediate_structures:
            copyfile(
                self.cfg["optimization"]["structure_output"],
                self.intermediate_name()
            )


def uniform_sphere(R):
    """
    Generates uniformly distributed points in a sphere

    Arguments:
        R (float): radius of the sphere
    Returns:
        np.array:
            triplet of coordinates x, y, z
    """
    phi = np.random.uniform(0, 2 * pi)
    costheta = np.random.uniform(-1, 1)
    u = np.random.uniform(0, 1)

    theta = acos( costheta )
    r = R * ( u**(1./3.) )

    x = r * sin( theta) * cos( phi )
    y = r * sin( theta) * sin( phi )
    z = r * cos( theta )

    return np.array([x,y,z])



def generate_territories(index, R=5000.0):
    '''
    Creates a single random structure with chromosome territories.
    Each "territory" is a sphere with radius 0.75 times the average
    expected radius of a chromosome.
    Arguments:
        chrom : alabtools.utils.Index
            the bead index for the system.
        R : float
            radius of the cell

    Returns:
        np.array : structure coordinates
    '''

    # chromosome ends are detected when
    # the name is changed
    n_tot = len(index)
    n_chrom = len(index.chrom_sizes)

    crds = np.empty((n_tot, 3))
    # the radius of the chromosome is set as 75% of its
    # "volumetric sphere" one. This is totally arbitrary.
    # Note: using float division of py3
    chr_radii = [0.75 * R * (float(nb)/n_tot)**(1./3) for nb in index.chrom_sizes]
    crad = np.average(chr_radii)
    k = 0
    for i in range(n_chrom):
        center = uniform_sphere(R - crad)
        for j in range(index.chrom_sizes[i]):
            crds[k] = uniform_sphere(crad) + center
            k += 1

    return crds


def generate_random_in_sphere(radii, R=5000.0):
    '''
    Returns:
        np.array : structure coordinates
    '''
    return np.array([uniform_sphere(R-r) for r in radii])
