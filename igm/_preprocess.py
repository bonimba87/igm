from __future__ import division, print_function

from alabtools.utils import Genome, Index, make_diploid, make_multiploid
from alabtools.analysis import HssFile, COORD_DTYPE
from alabtools import Contactmatrix
import os.path
import json
from six import string_types, raise_from
import numpy as np
import os
from shutil import copyfile
from .utils.log import logger

#===prepare genome and index instances
def PrepareGenomeIndex(cfg):

    """ Prepare genome and index instances

	cfg: configuration file object, as from igm.Config('####.json')

        - Generate the appropriate (diploid, haploid, male, female) genome/index objects (as in alabtools) 
    """
   
    # in the config.json file, "genome" has the attributes "assembly", "ploidy", "usechr" and "segmentation" 
    gcfg = cfg['genome']
    if 'usechr' not in gcfg:
        gcfg['usechr'] = ['#', 'X', 'Y']

    # types and lengths of chromosomes are read in from the assemblies available in alabtools
    genome = Genome(gcfg['assembly'], usechr=gcfg['usechr'])

    # segmentation would here be given as a string, so some magic is needed to convert that into int
    if isinstance(gcfg['segmentation'], string_types):
        if os.path.isfile(gcfg['segmentation']):
            index = Index(gcfg['segmentation'], genome=genome)
        else:
            try:
                gcfg['segmentation'] = int(gcfg['segmentation'])
            except ValueError:
                raise_from(ValueError('Invalid segmentation value (either the file is not found or it is not an integer)'), None)

    # if "segmentation" is an interger value, then generate the appropriate binning using Alabtools
    if isinstance(gcfg['segmentation'], int):
        index = genome.bininfo(gcfg['segmentation'])

    # check if dictionary entry for ploidy are incorrect
    if (not isinstance(gcfg['ploidy'], string_types)) and (not isinstance(gcfg['ploidy'], dict)):
        raise ValueError('Invalid ploidy value')

    # if ploidy is correct and is either "diploid" or "male"...
    if isinstance(gcfg['ploidy'], string_types):
        if gcfg['ploidy'] == 'diploid':
            index = make_diploid(index)    # turn haploid into diploid
        elif gcfg['ploidy'] == 'haploid':
            pass                           # no need to modify the "index" object
        elif gcfg['ploidy'] == 'male':
            gcfg['ploidy'] = {
                '#': 2,
                'X': 1,
                'Y': 1
            }
        else:
            gcfg['ploidy'] = json.parse(gcfg['ploidy'])

    # what's going on here?
    if isinstance(gcfg['ploidy'], dict):
        chrom_ids = []
        chrom_mult = []
        for c in sorted(gcfg['ploidy'].keys()):
            if c == '#':
                autosomes = [ i for i, x in enumerate(genome.chroms)
                              if x[-1].isdigit() ]        # find the autosomes
                chrom_ids += autosomes                    # add autosomes to chrom number list
                chrom_mult += [ gcfg['ploidy'][c] ] * len(autosomes)   # create list specifying number of copies for each autosomes, i.e. [2,2,2,2,...]
            else:
                if isinstance(c, string_types):
                    cn = genome.chroms.tolist().index('chr%s' % c)    # chrom_id (int!!!) for sex chromosomes
                elif isinstance(c, int):
                    cn = c
                else:
                    raise ValueError('Invalid chromosome ID in ploidy: %s' % repr(cn))
                chrom_ids += [ cn ]
                chrom_mult += [ gcfg['ploidy'][c] ]

        # this function returned the correctly parsed "genome" and "index" objects (haploid, female diploid or male diploid), with the correct number of copies for each chromosome 
        index = make_multiploid(index, chrom_ids, chrom_mult)

    return genome, index

def prepareHss(fname, nbead, nstruct, genome, index, radii, nucleus_shape='sphere', nucleus_parameters=5000.0, nucleus_volume=0, coord_chunks=None):

    """ Prepare the population file (.hss extension) """

    with HssFile(fname, 'w') as hss:

        #put everything into hssFile
        hss.set_nbead(nbead)
        hss.set_nstruct(nstruct)
        hss.set_genome(genome)
        hss.set_index(index)
        hss.set_radii(radii)
        if coord_chunks is None:
            hss.set_coordinates(np.zeros((nbead,nstruct,3)))
        else:
            hss.create_dataset('coordinates', shape=(nbead, nstruct, 3), dtype=COORD_DTYPE, chunks=coord_chunks)

        env = hss.create_group('envelope')
        env.create_dataset('shape', data=nucleus_shape)
        env.create_dataset('volume', data=nucleus_volume)
        env.create_dataset('params', data=nucleus_parameters)

def Preprocess(cfg):

    #Generate genome, index objects (as in alabtools)
    genome, index = PrepareGenomeIndex(cfg)

    logger.info(genome)
    logger.info(index.chrom)

    # number of structures in population, number of beads
    nstruct = cfg['model']['population_size']
    logger.info('Population size = ' + str(nstruct))
    nbead = len(index)

    _43pi = 4./3*np.pi

    # compute volume of the nucleus
    nucleus_shape = cfg.get('model/restraints/envelope/nucleus_shape')

    if  nucleus_shape == 'sphere':
        nucleus_radius     = cfg.get('model/restraints/envelope/nucleus_radius')
        nucleus_volume     = _43pi * (nucleus_radius**3)
        nucleus_parameters = nucleus_radius

    elif nucleus_shape == 'ellipsoid':
         sx = cfg.get('model/restraints/envelope/nucleus_semiaxes')
         nucleus_volume     = _43pi * sx[0] * sx[1] * sx[2]
         nucleus_parameters = sx

    elif nucleus_shape == 'exp_map':      # account for a random nucleus, initialize on a sphere
         nucleus_radius     = 4000         # this is the effective radius for the volume map/w nucleolus
         nucleus_volume     = _43pi * (nucleus_radius**3)
         nucleus_parameters = nucleus_radius

         logger.info('Initialization reference volume to determine radius...')
         logger.info(nucleus_radius)

    else:
            raise NotImplementedError(
                "Cannot compute volume for shape %s" % cfg.get('model/restraints/envelope/nucleus_shape')
            )

    if 'occupancy' in cfg['model']:
        logger.info('Using genome occupancy to extract radii = ')
        occupancy = cfg['model']['occupancy']
        logger.info(occupancy)

        # compute volume per basepair based on nuclear volume and chosen segmentation; hence, compute bead radius
        rho = occupancy * nucleus_volume / (sum(index.end - index.start))
        bp_sizes = index.end - index.start
        sphere_volumes = [rho * s for s in bp_sizes]
        radii = ( np.array(sphere_volumes) / _43pi )**(1./3)

    else:

        logger.info('There is no occupancy value in the configuration file')

        if 'bead_radius' in cfg['model']:      
            radii = np.array([cfg['model']['bead_radius'] for s in range(len(index))])
        else:
            raise NotImplementedError("There is no prescription for initializing the bead radii")


    # prepare Hss
    if not os.path.isfile(cfg['optimization']['structure_output']):
        prepareHss(cfg['optimization']['structure_output'], nbead, nstruct, genome, index, radii, nucleus_shape, nucleus_parameters, nucleus_volume)

    # now create a temporary struct-major file for runtime use
    if not os.path.isfile(cfg['optimization']['structure_output'] + '.T'):

        PACK_SIZE = 1e6
        pack_struct = max(1, int( PACK_SIZE / nbead / 3 ) )
        pack_struct = min(pack_struct, nstruct)

        prepareHss(cfg['optimization']['structure_output'] + '.T' , nbead,
                   nstruct, genome, index, radii, nucleus_shape,
                   nucleus_parameters, nucleus_volume,
                   coord_chunks=(nbead, pack_struct, 3))

    # if "tmp_dir" folder does not exist, create that
    if not os.path.exists(cfg['parameters']['tmp_dir']):
        os.makedirs(cfg['parameters']['tmp_dir'])

    # if we have a Hi-C probability matrix, use it to determine the consecutive
    # beads distances
    pbs = cfg.get('model/restraints/polymer/polymer_bonds_style')
    logger.info(pbs)
 
    if pbs == 'hic':
        if "Hi-C" not in cfg['restraints']:
            raise RuntimeError('Hi-C restraints specifications are missing in the cfg, but "polymer_bond_style" is set to "hic"')

        # read the HiC matrix and get the first upper diagonal (i, i+1).
        m = Contactmatrix(cfg['restraints']['Hi-C']['input_matrix']).matrix
        cps = np.zeros(len(index) - 1)
        for i in range(m.shape[0] - 1):
            f = m[i][i+1]     # this value is exactly as is from the HiC input matrix
            for j in index.copy_index[i]:
                cps[j] = f

        cpfname = os.path.join(cfg['parameters']['tmp_dir'], 'consecutive_contacts.npy')
        np.save(cpfname, cps)      # save file with the consecutive contact probabilities

        logger.info('Array of consecutive contacts just saved. IGM ready to go on...')

        # update 'runtime' entry of the configuration object, where we track parameters along the optimization
        cfg['runtime']['consecutive_contact_probabilities'] = cpfname







