from __future__ import division, print_function
import numpy as np
import h5py
import os
import os.path
import shutil
from tqdm import tqdm

from alabtools.analysis import HssFile

from ..core import Step
from ..utils.log import logger

try:
    # python 2 izip
    from itertools import izip as zip
except ImportError:
    pass

# dictionaries for the activation distance files
fish_restr = {'bead': [], 'nn_dist' : []}

# Compute all (i, i+1) polymer distances in the population
def get_polymer_dists(i,  crd):
    
    """    Compute all diploid distances between (pair) annotations  """

    dists = np.linalg.norm(crd[i, :, :] - crd[i+1, :, :], axis = 1)
    assert len(dists) == crd.shape[1]

    sorting_idx = np.argsort(np.argsort(dists))
    return dists, sorting_idx        # compute (ii, ii+1) distances in all S structures


class PolymerAssignmentStep(Step):

    def __init__(self, cfg):

        # inherit all attributes and methods of parent class "Step"
        super(PolymerAssignmentStep, self).__init__(cfg)


    def name(self):

        """ This is printed to logger, and indicates that the Polymer  assignment step has started  """

        s = 'PolymerAssignmentStep (iter={:s})'
        return s.format(
            str( self.cfg.get('runtime/opt_iter', 'N/A') )
        )


    def setup(self):

        """ Prepare parameters, prepare batches to avoid store everything in memory at once  """

        self.tmp_extensions = [".npz"]
        self.set_tmp_path()

        #self.keep_temporary_files = self.cfg.get("restraints/FISH/keep_temporary_files", False)

        # create folder
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        # read in population file and extract coordinates
        hss = HssFile(self.cfg.get("optimization/structure_output"), 'r')
        crd = hss['coordinates']


        # batch size and initialize empty batch list
        batch_size = 1000
        batches = []

        # range(0, n_beads - 1), so that the last bead in the genome is not included
        for i in range(0, crd.shape[0] - 1, batch_size):
               if i + batch_size > crd.shape[0]:
                     batches.append((len(batches), range(i, crd.shape[0]-1)))
               else:
                     batches.append((len(batches), range(i, i+batch_size)))

        self.argument_list = batches

    @staticmethod
    def task(batch, cfg, tmp_dir):

        polymer_restr = {'bead':[], 'nn_dist': []}
        
        # read in population file and extract coordinates
        hss = HssFile(cfg.get("optimization/structure_output"), 'r')
        crd = hss['coordinates']
    
        # from haploid to multiploid representation (if needed)
        copy_index = hss.index.copy_index

        assert(crd.shape[2] == 3)    # check consistency in array sizes
  
        # load input file with distributions (maybe check if number of distances = number of poluation structures)
        with h5py.File(cfg.get('restraints/polymer/polymer_file'), 'r') as ftf:

            edges     = ftf['bin_edges'][()]
            dist_prob = ftf['probability'][()] 

        bin_size = edges[1] - edges[0]

        # extract batchID and the loci in the batch        
        batch_id, entries = batch
    
        # loop over loci
        for i in entries:

                # draw a number of samples from distribution that is equal to number of structures; pick distance in bin center!
                sampled_distances = np.sort(np.random.choice(edges, crd.shape[1], p = dist_prob))
                
                # compute all (i, i+1) distances across the population and find their position in sorted list/
                _ , sorting_idx  = get_polymer_dists(i, crd)

                # assign to n-th distance in population the n-th experimental distance from distribution sampling
                polymer_restr['bead'].append(i)
                polymer_restr['nn_dist'].append(sampled_distances[sorting_idx]) 


        # intermediate file, create database, save npz file with features of the chunk (some entries might be empty lists)
        aux_file = os.path.join(tmp_dir, 'tmp.%d.polymer.npz' % batch_id)
        np.savez(aux_file,  nn_dist= np.array(polymer_restr['nn_dist']))      


    def reduce(self):

        """ Concatenate data from all batches (npz files) into a single hdf5 polymer_actdist file """

        # build suffix to append to actdist file (code does not overwrite actdist files)
        additional_data = []
        if 'opt_iter' in self.cfg['runtime']:
            additional_data.append(
                'iter_{}'.format(
                    self.cfg['runtime']['opt_iter']
                )
            )

        # create filename
        polymer_assignment_file = os.path.join(self.tmp_dir, self.cfg['restraints']['polymer']['assignment_file'])
        last_actdist_file = self.cfg['runtime']['polymer'].get("assignment_file", None)   # is there a current ass file?
 
        beads   = []
        nn_dist = []

        # concatenate: loop over chunks
        for batch_id, entries in self.argument_list:
            
            # load auxiliary files and fill in lists
            auxiliary_file = os.path.join(self.tmp_dir, 'tmp.%d.polymer.npz' % batch_id)
 
            t = np.load(auxiliary_file, allow_pickle=True)

            nn_dist.append(t['nn_dist'])
            beads.append(entries)

        tmp_assignment_file = polymer_assignment_file + '.tmp'

        logger.info(polymer_assignment_file)
        #logger.info(last_actdist_file)
        logger.info(tmp_assignment_file)

        # write fish actdist file for current iteration: need to distinguish if pairs or not, things are out
        with h5py.File(tmp_assignment_file, "w") as o5f:

             o5f.create_dataset('loci',      data =   np.concatenate((beads)),  dtype='i4')
             o5f.create_dataset('nn_dist',   data =  np.concatenate((nn_dist)),  dtype='f4')

        # save file temporary with appends
        swapfile = os.path.realpath('.'.join([polymer_assignment_file, ] + additional_data))
        if last_actdist_file is not None:
            shutil.move(last_actdist_file, swapfile)
        shutil.move(tmp_assignment_file, polymer_assignment_file)


        # ... update runtime parameter for next iteration/sigma value
        self.cfg['runtime']['polymer']["assignment_file"] = polymer_assignment_file


    def skip(self):
 
        """ Fix the dictionary values when already completed """

        self.set_tmp_path()

        # place file into the tmp_path folder
        polymer_assignment_file = os.path.join(self.tmp_dir, "polymer_assignment.h5")
        self.cfg['runtime']['polymer']["assignment_file"] = polymer_assignment_file


    def set_tmp_path(self):

        """ Auxiliary function to play around with paths and directories """

        curr_cfg = self.cfg['runtime']['polymer']
        poly_tmp_dir = curr_cfg.get('tmp_dir', 'poly_actdist')

        if os.path.isabs(poly_tmp_dir):
            self.tmp_dir = poly_tmp_dir
        else:
            self.tmp_dir = os.path.join( self.cfg['parameters']['tmp_dir'], poly_tmp_dir )
            self.tmp_dir = os.path.abspath(self.tmp_dir)

#----
