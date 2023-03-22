#
#  See Boninsegna et al, 2022, SI: Lamina DamID activtions tep
#


from __future__ import division, print_function
import numpy as np
import h5py
import os, os.path

from alabtools.analysis import HssFile

from ..core import Step
from ..utils.log import logger
from tqdm import tqdm
import shutil
from ..utils.files import make_absolute_path


try:
    # python 2 izip
    from itertools import izip as zip
except ImportError:
    pass

# dictionary for saving results of the DamID Assignment/Activation ([i, d_i, p_i])into the  damidactdist file h5
# see also preamble to the "get_damidactdist" function down in the code

damid_actdist_shape = [
    ('loc', 'int32'),
    ('dist', 'float32'),
    ('prob', 'float32')
]
damid_actdist_fmt_str = "%6d %.5f %.5f"

# --- AUXILIARY FUNCTIONS FOR COMPUTING DAMID RELATED BEAD-ENVELOPE CALCULATIONS ---#


def snormsq_sphere(x, R, r):

    """
    Compute radial distance of a bead to spherical nuclear envelope

    INPUT
	x (float), bead/locus coordinates
        R (float), radius of nuclear envelope, when spherical
	r (float), radius of bead

    OUTPUT (normalized) distance between bead surface and nuclear envelope
    			d = 1 if bead surface touches the envelope
			d < 1 otherwise
    """

    return np.sum(np.square(x), axis=1) / (R-r)**2



def snormsq_ellipse(x, semiaxes, r):

    """
    Compute radial distance of a bead to ellipsoidal nuclear envelope

    INPUT
        x (float), bead/locus coordinates
        r (float), radius of bead
        semiaxes (float, float, float), semiaxes of nuclear envelope, if ellipsoidal

    OUTPUT (normalized) distance between bead surface and nuclear envelope: x**2/(a-r)**2 + y**2/(b-r)**2 + z**2/(c-r)**2
			d = 1 if bead surface touches the envelope (this means the bead center is laying on a concentric ellipsoid
								    of semiaxes (a - r, b - r, c - r))
			d < 1 otherwise (the bead center is laying on a concentric ellipsoid with even shorter semiaxes) 
    """

    a, b, c = np.array(semiaxes) - r
    sq = np.square(x)
    return sq[:, 0]/(a**2) + sq[:, 1]/(b**2) + sq[:, 2]/(c**2)

# functions (sphere or ellipsoid) are put together into "snormsq" which takes the shape as an input
snormsq = {
    'sphere': snormsq_sphere,
    'ellipsoid': snormsq_ellipse
}



class DamidActivationDistanceStep(Step):

    def __init__(self, cfg):

        """ The value of DAMID sigma to be used this time around is computed and stored """

        # prepare the list of DAMID sigmas in the "runtime" status, unless already there
        if 'sigma_list' not in cfg.get("runtime/DamID"):
            cfg["runtime"]["DamID"]["sigma_list"] = cfg.get("restraints/DamID/sigma_list")[:]

        # compute current Damid sigma and save that to "runtime" status
        if "sigma" not in cfg.get("runtime/DamID"):
            cfg["runtime"]["DamID"]["sigma"] = cfg.get("runtime/DamID/sigma_list").pop(0)

        # parameter "iter_corr_knob" to control whether using iterative correction or not
        if "iter_corr_knob" not in cfg.get("runtime/DamID"):
            cfg["runtime"]["DamID"]["iter_corr_knob"] = cfg.get("optimization/iter_corr_knob")

        if cfg.get('runtime/DamID/iter_corr_knob') == 1:
            logger.info('Using iterative correction (standard choice)')
        else:
            logger.info('Not using iterative correction (number of contacts may blow up!)')

        super(DamidActivationDistanceStep, self).__init__(cfg)

    def name(self):

        """ This is printed to logger, and indicates that the DAmid activation step has started """

        s = 'DamidActivationDistanceStep (sigma={:.2f}%, iter={:s})'
        return s.format(
            self.cfg.get('runtime/DamID/sigma', -1) * 100.0,
            str( self.cfg.get('runtime/opt_iter', 'N/A') )
        )


    def setup(self):

        """ Prepare parameters, select those loci with a probabiliry larger than sigma, read in DAMID input file and preprocess by spitting that into batches, produce in.npy files """

        # read in damid sigma activation, and the filename containing raw damid data
        sigma         = self.cfg.get("runtime/DamID/sigma")
        input_profile = self.cfg.get("restraints/DamID/input_profile")

        last_damidactdist_file = self.cfg.get('runtime/DamID').get("damid_actdist_file", None)
        logger.info(last_damidactdist_file)

        batch_size = self.cfg.get('restraints/DamID/batch_size', 100)

        self.tmp_extensions = [".npy", ".tmp"]
        self.set_tmp_path()
        self.keep_temporary_files = self.cfg.get("restraints/DamID/keep_temporary_files", False)

        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        # load txt file containing experimental DAMID data, as per configuration json file
        profile = np.loadtxt(input_profile, dtype='float32')

        # collect those loci whose probability of contact with lamina falls within the threshold "sigma" of the current iteration
        mask    = profile >= sigma
        ii      = np.where(mask)[0]
        p_exp   = profile[mask]

        # read in last_damid_actdist file from previous iteration
        if last_damidactdist_file is not None:
      
            with h5py.File(last_damidactdist_file) as h5f:        
                last_prob = {int(i) : p for i, p in zip(h5f["loc"], h5f["prob"])}  #-LB this is not correct, because we are selecting the same haploid copy multiple times, this is not consistent with the numbering in "ii" at all
        else:
                last_prob = {}


        # LB this part is coded differently form the HiC, I gues it is easier here
        # split the full matrix into batches of size batch_size...each chunk is then saved to a temporary damid.in.npy file
        n_args_batches = len(ii) // batch_size
        if len(ii) % batch_size != 0:
            n_args_batches += 1

        # generate different in files from the input: each file contains one batch of input data
        for b in range(n_args_batches):
            start = b * batch_size
            end = min((b+1) * batch_size, len(ii))
            params = np.array(
                [
                    ( ii[k], p_exp[k], last_prob.get(ii[k], 0.) )
                    for k in range(start, end)
                ],
                dtype=np.float32
            )
            # saving step
            fname = os.path.join(self.tmp_dir, '%d.damid.in.npy' % b)
            np.save(fname, params)

        self.argument_list = range(n_args_batches)

    @staticmethod
    def task(batch_id, cfg, tmp_dir):

        """ Compute damid activation distances for batch of loci identified by parameter batch_id, and save to "out.npy" files """

        nucleus_parameters = None
        shape = cfg.get('model/restraints/envelope/nucleus_shape')

        if shape == 'sphere':
            nucleus_parameters = cfg.get('model/restraints/envelope/nucleus_radius')
        elif shape == 'ellipsoid':
            nucleus_parameters = cfg.get('model/restraints/envelope/nucleus_semiaxes')
        else:
            raise NotImplementedError('DamID restraint for shape %s has not been implemented yet.' % shape)


        # read in the iterative correction knob
        it_corr = cfg.get('runtime/DamID/iter_corr_knob')
 
        with HssFile(cfg.get("optimization/structure_output"), 'r') as hss:

            # read params from temporary damid.in.npy files
            fname = os.path.join(tmp_dir, '%d.damid.in.npy' % batch_id)
            params = np.load(fname)

            # compute DamID  activdation distance into corresponding output to save to out.tmp file
            results = []
            for I, p_exp, plast in params:
                res = get_damid_actdist(
                    int(I), p_exp, plast, hss, it_corr,
                    contact_range=cfg.get('restraints/DamID/contact_range', 0.05),
                    shape=shape,
                    nucleus_param=nucleus_parameters
                )
                results += res #(i, damid_actdist_i, p_i)
            #-

        # save output for this chunk to file, using the format specified by string 'damid_actdist_fmt_str'
        fname = os.path.join(tmp_dir, '%d.out.tmp' % batch_id)
        with open(fname, 'w') as f:
            f.write('\n'.join([damid_actdist_fmt_str % x for x in results]))


    def reduce(self):

        """ Concatenate data from all batches into a single hdf5 damid_actdist file """

        # create filename
        damid_actdist_file = os.path.join(self.tmp_dir, "damid_actdist.hdf5")
        last_damidactdist_file = self.cfg['runtime']['DamID'].get("damid_actdist_file", None)   # stored from the previous iteration
 
        logger.info(last_damidactdist_file)

        loc = []
        dist = []
        prob = []

        # (also see 'reduce' step in ActivationDistanceStep.py) Read in all .out.tmp files and concatenate all data into a single
        # 'damid_actdist_file' file, of type h5df (see 'create-dataset attributes)
        
        # concatenate...
        for i in tqdm(self.argument_list, desc='(REDUCE)'):
            
            fname = os.path.join(self.tmp_dir, '%d.out.tmp' % i)
            partial_damid_actdist = np.genfromtxt( fname, dtype=damid_actdist_shape )
            
            if partial_damid_actdist.ndim == 0:
                partial_damid_actdist = np.array([partial_damid_actdist], dtype=damid_actdist_shape)

            loc.append( partial_damid_actdist['loc'])
            dist.append(partial_damid_actdist['dist'])
            prob.append(partial_damid_actdist['prob'])

        # suffix
        additional_data = []
        if 'DamID' in self.cfg['runtime']:
            additional_data.append(
                'DamID_{:.2f}'.format(self.cfg['runtime']['DamID']['sigma']))
        if 'opt_iter' in self.cfg['runtime']:
            additional_data.append(
                'iter_{}'.format(
                    self.cfg['runtime']['opt_iter']-1    #-LB need to reduce iteration number by 1 for consistency
                )
            )
        
        tmp_damid_actdist_file = damid_actdist_file + '.tmp'

        #... write to tmp damid actdist file
        with h5py.File(tmp_damid_actdist_file, "w") as h5f:
            h5f.create_dataset("loc", data=np.concatenate(loc))
            h5f.create_dataset("dist", data=np.concatenate(dist))
            h5f.create_dataset("prob", data=np.concatenate(prob))

        swapfile = os.path.realpath('.'.join([damid_actdist_file, ] + additional_data))
        logger.info(swapfile)

        if last_damidactdist_file is not None:
             shutil.move(last_damidactdist_file, swapfile)           # rename "last_damidactdist_file" as "swapfile"
       
        shutil.move(tmp_damid_actdist_file, damid_actdist_file)

        # ... update runtime parameter for next iteration/sigma value
        self.cfg['runtime']['DamID']["damid_actdist_file"] = damid_actdist_file



    def skip(self):
        """
        Fix the dictionary values when DamiD Activation Step already completed: need to have everuthing in place for the next M-step
        (file names, directories, runtime stuff)
        """

        self.tmp_dir = make_absolute_path(
            self.cfg.get('restraints/DamID/tmp_dir', 'damid_actdist'),
            self.cfg.get('parameters/tmp_dir')
        )
        self.damid_actdist_file = os.path.join(self.tmp_dir, "damid_actdist.hdf5")
        self.cfg['runtime']['DamID']["damid_actdist_file"] = self.damid_actdist_file


    # -lB need to figure this function out --- #
    def set_tmp_path(self):
        """ Auxiliary function to play around with paths and directories """
        curr_cfg = self.cfg['restraints']['DamID']
        damid_tmp_dir = curr_cfg.get('damid_actdist_dir', 'damid_actdist')

        if os.path.isabs(damid_tmp_dir):
            self.tmp_dir = damid_tmp_dir
        else:
            self.tmp_dir = os.path.join( self.cfg['parameters']['tmp_dir'], damid_tmp_dir )
            self.tmp_dir = os.path.abspath(self.tmp_dir)


def cleanProbability(pij, pexist):

    """ Clean probability values based on the number of restraints already applied to structures 
	This is the iterative refinement, see also Supp Info and "ActivationDistance.py" file
    """

    if pexist < 1:
        pclean = (pij - pexist) / (1.0 - pexist)
    else:
        pclean = pij
    return max(0, pclean)



def get_damid_actdist(I, p_exp, plast, hss, it_corr, contact_range=0.05, shape="sphere", nucleus_param=5000.0):
    """
    Serial function to compute the damid activation distance for a locus.

    Parameters
    ----------
        I : int
            index of (unphased) locus
        p_exp : float
            experimental contact probability with the lamina (from input DamID file, upon preprocessing of raw data)
        plast : float
            the last refined probability (using p_exp and the iterative refinement)
        hss : alabtools.analysis.HssFile
            file containing coordinates
        contact_range : float
            built-in tolerance used to define a contact with the lamina (see Boninsegna et al, 2022, SI)
        it_corr: boolean
            (1) Use iterative refinement and correct p_exp, based on p_now (see later) and p_last
            (0) No iterative refinement 
        shape : str
            shape of the envelope
        nucleus_param : variable
            parameters for the envelope (as of now, sphere or ellipsoid semiaxes)

    Returns
    -------
        list of (i, actdist_i, p_i) arrays

        i (int): the (possibily diploid) locus index
        actdist_i (float): the activation distance
        p_i (float): the (possibly) corrected (upon iterative refinement) probability

        If I = (i,i'), then we'll have [[i, actdist_I, p_I],[i', actdist_I, p_I]]

    """

    # import here in case is executed on a remote machine
    import numpy as np

    n_struct = hss.get_nstruct()
    copy_index = hss.get_index().copy_index

    ii = copy_index[I]
    n_copies = len(ii)

    r = hss.get_radii()[ ii[0] ]

    d_sq = np.empty(n_copies*n_struct)

    # compute distribution of distances d(i, LAMINA) and d(i', LAMINA)
    for i in range(n_copies):
        x = hss.get_bead_crd(ii[ i ])
        R = np.array(nucleus_param)*(1 - contact_range)
        d_sq[ i*n_struct:(i+1)*n_struct ] = snormsq[shape](x, R, r)

    # this defines the contact with the lamina, se SI
    rcutsq = 1.0
    
    # sort ditances to lamina in decreasing order, from farthest to closest
    d_sq[::-1].sort()

    contact_count = np.count_nonzero(d_sq >= rcutsq)
    
    # compute probability of contact with lamina in current population for locus i
    pnow = float(contact_count) / (n_struct * n_copies)

    # iterative refinement: if applicable, correct p_exp using information from p_last (last Assignment) and p_now (last Modeling)
    if it_corr == 1:

         t = cleanProbability(pnow, plast)
         p = cleanProbability(p_exp, t)
    else:
         p = p_exp
         

    # set a super large actdist for the case p = 0
    activation_distance = 2

    if p>0:
        # determine index pointing to p-th quantile, which defines the activation distance
        o = min(n_copies * n_struct - 1,
                int( round(n_copies * n_struct * p ) ) )

        # identify the DamID activation distance**2 as the o-th quantile, then take the sqrt
        activation_distance = np.sqrt(d_sq[o])

    return [ (i, activation_distance, p) for i in ii ]

