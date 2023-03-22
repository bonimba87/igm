from __future__ import division, print_function
import numpy as np
import h5py
from ..model import Model
from ..restraints.restraint import Restraint
__hms_version__ = 1

class HmsFile(h5py.File):
    """
    h5py.File like object for hms model structure files.
    """
    
    def __init__(self, *args, **kwargs):
        h5py.File.__init__(self, *args, **kwargs)
        try:
            self._version = self.attrs['version']
        except(KeyError):
            self._version = __hms_version__
            self.attrs.create('version', __hms_version__, dtype='int32')
        try:
            self._nbead = self.attrs['nbead']
        except(KeyError):
            self._nbead = 0
            self.attrs.create('nbead', 0, dtype='int32')
        try:
            self._struct_id = self.attrs['struct_id']
        except(KeyError):
            self._struct_id = 0
            self.attrs.create('struct_id', 0, dtype='int32')
            
        try:
            self._restraints_grp = self['/restraints/']
        except(KeyError):
            self._restraints_grp = self.create_group("restraints")
    #===
    def get_version(self):
        return self._version

    def get_nbead(self):
        return self._nbead
    
    def get_struct_id(self):
        return self._struct_id
    
    def get_coordinates(self, read_to_memory=True):

        '''
        Parameters
        ----------
        read_to_memory (bool) :
            If True (default), the coordinates will be read and returned
            as a numpy.ndarray. If False, a h5py dataset object will be
            returned. In the latter case, note that the datased is valid 
            only while the file is open.
        '''
        
        if read_to_memory:
            return self['coordinates'][:]
        return self['coordinates']
    
    def get_total_restraints(self):
        total = 0
        for r in self._restraints_grp.keys():
            total += self._restraints_grp[r].attrs["total"]
        return total
    
    def get_total_violations(self):
        total = 0
        for r in self._restraints_grp.keys():
            total += self._restraints_grp[r].attrs["nvios"]
        return total
    
    def set_nbead(self, n):
        self.attrs['nbead'] = self._nbead = n
    
    def set_struct_id(self, n):
        self.attrs['struct_id'] = self._struct_id = n
        
    def saveModel(self, struct_id, model):
        assert isinstance(model, Model), "Parameter has to be a Model instance"
        
        crd = model.getCoordinates()
        self.saveCoordinates(struct_id, crd)
        
    def saveCoordinates(self, struct_id, crd):
        if "coordinates" in self:
            self["coordinates"][...] = crd
        else:
            self.create_dataset("coordinates", data = crd,
                                chunks=True, compression="gzip")
        
        self.set_nbead(len(crd))
        self.set_struct_id(struct_id)
        
    def saveViolations(self, restraint, tolerance=0.05):
        assert isinstance(restraint, Restraint), "Parameter has to be a Restraint instance"
        
        violations, ratios = restraint.get_violations(tolerance)
        name = type(restraint).__name__

        if name in self._restraints_grp:
            self._restraints_grp[name]["violations"] = np.array(violations).astype("S50")
            self._restraints_grp[name]["ratios"] = np.array(ratios)
            self._restraints_grp[name].attrs["total"] = len(restraint.forceID)
            self._restraints_grp[name].attrs["nvios"] = len(ratios)
        else:
            subgrp = self._restraints_grp.create_group(name)
            subgrp.create_dataset("violations", data = np.array(violations).astype("S50"),
                                  chunks = True, compression = "gzip")
            subgrp.create_dataset("ratios", data = np.array(ratios),
                                  chunks = True, compression = "gzip")
            subgrp.attrs.create("total", len(restraint.forceID), dtype='int32')
            subgrp.attrs.create("nvios", len(ratios), dtype='int32')
        #-
        return len(ratios)

def make_absolute_path(path, basedir='.'):
    import os.path
    if os.path.isabs(path):
        return path
    return os.path.abspath( os.path.join(basedir, path) )

def h5_create_group_if_not_exist(root, groupname):
    if groupname in root:
        return root[groupname]
    else:
        return root.create_group(groupname)

def h5_create_or_replace_dataset(root, dataname, data, **kwargs):
    if dataname in root:
        root[dataname][...] = data
    else:
        root.create_dataset(dataname, data=data, **kwargs)
    return root[dataname]

class VolumeFile(object):

    def __init__(self, filename, *args, **kwargs):
          self.filename = filename

    def load_file(self):
          
        f = open(self.filename, "r")

        # read in nucleus/nuclear body switch
        self.body_idx = [int(x) for x in next(f).split()][0]

        # compute number of voxels per size
        nvoxel = np.array([int(x) for x in next(f).split()])

        # "geometric center of the grid"
        self.center = np.array([float(x) for x in next(f).split()])

        # float information about grid features (origin and grid)
        self.origin = np.array([float(x) for x in next(f).split()])
        self.grid   = np.array([float(x) for x in next(f).split()])

        # initialize empty matrices    
        self.matrice    = np.zeros((nvoxel[0], nvoxel[1], nvoxel[2],3)).astype('int')
        self.int_matrix = np.zeros((nvoxel[0], nvoxel[1], nvoxel[2])).astype('int')

        for i in range(nvoxel[0] * nvoxel[1] * nvoxel[2]):

                     # read septuplet, (i,j,k) and (EDT[i], EDT[j], EDT[k]), interior/exterior label
                     a, b, c, edt_a, edt_b, edt_c, in_out = [int(x) for x in next(f).split()]

                     # cast that into EDT matrix
                     self.matrice[a,b,c] = np.array([edt_a, edt_b, edt_c])

                     # store interior/exterior label
                     self.int_matrix[a,b,c] = in_out

        # at the end of loop , check if number of remaining lines is consistent with number of map voxels
        if (i != (nvoxel[0] * nvoxel[1] * nvoxel[2] - 1)):
                     print(nvoxel[0] * nvoxel[1] * nvoxel[2])
                     print("ACHTUNG!")
                     stop
    
        self.nvoxel = nvoxel

