from __future__ import division, print_function

class Restraint(object):
    OTHER = -1
    CONSECUTIVE = 0
    HIC = 1
    DAMID = 2
    FISH_RADIAL = 3
    FISH_PAIR = 4
    BARCODED_CLUSTER = 5
    ENVELOPE = 6
    EXCLUDED_VOLUME = 7
    """
    
    Restraint object, takes care of data and translate to forces in model.
    
    Also keep track of forces added and can evaluate
    
    """
    
    def __init__(self, data, args):
        self.forceID = []        
    
    def _apply_model(self, model, override=False):
        """
        Attach restraint to model
        """
        try:
            self.model
            if override:
                self.model = model
            else:
                raise(RuntimeError("Restraint alreay applyed to model! Set override to true to proceed."))
        except AttributeError:
            self.model = model
        
    def _apply(self, model, override=False):
        self._apply_model(model, override)
        
    
    def __len__(self):
        return len(self.forceID)
    
    def __getitem__(self, key):
        return self.forceID[key]
    
    def evaluate(self):
        """
        evaluate the restraint violations 
        """
        score = 0
        violations = 0
        
        for fid in self.forceID:
            s = self.model.evalForceScore(fid)
            if s > 0:
                violations += 1
            #-
            score += s
        
        return (violations, score)
    
    def get_violations(self, tolerance):
        violations = []
        ratios = []
        
        for fid in self.forceID:
            s = self.model.evalForceViolationRatio(fid)
            if s > tolerance:
                violations.append(repr(self.model.forces[fid]))
                ratios.append(s)
        
        return (violations, ratios)
        
    
        
