import enum
import numpy as np
import time

class Terminate(enum.Enum):
        Null = 0
        Fail = 1

class RLPath(object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.logps = []
        self.rewards = []

        self.terminate = Terminate.Null

        self.clear()

        return
    
    def pathlength(self):
        return len(self.actions)

    def is_valid(self):
        valid = True
        l = self.pathlength()

        valid &= len(self.states) == l + 1
        valid &= len(self.actions) == l
        valid &= len(self.logps) == l
        valid &= len(self.rewards) == l
        valid |= (l == 0)

        return valid

    def check_vals(self):
        for key, vals in vars(self).items():
            if type(vals) is list and len(vals) > 0:
                for v in vals:
                    if not np.isfinite(v).all():
                        return False
        return True

    def clear(self):
        for key, vals in vars(self).items():
            if type(vals) is list:
                vals.clear()

        self.terminate = Terminate.Null
        return
    
    def discounted_sum(self, discount):
        factors = np.empty(len(self.rewards))
        factors[0] = 1
        factors[1:] = discount
        factors = np.cumprod(factors)
        return np.array(self.rewards) @ factors
        
    def calc_return(self):
        return sum(self.rewards)

    def terminated(self):
        return self.terminate == Terminate.Null

def compute_g(path):
    g0 = (np.array(path.actions)[:,0] > 0.5).astype(np.float)
    return g0.reshape(-1,1)
    
class RLPath2(object):
    def __init__(self, path, compute_g):
        self.states = np.array(path.states)
        self.actions = np.array(path.actions)
        self.rewards = np.array(path.rewards)
        self.costs = - self.rewards
        self.c = - self.rewards
        self.g = compute_g(path)

        self.terminate = Terminate.Null

        self.clear()
        return
    
    def calculate_cost(self, scale, lamb):
        self.costs = (self.c + np.dot(lamb[:-1], self.g.T)) / scale
        self.rewards = -self.costs
    
    def set_cost(self, scale, key, idx=None):
        if key == 'c':
            self.costs = self.c / scale
            self.rewards = -self.costs
        elif key == 'g':
            assert idx is not None
            # Pick the idx'th constraint
            self.costs = self.g[:,idx] / scale
            self.rewards = -self.costs 
        else:
            raise
    
    def pathlength(self):
        return len(self.actions)

    def is_valid(self):
        valid = True
        l = self.pathlength()

        valid &= len(self.states) == l + 1
        valid &= len(self.actions) == l
        valid &= len(self.rewards) == l
        valid &= len(self.costs) == l
        valid &= len(self.c) == l
        valid &= len(self.g) == l
        valid |= (l == 0)

        return valid

    def check_vals(self):
        for key, vals in vars(self).items():
            if type(vals) is list and len(vals) > 0:
                for v in vals:
                    if not np.isfinite(v).all():
                        return False
        return True

    def clear(self):
        for key, vals in vars(self).items():
            if type(vals) is list:
                vals.clear()

        self.terminate = Terminate.Null
        return
    
    def discounted_sum(self, discount, which="costs"):
        factors = np.empty(len(self.rewards))
        factors[0] = 1
        factors[1:] = discount
        factors = np.cumprod(factors)
        if which == "rewards":
            main = self.rewards
        elif which == "costs":
            main = self.costs
        else:
            raise NotImplementedError
        return main @ factors

    def calc_return(self):
        return sum(self.rewards)

    def terminated(self):
        return self.terminate == Terminate.Null