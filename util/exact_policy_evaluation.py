import numpy as np
import scipy.signal as signal
import os
from util.rl_path import RLPath2

class ExactPolicyEvaluator(object):
    def __init__(self, discount, init_seed, num_paths, compute_g):
        '''
        An implementation of Exact Policy Evaluation through Monte Carlo
        '''
        self.discount = discount
        self.init_seed = init_seed
        self.num_paths = num_paths
        self.compute_g = compute_g

    def run(self, policy, scale_c, scale_g):
        trial_c = []
        trial_g = []
        
        policy.agent._env.seed(self.init_seed)
        for _ in range(self.num_paths):
            path = policy.agent._rollout_path(True)
            path = RLPath2(path, self.compute_g)

            c = (path.c / scale_c).tolist()
            g = (path.g / scale_g).tolist()
            trial_c.append(c)
            trial_g.append(g)

        c = np.mean([self.discounted_sum(x, self.discount) for x in trial_c])
        g = np.mean([ [self.discounted_sum(cost, self.discount) for cost in np.array(x).T] for x in trial_g], axis=0).tolist()

        return c, g, -c
    
    def discounted_sum(self, x, discount):
        factors = np.empty(len(x))
        factors[0] = 1
        factors[1:] = discount
        factors = np.cumprod(factors)
        return x @ factors






