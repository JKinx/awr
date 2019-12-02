import numpy as np
import scipy.signal as signal
import os

class ExactPolicyEvaluator(object):
    def __init__(self, discount, initial_states):
        '''
        An implementation of Exact Policy Evaluation through Monte Carlo
        '''
        self.discount = discount
        self.initial_states = initial_states

    def run(self, policy):
        trial_c = []
        trial_g = []

        for state in self.initial_states:
            path = policy.agent._rollout_path(True, state)
            path = add_constraints(path)

            c = path["c"].tolist()
            g = path["g"].tolist()
            trial_c.append(c)
            trial_g.append(g)

        c = np.mean([self.discounted_sum(x, self.discount) for x in trial_c])
        g = np.mean([ [self.discounted_sum(cost, self.discount) for cost in np.array(x).T] for x in trial_g], axis=0).tolist()

        return c, g, -c






