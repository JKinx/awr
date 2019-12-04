import gym
import numpy as np
import random
from sklearn.ensemble import ExtraTreesRegressor
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")
import utils.Q as qnn

def is_fitted(sklearn_regressor):
    """Helper function to determine if a regression model from scikit-learn
    has ever been `fit`"""
    return hasattr(sklearn_regressor, 'n_outputs_')

class FittedQEvaluation(object):
    def __init__(self, discount, state_size, action_size, lr):
        self.discount = discount
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        
    def regressor_fitted(self):
        return self.regressor.fitted
        
    def Q(self, state_actions):
        """Return the Q function estimate of `states` for each action"""    
        if not self.regressor_fitted():
            return np.zeros(state_actions.shape[0])
        return self.regressor.predict(state_actions)

    def fit_Q(self, eval_policy, episodes, num_iters=100, discount=0.95):        
        batches = []
        batch_len = len(episodes) // 10
        init_states = []
        
        for i in range(10):
            Is = []
            S2s = []
            Rs = []
            Ts = []

            for S,A,R,S2,T in episodes[i * batch_len : (i + 1) * batch_len]:
                init_states.append(S[0])
                I = np.hstack([S, A])
                Is.append(I)
                Rs.append(R)
                S2s.append(S2)
                Ts.append(T)
            
            batches.append((np.concatenate(Is, 0), np.concatenate(Rs, 0),
                            np.concatenate(S2s, 0), np.concatenate(Ts, 0)))

        init_states = np.stack(init_states)
        values = []
        
        for i in tqdm(range(num_iters)):
            ins = []
            outs = []
            for (Is, Rs, S2s, Ts) in batches:
                ins.append(Is)
                pi_S2s = eval_policy(S2s)
                S2pi_S2s = np.hstack([S2s, pi_S2s])
                Os = Rs + discount * (Ts * self.Q(S2pi_S2s))
                outs.append(Os)
            for (Is, Os) in zip(ins, outs):
                self.regressor.fit(Is, Os) 

            S = init_states
            A = eval_policy(S)
            SA = np.hstack([S, A]) 
            values.append(self.Q(SA).mean())

        return np.mean(values[-10:]), values

    def run(self, policy, which_cost, dataset, epochs=100, g_idx=None):
        self.regressor = qnn(self.state_size, self.action_size, self.lr).cuda()
        dataset.set_cost(which_cost, idx=g_idx)
        episodes = dataset.get_episodes()

        return self.fit_Q(policy, episodes, epochs, self.discount)






        