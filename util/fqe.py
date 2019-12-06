import gym
import numpy as np
import random
from sklearn.ensemble import ExtraTreesRegressor
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")
from util.qnn import Q
from tqdm import tqdm
from copy import deepcopy

def is_fitted(sklearn_regressor):
    """Helper function to determine if a regression model from scikit-learn
    has ever been `fit`"""
    return hasattr(sklearn_regressor, 'n_outputs_')

class FittedQEvaluation(object):
    def __init__(self, discount, state_size, action_size, lr, update_every=1, num_batches=5, epochs=100):
        self.discount = discount
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.update_every = update_every
        self.num_batches = num_batches
        self.epochs = epochs
        self.best_regressor = None
        
    def regressor_fitted(self):
        return self.regressor.fitted
        
    def Q(self, state_actions, train=False):
        """Return the Q function estimate of `states` for each action"""  
        if True:
            if not self.regressor_fitted():
                return np.zeros(state_actions.shape[0])
            return self.regressor.predict(state_actions)
        else:
            assert self.best_regressor is not None
            return self.best_regressor.predict(state_actions)

    def fit_Q(self, eval_policy, episodes, init_states, num_iters=100, discount=0.95):        
        batches = []
        batch_len = len(episodes) // self.num_batches
        
        for i in range(self.num_batches):
            Is = []
            S2s = []
            Rs = []
            Ts = []

            for S,A,R,S2,T in episodes[i * batch_len : (i + 1) * batch_len]:
                I = np.hstack([S, A])
                Is.append(I)
                Rs.append(R)
                S2s.append(S2)
                Ts.append(T)
            
            batches.append((np.concatenate(Is, 0), np.concatenate(Rs, 0),
                            np.concatenate(S2s, 0), np.concatenate(Ts, 0)))

        values = []
        losses = []
        best_loss = float("inf")
        
        for i in tqdm(range(num_iters)):
            ins = []
            outs = []
            loss = []
            for (Is, Rs, S2s, Ts) in batches:
                ins.append(Is)
                pi_S2s = eval_policy(S2s)
                S2pi_S2s = np.hstack([S2s, pi_S2s])
                Os = Rs + discount * (Ts * self.Q(S2pi_S2s, True))
                outs.append(Os)
            for (Is, Os) in zip(ins, outs):
                loss.append((self.regressor.fit(Is, Os)).cpu().item())
            losses.append(np.array(loss).mean())
            
#             if i % self.update_every == 0 and losses[-1] < best_loss:
#                 best_loss = losses[-1]
#                 self.best_regressor = deepcopy(self.regressor)

            S = init_states
            A = eval_policy(S)
            SA = np.hstack([S, A]) 
            values.append(self.Q(SA).mean())

        return np.mean(values[-10:]), values, losses

    def run(self, policy, which_cost, dataset, init_states, g_idx=None):
        self.regressor = Q(self.state_size, self.action_size, self.lr).cuda()
        dataset.set_cost(which_cost, idx=g_idx)
        episodes = dataset.get_episodes()

        return self.fit_Q(policy, episodes, init_states, self.epochs, self.discount)






        