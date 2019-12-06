import argparse
import gym
import numpy as np
import os
import sys
import tensorflow as tf

import pickle

from util.replay_buffer import ReplayBuffer
from util.rl_path import RLPath2
from util.rl_path import compute_g
from tqdm import tqdm

from util.best_response import BestResponse
import awr_configs
from util.exponentiated_gradient import ExponentiatedGradient

from util.fqe import FittedQEvaluation

from util.exact_policy_evaluation import ExactPolicyEvaluator

from util.optimization_problem import OptProblem

from argparse import Namespace

def main(args):
    constraints = np.array([1,0])
    
    train_data = pickle.load(open("paths.5.half.pkl", "rb"))
    train_data2 = [RLPath2(path, compute_g) for path in tqdm(train_data)]
    dataset = ReplayBuffer(10000000)
    for path in tqdm(train_data2):
        dataset.store(path)
        
    init_states = pickle.load(open("init_states606.pkl", "rb"))
    
    args = {
        "env" : "LunarLanderContinuous-v2",
        "train" : True,
        "test" : False,
        "max_iter" : 2, 
        "test_episodes" : 1,
        "output_dir" : "output",
        "output_iters" : 10,
        "gpu" : "0",
        "visualize" : False
    }
    args = Namespace(**args)
    best_response_algorithm = BestResponse(args)
    
    lambda_bound = 30
    eta = 1
    starting_lambda = [1, 100]
    online_convex_algorithm = ExponentiatedGradient(
        lambda_bound, len(constraints),
        eta=eta, starting_lambda=starting_lambda)
    
    discount = 0.95
    state_size = 8
    action_size = 2
    lr = 0.001
    fqe_epochs = 100
    fqe_batches = 3
    fitted_off_policy_evaluation_algorithm = FittedQEvaluation(discount, state_size, action_size, 
                                                               lr, epochs=fqe_epochs, batches=fqe_batches)
    
    init_seed = 606
    num_paths = 2
    exact_policy_algorithm = ExactPolicyEvaluator(discount, init_seed, num_paths, compute_g)
    
    
    problem = OptProblem(constraints, 
                         dataset, 
                         init_states, 
                         best_response_algorithm, 
                         online_convex_algorithm, 
                         fitted_off_policy_evaluation_algorithm, 
                         exact_policy_algorithm, 
                         lambda_bound, 
                         max_iterations=10)

    lambdas = []
    policies = []

    iteration = 0
    while not problem.is_over():
        iteration += 1
        for i in range(1):

            print('*' * 20)
            print('Iteration %s, %s' % (iteration, i))
            if len(lambdas) == 0:
                # first iteration
                lambdas.append(online_convex_algorithm.get())
                print('lambda_{0}_{2} = {1}'.format(iteration, lambdas[-1], i))
            else:
                # all other iterations
                lambda_t = problem.online_algo()
                lambdas.append(lambda_t)
                print('lambda_{0}_{3} = online-algo(pi_{1}_{3}) = {2}'.format(iteration, iteration-1, lambdas[-1], i))

            lambda_t = lambdas[-1]
            pi_t = problem.best_response(lambda_t)
            values = []

            # policies.append(pi_t)
            problem.update(pi_t, values, iteration)  # Evaluate C(pi_t), G(pi_t) and save


if __name__ == '__main__':
    main(sys.argv)
