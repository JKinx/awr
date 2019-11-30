import argparse
import gym
import numpy as np
import os
import sys
import tensorflow as tf

import awr_configs
import learning.awr_agent as awr_agent
from util.optimization_problem import OptProblem
from util.exponentiated_gradient import ExponentiatedGradient


def main(args):

    # TODO: setup problem as well as algorithms
    online_convex_algorithm = ExponentiatedGradient(
        lambda_bound, len(constraints),
        eta=eta, starting_lambda=starting_lambda)

    problem = OptProblem(constraints,
                         action_space_dim,
                         best_response_algoonline_convex_algorithmrithm,  # TODO
                         ,
                         fitted_off_policy_evaluation_algorithm,  # TODO
                         exact_policy_algorithm,  # TODO
                         lambda_bound,
                         epsilon,
                         env,
                         max_number_of_main_algo_iterations,
                         num_frame_stack,
                         pic_size)

    lambdas = []
    policies = []

    # TODO: collect data

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
            pi_t, values = problem.best_response(lambda_t, desc='FQI pi_{0}_{1}'.format(iteration, i), exact=exact_policy_algorithm)

            # policies.append(pi_t)
            problem.update(pi_t, values, iteration)  # Evaluate C(pi_t), G(pi_t) and save


if __name__ == '__main__':
    main(sys.argv)
