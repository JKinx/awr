import numpy as np
from copy import deepcopy
from util.value_function import ValueFunction
import pandas as pd
import deepdish as dd
from tqdm import tqdm


class OptProblem(object):
    def __init__(self, constraints, dataset, init_states, best_response_algorithm, online_convex_algorithm, fitted_off_policy_evaluation_algorithm, exact_policy_algorithm, lambda_bound = 1., epsilon = .01, env= None, max_iterations=None):
        '''
        This is a problem of the form: min_pi C(pi) where G(pi) < eta.

        dataset: list. Will be {(x,a,x',c(x,a), g(x,a)^T)}
        action_space_dim: number of dimension of action space
        dim: number of constraints
        best_response_algorithm: function which accepts a |A| dim vector and outputs a policy which minimizes L
        online_convex_algorithm: function which accepts a policy and returns an |A| dim vector (lambda) which maximizes L
        lambda_bound: positive int. l1 bound on lambda |lambda|_1 <= B
        constraints:  |A| dim vector
        epsilon: small positive float. Denotes when this problem has been solved.
        env: The environment. Used for exact policy evaluation to test fittedqevaluation
        '''

        self.dataset = dataset
        self.init_states = init_states
        self.constraints = constraints
        self.C = ValueFunction()
        self.G = ValueFunction()
        self.C_exact = ValueFunction()
        self.G_exact = ValueFunction()
        self.dim = len(constraints)
        self.lambda_bound = lambda_bound
        self.epsilon = epsilon
        self.best_response_algorithm = best_response_algorithm
        self.online_convex_algorithm = online_convex_algorithm
        self.exact_lambdas = []
        self.fitted_off_policy_evaluation_algorithm = fitted_off_policy_evaluation_algorithm
        self.exact_policy_evaluation = exact_policy_algorithm
        self.env = env
        self.prev_lagrangians = []
        self.max_iterations = max_iterations if max_iterations is not None else np.inf
        self.iteration = -2

    def best_response(self, lamb):
        '''
        Best-response(lambda) = argmin_{pi} L(pi, lambda)
        '''
        self.dataset.calculate_cost(lamb)
        policy = self.best_response_algorithm.run(self.dataset)
        return policy

    def online_algo(self):
        '''
        No regret online convex optimization routine
        '''
        gradient = self.G.last() - self.constraints
        lambda_t = self.online_convex_algorithm.run(gradient)

        return lambda_t

    def lagrangian(self, C, G, lamb):
        # C(pi) + lambda^T (G(pi) - eta), where eta = constraints, pi = avg of all pi's seen
        return C.avg() + np.dot(lamb, (G.avg() - self.constraints))

    def max_of_lagrangian_over_lambda(self):
        '''
        The maximum of C(pi) + lambda^T (G(pi) - eta) over lambda is
        B*e_{k+1}, all the weight on the phantom index if G(pi) < eta for all constraints
        B*e_k otherwise where B is the l1 bound on lambda and e_k is the standard
        basis vector putting full mass on the constraint which is violated the most
        '''

        # Actual calc
        maximum = np.max(self.G.avg() - self.constraints)
        index = np.argmax(self.G.avg() - self.constraints)

        if maximum > 0:
            lamb = self.lambda_bound * np.eye(1, self.dim, index)
        else:
            lamb = np.zeros(self.dim)
            lamb[-1] = self.lambda_bound

        maximum = np.max(self.G_exact.avg() - self.constraints)
        index = np.argmax(self.G_exact.avg() - self.constraints)

        print('Lambda maximizing lagrangian: %s' % lamb)
        return self.lagrangian(self.C, self.G, lamb)

    def min_of_lagrangian_over_policy(self, lamb):
        '''
        This function evaluates L(best_response(avg_lambda), avg_lambda)
        '''

        # print 'Calculating best-response(lambda_avg)'
        best_policy = self.best_response(lamb)

        # print 'Calculating C(best_response(lambda_avg))'
        # dataset = deepcopy(self.dataset)
        C_br, _, _ = self.fitted_off_policy_evaluation_algorithm.run(best_policy,'c', self.dataset, self.init_states)
        c_scale = self.dataset.scale

        # print 'Calculating G(best_response(lambda_avg))'
        G_br = []
        g_scales = []
        for i in range(self.dim-1):
            # dataset = deepcopy(self.dataset)
            output, _, _ = self.fitted_off_policy_evaluation_algorithm.run(best_policy,'g', self.dataset, 
                                                                           self.init_states, g_idx=i)
            g_scales.append(self.dataset.scale)
            G_br.append(output)

        G_br.append(0)
        G_br = np.array(G_br)

        print('Calculating exact C, G policy evaluation')
        exact_c, exact_g, performance = self.exact_policy_evaluation.run(best_policy, c_scale, np.array(g_scales))

        print()
        print('C(pi(lambda_avg)) Exact: %s, Evaluated: %s, Difference: %s' % (exact_c, C_br, np.abs(C_br-exact_c)))
        print('G(pi(lambda_avg)) Exact: %s, Evaluated: %s, Difference: %s' % (exact_g, G_br[:-1], np.abs(G_br[:-1]-exact_g)))
        print('Mean lambda: %s' % lamb)
        print()

        return C_br + np.dot(lamb, (G_br - self.constraints)), C_br, G_br, exact_c, exact_g

    def update(self, policy, values, iteration):

        # update C
        # dataset = deepcopy(self.dataset)
        C_pi, eval_values, _ = self.fitted_off_policy_evaluation_algorithm.run(policy,'c', self.dataset, self.init_states)
        c_scale = self.dataset.scale
        self.C.append(C_pi)
        C_pi = np.array(C_pi)
        self.C.add_exact_values(values)
        self.C.add_eval_values(eval_values, 0)

        # update G
        G_pis = []
        g_scales = []
        for i in range(self.dim-1):
            # dataset = deepcopy(self.dataset)
            output, eval_values, _ = self.fitted_off_policy_evaluation_algorithm.run(policy,'g', self.dataset, 
                                                                                     self.init_states, g_idx = i)
            g_scales.append(self.dataset.scale)
            G_pis.append(output)
            self.G.add_eval_values(eval_values, i)
        G_pis.append(0)
        self.G.append(G_pis)
        G_pis = np.array(G_pis)

        # Get Exact Policy
        exact_c, exact_g, performance = self.calc_exact(policy, c_scale, np.array(g_scales))

        print()
        print('C(pi_%s) Exact: %s, Evaluated: %s, Difference: %s' % (iteration, exact_c, C_pi, np.abs(C_pi-exact_c)))
        print('G(pi_%s) Exact: %s, Evaluated: %s, Difference: %s' % (iteration, exact_g, G_pis[:-1], np.abs(G_pis[:-1]-exact_g)))
        print()

    def calc_exact(self, policy, c_scale, g_scales):
        print('Calculating exact C, G policy evaluation')
        exact_c, exact_g, performance = self.exact_policy_evaluation.run(policy, c_scale, g_scales)

        self.C_exact.add_exact_values([performance])
        self.C_exact.append(exact_c)
        self.G_exact.append(np.hstack([exact_g, np.array([0])]))
        return exact_c, exact_g, performance

    def is_over(self, policies, lambdas, infinite_loop=False, calculate_gap=True, results_name='results.csv', policy_improvement_name='policy_improvement.h5'):
        # lambdas: list. We care about average of all lambdas seen thus far
        # If |max_lambda L(avg_pi, lambda) - L(best_response(avg_lambda), avg_lambda)| < epsilon, then done
        self.iteration += 1

        if calculate_gap:
            if len(lambdas) == 0:
                return False
            if len(lambdas) == 1:
                # use stored values
                x = self.max_of_lagrangian_over_lambda()
                y = self.C.last() + np.dot(lambdas[-1], (self.G.last() - self.constraints))
                c_br, g_br, c_br_exact, g_br_exact = self.C.last(), self.G.last(), self.C_exact.last(), self.G_exact.last()[:-1]
            else:
                x = self.max_of_lagrangian_over_lambda()
                y, c_br, g_br, c_br_exact, g_br_exact = self.min_of_lagrangian_over_policy(np.mean(lambdas, 0))

            difference = x-y

            c_exact, g_exact = self.C_exact.avg(), self.G_exact.avg()[:-1]
            c_approx, g_approx = self.C.avg(), self.G.avg()[:-1]

            print('actual max L: %s, min_L: %s, difference: %s' % (x, y, x-y))
            print('Average policy. C Exact: %s, C Approx: %s' % (c_exact, c_approx))
            print('Average policy. G Exact: %s, G Approx: %s' % (g_exact, g_approx))
        else:
            if len(lambdas) == 0:
                return False
            c_exact, g_exact = self.C_exact.avg(), self.G_exact.avg()[:-1]
            c_approx, g_approx = self.C.avg(), self.G.avg()[:-1]
            x = 0
            y, c_br, g_br, c_br_exact, g_br_exact = 0, 0, [0]*(len(self.constraints)), 0, [0]*(len(self.constraints)-1)

        self.prev_lagrangians.append(np.hstack([self.iteration, x, y, c_exact, g_exact, c_approx, g_approx, self.C_exact.last(), self.G_exact.last()[:-1], self.C.last(), self.G.last()[:-1], lambdas[-1][:-1], c_br_exact, g_br_exact, c_br, g_br[:-1]  ]))

        self.save(results_name, policy_improvement_name)
        if infinite_loop:
            # Run forever to gather long curve for experiment
            return False
        else:
            if difference < self.epsilon:
                return True
            elif self.iteration >= self.max_iterations:
                return True
            else:
                return False

    def save(self, results_name, policy_improvement_name):
        labels = []
        for i in range(len(self.constraints)-1):
            labels.append(['g_exact_avg_%s' % i,
                           'g_avg_%s' % i,
                           'g_pi_exact_%s' % i,
                           'g_pi_%s' % i,
                           'g_br_exact_%s' % i,
                           'g_br_%s' % i,
                           'lambda_%s' % i])

        labels = np.array(labels).T.tolist()
        df = pd.DataFrame(self.prev_lagrangians, columns=np.hstack(['iteration', 'max_L', 'min_L', 'c_exact_avg', labels[0], 'c_avg', labels[1], 'c_pi_exact', labels[2], 'c_pi', labels[3], labels[6], 'c_br_exact', labels[4], 'c_br', labels[5]]))
        df.to_csv(results_name, index=False)

        data = {}
        data['c_performance'] = self.C_exact.exact_values
        data['c_eval'] = self.C.eval_values
        data['g_eval'] = self.G.eval_values
        data['g_exacts'] = [x.tolist() for x in self.G_exact.prev_values]
        data['c_exacts'] = [x.tolist() for x in self.C_exact.prev_values]
        data['c_eval_actuals'] = self.C.exact_values
        dd.io.save(policy_improvement_name, data)
