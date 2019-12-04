import numpy as np


class ValueFunction(object):
    def __init__(self):
        '''
        '''
        self.prev_values = []
        self.exact_values = []
        self.eval_values = {}

    def append(self, value):
        self.prev_values.append(value)

    def avg(self, append_zero=False):
        if append_zero:
            return np.hstack([np.mean(self.prev_values, 0), np.array([0])])
        else:
            return np.mean(self.prev_values, 0)

    def last(self, append_zero=False):
        if append_zero:
            return np.hstack([self.prev_values[-1], np.array([0])])
        else:
            return np.array(self.prev_values[-1])

    def add_exact_values(self, values):
        self.exact_values.append(values)

    def add_eval_values(self, eval_values, idx):
        if idx not in self.eval_values:
            self.eval_values[idx] = []

        self.eval_values[idx].append(eval_values)
