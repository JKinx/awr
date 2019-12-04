import argparse
import gym
import numpy as np
import os
import sys
import tensorflow as tf
import pickle
from tqdm import tqdm
from copy import deepcopy as dc
from util.policy import Policy

import awr_configs
import learning.awr_agent as awr_agent

class BestResponse(object):
    def __init__(self, args):
        self.args = arg

    def run(self, dataset):
        enable_gpus(self.arg.gpu)

        self.env = self.build_env(self.args.env)
        self.agent = self.build_agent(self.env)
        self.agent.visualize = self.args.visualize

        self.agent._replay_buffer = dc(dataset)

        self.agent.op_train(max_iter=self.args.max_iter,
                    test_episodes=self.args.test_episodes,
                    output_dir=self.args.output_dir,
                    output_iters=self.args.output_iters)

        return Policy(self.agent)

    def build_agent(self, env):
        env_id = self.args.env
        agent_configs = {}
        if (env_id in awr_configs.AWR_CONFIGS):
            agent_configs = awr_configs.AWR_CONFIGS[env_id]

        graph = tf.Graph()
        sess = tf.Session(graph=graph)
        agent = awr_agent.AWRAgent(env=env, sess=sess, **agent_configs)

        return agent

    def enable_gpus(self, gpu_str):
        if (gpu_str is not ""):
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
        return

    def build_env(self, env_id):
        assert(env_id is not ""), "Unspecified environment."
        env = gym.make(env_id)
        return env
