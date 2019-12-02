import torch
import torch.nn as nn
import numpy as np

def sample_action(agent, s, action_std):
    n = len(s.shape)
    s = np.reshape(s, [-1, agent.get_state_size()])

    feed = {
        agent._s_tf : s
    }

    run_tfs = [agent._norm_a_pd_tf.parameters["loc"]]

    out = agent._sess.run(run_tfs, feed_dict=feed)
    loc = torch.tensor(out[0])
    
    a = np.array(torch.distributions.Normal(loc, scale=action_std).sample().tolist())
    
    if n == 1:
        a = a[0]
    
    return a

class Policy(nn.Module):
    """Policy class with an epsilon-greedy dqn model"""
    def __init__(self, agent, action_std=None):
        super().__init__()
        self.agent = agent

        self.action_std = self.agent._action_std if action_std is None else action_std

    def forward(self, states):
        return sample_action(self.agent, states, self.action_std)