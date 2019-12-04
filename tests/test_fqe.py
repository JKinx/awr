import awr_configs
import learning.awr_agent as awr_agent
import gym
import tensorflow as tf
import numpy as np
import torch
import util.rl_path as rl_path
from tqdm import tqdm as tqdm
import random
# from sklearn.ensemble import ExtraTreesRegressor
import torch
import torch.nn as nn
import warnings
import pickle
from matplotlib import pyplot as plt
import argparse

warnings.filterwarnings("ignore")


def sample_action(agent, s, action_std):
    n = len(s.shape)
    s = np.reshape(s, [-1, agent.get_state_size()])

    feed = {
        agent._s_tf: s
    }

    run_tfs = [agent._norm_a_pd_tf.parameters["loc"]]

    out = agent._sess.run(run_tfs, feed_dict=feed)
    loc = torch.tensor(out[0])

    a = np.array(torch.distributions.Normal(loc, scale=action_std).sample().tolist())

    if n == 1:
        a = a[0]

    return a


def rollout_path(agent, action_std):
    path = rl_path.RLPath()

    s = agent._env.reset()
    s = np.array(s)
    path.states.append(s)

    done = False
    while not done:
        a = sample_action(agent, s, action_std)
        s, r, done, info = agent._step_env(a)
        s = np.array(s)

        path.states.append(s)
        path.actions.append(a)
        path.rewards.append(r)
        path.logps.append(0)

    path.terminate = agent._check_env_termination()

    return path


class Policy(nn.Module):
    """Policy class with an epsilon-greedy dqn model"""
    def __init__(self, agent, action_std):
        super().__init__()
        self.agent = agent
        self.action_std = action_std

    def forward(self, states):
        return sample_action(self.agent, states, self.action_std)


class Q(nn.Module):
    """Q-network using a NN"""
    def __init__(self, state_dim, action_dim, lr):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fitted = False

        self.model = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def forward(self, state):
        """Forward"""
        state = torch.tensor(state).cuda().float()
        return self.model(state)

    def predict(self, state):
        """Forward without gradients (used for predictions)"""
        state = torch.tensor(state).cuda().float()
        with torch.no_grad():
            return self.model(state).squeeze().cpu().numpy()

    def fit(self, state, true_value):
        """Fit NN with a single backward step"""
        self.fitted = True
        state = torch.tensor(state).cuda().float()
        true_value = torch.tensor(true_value).cuda().float()
        self.optimizer.zero_grad()
        out = self(state).squeeze()
        loss = self.criterion(out, true_value)
        loss.backward()
        self.optimizer.step()


def is_fitted(sklearn_regressor):
    """Helper function to determine if a regression model from scikit-learn
    has ever been `fit`"""
    return hasattr(sklearn_regressor, 'n_outputs_')


class FittedQEvaluation(object):
    def __init__(self, regressor=None):
        self.regressor = regressor
        self.tree_regressor = regressor is None

    def regressor_fitted(self):
        if self.tree_regressor:
            return is_fitted(self.regressor)
        else:
            return self.regressor.fitted

    def Q(self, state_actions):
        """Return the Q function estimate of `states` for each action"""    
        if not self.regressor_fitted():
            return np.zeros(state_actions.shape[0])
        return self.regressor.predict(state_actions)

    def fit_Q(self, eval_policy, episodes, num_iters=100, discount=0.95):        
        batches = []
        batch_len = len(episodes) // 10

        for i in range(10):
            Is = []
            S2s = []
            Rs = []
            Ts = []

            for I,R,S2,T in episodes[i * batch_len : (i + 1) * batch_len]:
                Is.append(I)
                Rs.append(R)
                S2s.append(S2)
                Ts.append(T)

            batches.append((np.concatenate(Is, 0), np.concatenate(Rs, 0),
                            np.concatenate(S2s, 0), np.concatenate(Ts, 0)))

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


def get_data(paths):
    episodes = []

    for path in paths:
        I = np.hstack([np.array(path.states)[:-1], np.array(path.actions)])
        R = path.rewards
        S2 = np.array(path.states)[1:]
        episodes.append((I, R, S2))

    return episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_std', type=float, default=0.2,
                        help='std of batch data')
    parser.add_argument('--eval_std', type=float, default=0.2,
                        help='std in eval')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='number of epochs in FQE fit')
    args = parser.parse_args()

    data_std = args.data_std
    configs = awr_configs.AWR_CONFIGS['Reacher-v2']
    datas = get_data(pickle.load(open("../output/Reacher-v2_{}_offline/Reacher-v2_{}_offline_paths.pickle".format(data_std, data_std), "rb")))

    terminals = []
    for el in datas:
        terminal = np.ones(len(el[0]))
        terminal[-1] = 0
        terminals.append(terminal)

    datas2 = []
    for el,terminal in zip(datas, terminals):
        datas2.append((el[0],el[1],el[2],terminal))

    env = gym.make("Reacher-v2")
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    agent = awr_agent.AWRAgent(env=env, sess=sess, **configs)
    agent.load_model("../output/Reacher-v2_{}_offline/model.ckpt".format(data_std))

    qnn = Q(agent.get_state_size(), agent.get_action_size(), 0.001).cuda()

    eval_std = args.eval_std  # For each, try different eval_std
    num_epochs = args.n_epochs
    FQE = FittedQEvaluation(qnn)
    policy = Policy(agent, eval_std)

    FQE.fit_Q(policy, datas2, num_epochs , agent._discount)

    vals0 = []
    for _ in tqdm(range(100)):
        path = rollout_path(agent, eval_std)
        true = sum([r * (agent._discount ** i) for i,r in enumerate(path.rewards)])
        pred = FQE.regressor.predict(np.hstack([path.states[0], path.actions[0]]).reshape(1,-1))
        vals0.append([true, pred])

    # val[0] true, val[1] prediction
    # TODO:
    # 1) Draw line of best fit and y=x
    # 2) Calculate MSE loss or something (maybe R^2 too?)
    xs = [val[0] for val in vals0]
    ys = [val[1] for val in vals0]

    # Plot linear regression line
    plt.plot(np.unique(xs), np.poly1d(np.polyfit(xs, ys, 1))(np.unique(xs)))

    # Plot y=x
    plt.plot(np.unique(xs), np.unique(xs), 'k--')

    # Mean-squared Error
    mse = np.mean((np.array(ys) - np.array(xs)) ** 2)
    print('MSE: {}'.format(mse))

    plt.scatter(xs, ys, color='r')
    plt.xlabel("True Value")
    plt.ylabel("Prediction")
    plt.title("FQE: {}, {}, MSE: {}".format(data_std, eval_std, mse))
    plt.savefig("../output/Reacher-v2_FQE_{}_{}_{}.jpg".format(data_std, eval_std, num_epochs))
    plt.show()
