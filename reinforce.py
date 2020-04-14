from typing import Iterable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class PiApproximationWithNN():
    def __init__(self, state_dims, num_actions, alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """

        self.state_dims = state_dims
        self.num_actions = num_actions
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.alpha = alpha

        # model to learn policy parameters
        self.model = nn.Sequential(nn.Linear(state_dims, 32),
                                   nn.ReLU(),
                                   nn.Linear(32, 32),
                                   nn.ReLU(),
                                   nn.Linear(32, 32),
                                   nn.ReLU(),
                                   nn.Linear(32, num_actions),
                                   nn.Softmax()
                                   ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha, betas=(0.9, 0.999))

        # set model in training mode
        self.model.train()

    def __call__(self, s) -> int:
        """
        Passes state through network and returns selected action
        from a probability distribution.
        :param s: state
        :return: action
        """
        state = torch.FloatTensor(s).view(1, self.state_dims)
        actionProb = self.model(state).detach().cpu().numpy()
        action = np.random.choice(self.num_actions, 1, p=actionProb[0, :])[0]

        return action

    def update(self, s, a, gamma_t, delta):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """

        self.optimizer.zero_grad()

        # wrap state and return in Tensor objects
        sTensor = torch.FloatTensor(s)
        actionProb = self.model(sTensor)[a]

        loss = - torch.tensor(gamma_t * delta) * torch.log(actionProb)

        loss.backward()
        self.optimizer.step()

class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    """
    def __init__(self,b):
        self.b = b

    def __call__(self,s) -> float:
        return self.b

    def update(self,s,G):
        pass

class VApproximationWithNN(Baseline):

    def __init__(self, state_dims, alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        self.dims = state_dims
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.model = nn.Sequential(nn.Linear(state_dims, 32),
                                   nn.ReLU(),
                                   nn.Linear(32, 32),
                                   nn.ReLU(),
                                   nn.Linear(32, 32),
                                   nn.ReLU(),
                                   nn.Linear(32, 1)).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha, betas=(0.9, 0.999))

    def __call__(self,s) -> float:
        # return predicted value of state
        s = torch.FloatTensor(s).view(1, self.dims)
        stateValue = self.model(s).detach().item()
        return stateValue

    def update(self,s,G):

        # zero gradient
        self.optimizer.zero_grad()

        # convert state and return to tensors
        sTensor = torch.FloatTensor(s).view(1, self.dims)
        gTensor = torch.FloatTensor(np.array(G)).view(1, 1)

        # approximate value
        value = self.model(sTensor)

        # compute loss
        loss = nn.MSELoss()
        loss = loss(value, gTensor)

        # back prop
        loss.backward()

        # optimization step
        self.optimizer.step()


def REINFORCE(
    env, #open-ai environment
    gamma:float,
    num_episodes:int,
    pi:PiApproximationWithNN,
    V:Baseline) -> Iterable[float]:
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    # list of gradient for every episode
    grads = []

    for e in range(num_episodes):

        # reset env
        obs = env.reset()
        action = pi(obs)

        done = False
        states = []
        actions = []
        rewards = [0]

        states.append(obs)
        actions.append(action)

        t = 0

        # generate episode following the policy
        while not done:
            obs, reward, done, info = env.step(action)
            action = pi(obs)

            states.append(obs)
            rewards.append(reward)
            actions.append(action)

            t += 1

        # update policy params
        T = t + 1

        for t in range(T):
            g = np.array([pow(gamma, k-t-1) * rewards[k] for k in range(t + 1, T)]).sum()
            s = states[t]
            delta = g - V(s)
            gamma_t = pow(gamma, t)

            if not t:
                grads.append(g)

            # update weights
            V.update(s, g)
            # update policy params
            pi.update(s, actions[t], gamma_t, delta)

    return grads