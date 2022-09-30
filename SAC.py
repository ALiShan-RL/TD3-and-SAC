import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(state_dim, 256)
        self.linear2 = nn.Linear(256, 256)

        self.mean_linear = nn.Linear(256, action_dim)
        self.log_std_linear = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)  # REVIEW - why not tanh?
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  # REVIEW - why clamp?
        # why no log_std = torch.tanh(log_std) ?
        return mean, log_std

    def sample(self, state, deterministic=False):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            action = mean
        else:
            action = normal.rsample()

        log_prob = normal.log_prob(action).sum(axis=-1)
        log_prob -= (2*(np.log(2) - action - F.softplus(-2*action))).sum(axis=-1)

        action = torch.tanh(action) * self.max_action
        return action, log_prob

class QNetwotk(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwotk, self).__init__()

        self.linear1 = nn.Linear(state_dim + action_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)

        self.linear4 = nn.Linear(state_dim + action_dim, 256)
        self.linear5 = nn.Linear(256, 256)
        self.linear6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.linear1(sa))
        q1 = F.relu(self.linear2(q1))
        q1 = self.linear3(q1)

        q2 = F.relu(self.linear4(sa))
        q2 = F.relu(self.linear5(q2))
        q2 = self.linear6(q2)
        return q1, q2


class SAC(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_freq=1,
            alpha=0.2,
    ):
        self.actor = GaussianPolicy(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = QNetwotk(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_freq = policy_freq

        self.alpha = alpha

        self.total_it = 0
    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action, _ = self.actor.sample(state, deterministic)
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_action, next_state_log_pi = self.actor_target.sample(next_state)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            next_state_log_pi = torch.unsqueeze(next_state_log_pi, dim=-1)
            target_Q = self.discount * not_done * (target_Q - self.alpha * next_state_log_pi) + reward

        current_Q1, current_Q2 = self.critic(state, action)
        loss1 = F.mse_loss(current_Q1, target_Q)
        loss2 = F.mse_loss(current_Q2, target_Q)
        critic_loss = loss1 + loss2
        #critic_loss = F.mse_loss(current_Q1, target_Q) / 2 + F.mse_loss(current_Q2, target_Q) / 2

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:   # SAC dont need that

            # Compute actor losse
            action, log_pi = self.actor.sample(state)
            Q1, Q2 = self.critic(state, action)
            Q = torch.min(Q1, Q2)
            actor_loss = (self.alpha*log_pi - Q).mean()


            # Optimize the actor

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
