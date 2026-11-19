import torch
import torch.nn as nn
import numpy as np
import random
from networks import AgentNet
DEVICE = "cpu" #"#torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, -1.0, 1.0)
        nn.init.uniform_(m.bias, -1.0, 1.0)


def init_weights_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.uniform_(m.bias, -1.0, 1.0)



class Agent(object):
    def __init__(self, id, in_dim, out_dim, num_layers, activation, hidden_dim, lr=None, training_steps=None):
        self.id = id
        self.net = AgentNet(in_dim=in_dim, out_dim=out_dim, num_layers=num_layers, activation=activation, hidden_dim=hidden_dim)
        self.net.apply(init_weights)
        self.net.to(DEVICE)
        self.in_dim = in_dim
        self.num_layers = num_layers
        self.activation = activation
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.training_steps = training_steps

        self.history_len = 1
        self.loss = 0.0
        self.states = None

    def reset(self):
        self.net = AgentNet(in_dim=self.in_dim, num_layers=self.num_layers, activation=self.activation)
        self.net.apply(init_weights)
        self.net.to(DEVICE)

    def clear_state_out(self):
        self.states = []
        self.outs = []

    def to_net_input(self, states):
        states = np.array(states)
        x = torch.from_numpy(states).float()
        return x

    def get_out(self, state):
        x = self.to_net_input([state])
        with torch.no_grad():
            out = self.net(x)
        out = out.cpu().data.numpy()

        return out

    def get_batch_out(self, states):
        x = np.array(states)
        x = torch.from_numpy(x).float()
        x = x.to(DEVICE)
        with torch.no_grad():
            out = self.net.forward(x)
        out = out.cpu().data.numpy()
        return out

    def learn(self, states, targets, lr=None, steps=None):

        states = np.array(states)
        states = torch.from_numpy(states).float().to(DEVICE)
        targets = torch.from_numpy(targets).float().to(DEVICE)

        if lr is None:
            lr = self.lr

        if steps is None:
            steps = self.training_steps

        optimizer = torch.optim.AdamW(params=self.net.parameters(), lr=lr)
        mse_loss = torch.nn.MSELoss()
        for k in range(steps):
            out = self.net.forward(states)
            loss = mse_loss(out, targets)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            optimizer.step()

