# =================
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTHONHASHSEED"] = str(0)
# ==============================================

import copy
import random
import sys
import time
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import markers
import torch
import torch.nn.functional as F
from agent import Agent
import numpy as np
from black_box_env import BlackBoxEnv
import matplotlib.pyplot as plt
from utils.utils_train import set_seed
import pickle
import argparse
import os


class LFB_ES(object):
    def __init__(self, env,
                 pop_size=40,
                 max_evaluation=10000,
                 activation=torch.cos,
                 min_layer=8,
                 max_layer=8,
                 hidden_dim=32,
                 lr=0.0001):
        self.env = env
        self.pop_size = pop_size
        self.max_evaluation = max_evaluation
        self.activation = activation
        self.min_layer = min_layer
        self.max_layer = max_layer
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.training_steps = [5, 10, 15, 20]
        self.agents = self.init_agent(max_agent=self.pop_size, activation=self.activation, hidden_dim=self.hidden_dim)

    def get_setting_info(self):
        info = f'popsize({self.pop_size})'
        info += f'_evaluation({self.max_evaluation})'
        info += f'_layer({self.min_layer}_{self.max_layer})'

        net_info = self.agents[0].net.get_setting()
        info += f'_{net_info}'
        info += f'_lr({self.lr})'
        info += f'_steps({self.training_steps})'
        return info

    def init_agent(self, max_agent, activation, hidden_dim):
        agents = []
        for k in range(max_agent):
            num_layer = random.randint(self.min_layer, self.max_layer)
            lr = random.choice([0.1, 0.01, 0.001, 0.0001])
            steps = random.choice(self.training_steps)
            agent = Agent(id=k, in_dim=1, out_dim=1, num_layers=num_layer, activation=activation, hidden_dim=hidden_dim, lr=lr, training_steps=steps)
            agents.append(agent)

            parms = agent.net.get_flat_params()
            #print(f"agent {k} parameter num: {len(parms)}")
        return agents

    def train(self,):
        print(self.get_setting_info(), f'curve_id:{self.env.curve_id}')

        best_index = 0
        loss_history = []

        prev_best_loss = float('inf')
        start_time = time.time()
        num_evaluation = 0
        generation = 0
        while num_evaluation < self.max_evaluation:
            generation += 1
            num_evaluation += self.pop_size
            for k in range(self.pop_size):
                loss = self.env.run(self.agents[k])
                self.agents[k].loss = loss

            losses = [x.loss for x in self.agents]
            best_index = np.argmin(losses)
            best_loss = losses[best_index]
            loss_history.append(best_loss)
            best_agent = self.agents[best_index]

            targets = self.agents[best_index].get_batch_out(self.env.states)

            self.agent_evolve(self.env.states, targets)

            time_cost = time.time() - start_time
            if generation % 100 == 0:
                print(f'generation:{generation}, num_evaluation: {num_evaluation}, best agent:{best_index}, loss:{best_loss:.8f}, lr:{best_agent.lr}, time_cost:{time_cost:.1f}s')

        print(f'generation:{generation}, num_evaluation: {num_evaluation}, best agent:{best_index}, loss:{best_loss:.8f}, lr:{best_agent.lr}, time_cost:{time_cost:.1f}s')

        return self.agents[best_index], loss_history

    def test(self, agent):
        x = np.linspace(-1, 1, 300)
        y = self.env.curve_function(x)
        y_pred = agent.get_batch_out(x)
        return x, y, y_pred

    def get_all_agent_action(self, state):
        actions = []
        for k in range(len(self.agents)):
            action = self.agents[k].get_action(states=[state])
            actions.append(action)
        actions = np.array(actions)
        return actions

    def agent_evolve(self, states, targets):
        losses = [x.loss for x in self.agents]
        best_index = np.argmin(losses)

        for k in range(len(self.agents)):
            if k != best_index:
                self.agents[k].learn(states, targets, self.lr, None)

    def show_curve(self, x, y, y_pred, title):
        plt.ion()
        plt.clf()
        plt.plot(x, y, marker="*", linestyle='', label='real')
        plt.plot(x, y_pred, color='purple', label='predict')
        plt.legend()
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.show(block=False)
        plt.pause(1)

    def show_net_weights(self, net):
        params = net.get_flat_params()
        plt.clf()
        plt.plot(params, color='blue', label='net')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show(block=False)
        plt.pause(1)


def plot_fit_curve(title, x, y_true, y_pred, label1, label2):
    plt.figure(figsize=(12, 8))
    plt.plot(x, y_real, c="red", label=label1)
    plt.plot(x, y_pred, c="blue", ls="--", label=label2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.title(title)
    plt.show()
    plt.pause(1)


def fit_curve(curve_id, log_dir, save_log):
    parser = get_parser()
    args = parser.parse_args()

    act_map = {
        "cos": torch.cos,
        "relu": F.relu,
        "tanh": torch.tanh
    }
    activation = act_map[args.activation]

    losses = []
    for round in range(args.max_round):
        seed = args.seed + round
        set_seed(seed)

        print(f'======================================== round:{round}, seed:{seed} ==================================')
        env = BlackBoxEnv(curve_id=curve_id)
        lfb_es = LFB_ES(
            env=env,
            pop_size=args.pop_size,
            max_evaluation=args.max_evaluation,
            activation=activation,
            min_layer=args.min_layer,
            max_layer=args.max_layer,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            #training_step=args.training_step
        )

        [best_agent, loss_history] = lfb_es.train()

        loss = env.run(best_agent)
        logs = {"loss_history": loss_history,
                "x": best_agent.states,
                "y": env.curve_function(best_agent.states),
                "y_pred": best_agent.outs,
                "loss": best_agent.loss
                }

        if save_log:
            setting_info = lfb_es.get_setting_info()
            log_name = f'./{log_dir}/curve({curve_id})_seed({seed})_lfb_es_{setting_info}_loss({loss:.5f}).np'
            with open(log_name, 'wb') as f:
                f.write(pickle.dumps(logs))

        losses.append(loss)

    print(f'curve:{curve_id}')
    print(f"loss: mean:{np.mean(losses):.5f}, std:{np.std(losses):.5f}",)

    with open(f'{log_dir}/result_{args.activation}_mulstep_popsize{args.pop_size}.txt', 'a') as f:
        print(f'curve:{curve_id}, evaluation:{args.max_evaluation}', file=f)
        print(f"loss:{losses}", file=f)
        print(f"loss: mean:{np.mean(losses):.5f}, std:{np.std(losses):.5f}", file=f)
        print('=======================================================================================', file=f, flush=True)

    print("---- finished ----")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    #parser.add_argument("--curve_id", type=id, default=0)
    parser.add_argument("--pop_size", type=int, default=40)
    parser.add_argument("--max_evaluation", type=int, default=40000)
    parser.add_argument("--activation", type=str, default="cos", choices=["cos", "relu", "tanh"])
    parser.add_argument("--min_layer", type=int, default=2)
    parser.add_argument("--max_layer", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0001)
    #parser.add_argument("--training_step", type=int, default=50)
    parser.add_argument("--max_round", type=int, default=10)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    log_dir = f"./logs/lfb/layer{args.min_layer}_{args.max_layer}"
    os.makedirs(log_dir, exist_ok=True)
    for k in range(6, 8):
        fit_curve(k, log_dir, save_log=False)

    print('========== finished =========')


