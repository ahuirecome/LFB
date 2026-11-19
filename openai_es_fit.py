import numpy as np
import torch
from utils.utils_train import set_seed
import pickle
import time
from black_box_env import BlackBoxEnv
import torch.nn.functional as F
import argparse
from agent import Agent
import os
import random


class OpenAI_ES(object):
    def __init__(self, env,
                 pop_size=40,
                 max_evaluation=10000,
                 activation=torch.relu,
                 num_layer=8,
                 hidden_dim=32,
                 lr=0.02,
                 noise_std=0.08):
        self.env = env
        self.pop_size = pop_size
        self.max_evaluation = max_evaluation
        self.activation = activation
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.noise_std = noise_std
        self.agent = Agent(id=0, in_dim=1, out_dim=1, num_layers=self.num_layer, activation=self.activation, hidden_dim=self.hidden_dim)

    def get_setting_info(self):
        info = f'popsize({self.pop_size})'
        info += f'_evaluation({self.max_evaluation})'
        info += f'_layer({self.num_layer})'
        net_info = self.agent.net.get_setting()
        info += f'_{net_info}'
        info += f'_lr({self.lr})'
        info += f'_noise_std({self.noise_std})'
        return info

    def train(self, seed, log_dir, save_log=False):
        print(self.get_setting_info(), f'curve_id:{self.env.curve_id}')
        mean_params = self.agent.net.get_flat_params()
        param_dim = len(mean_params)
        print(f"parameter num: {param_dim}")

        start_time = time.time()
        loss_history = []

        gen = 0
        num_evaluation = 0
        while num_evaluation < self.max_evaluation:
            noise = torch.randn(self.pop_size, param_dim).cpu().data.numpy() * self.noise_std
            rewards = np.zeros(self.pop_size)

            num_evaluation += self.pop_size

            for i in range(self.pop_size):
                perturb_params = mean_params + noise[i]
                self.agent.net.set_flat_params(perturb_params)
                loss = self.env.run(self.agent)
                rewards[i] = -loss

            reward_mean = rewards.mean()
            reward_std = rewards.std()
            if reward_std > 1e-8:
                norm_rewards = (rewards - reward_mean) / reward_std
            else:
                norm_rewards = rewards

            # update parameter
            grad = np.zeros(len(mean_params))
            for i in range(self.pop_size):
                grad += norm_rewards[i] * noise[i]
            grad /= (self.pop_size * self.noise_std)
            mean_params += self.lr * grad

            # evaluate and record loss
            self.agent.net.set_flat_params(mean_params)
            current_loss = self.env.run(self.agent)
            loss_history.append(current_loss)

            gen += 1
            if gen % 100 == 0:
                elapsed_time = time.time() - start_time
                print(f"Generation: {gen:4d} |evaluation: {num_evaluation}  | MSE Loss: {current_loss:.6f} | Elapsed Time: {elapsed_time:.2f}s")

        ################################################## Save Result ################################################
        loss = self.env.run(self.agent)
        logs = {"loss_history": loss_history,
                "x": self.agent.states,
                "y": self.env.curve_function(self.agent.states),
                "y_pred": self.agent.outs,
                "loss": self.agent.loss
                }

        if save_log:
            setting = self.get_setting_info()
            fname = f'{log_dir}/curve({self.env.curve_id})_seed({seed})_openai_es_{setting}_loss({loss:.5f}).np'
            with open(fname, 'wb') as f:
                f.write(pickle.dumps(logs))

        return  loss


def fit_curve(curve_id, log_dir, lr, noise_std, save_log=False):
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
        openai_es = OpenAI_ES(
            env=env,
            pop_size=args.pop_size,
            max_evaluation=args.max_evaluation,
            activation=activation,
            num_layer=args.num_layer,
            hidden_dim=args.hidden_dim,
            lr=lr,
            noise_std=noise_std,
        )
        loss = openai_es.train(seed, log_dir, save_log)
        losses.append(loss)

    print(f'curve:{curve_id}')
    print(f"loss:{losses}")
    print(f"loss: mean:{np.mean(losses):.5f}, std:{np.std(losses):.5f}",)

    with open(f'{log_dir}/result_{args.activation}.txt', 'a') as f:
        print(f'curve:{curve_id}, max_evaluation:{args.max_evaluation}', file=f)
        print(f"loss:{losses}", file=f)
        print(f"loss: mean:{np.mean(losses):.5f}, std:{np.std(losses):.5f}", file=f)
        print('=======================================================================================', file=f,
              flush=True)

    return np.mean(losses)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--curve_id", type=id, default=0)
    parser.add_argument("--pop_size", type=int, default=40)
    parser.add_argument("--max_evaluation", type=int, default=40000)
    parser.add_argument("--activation", type=str, default="cos", choices=["cos", "relu", "tanh"])
    parser.add_argument("--num_layer", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=12)
    #parser.add_argument("--lr", type=float, default=0.09)
    #parser.add_argument("--noise_std", type=float, default=0.1)
    parser.add_argument("--max_round", type=int, default=10)
    return parser


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    log_dir = f"./logs/openai/layer{args.num_layer}"
    os.makedirs(log_dir, exist_ok=True)
    lr = 0.09
    noise_std = 0.1
    losses = []
    for k in range(0, 8):
        loss = fit_curve(k, log_dir, lr=lr, noise_std=noise_std, save_log=False)
        losses.append(loss)


    print('========== finished =========')