import numpy as np
import cma
import math
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import torch
import time
import torch.nn.functional as F
from utils.utils_train import set_seed
import pickle
from black_box_env import BlackBoxEnv
from agent import Agent
import argparse
import os
import random


class CMA_ES(object):
    def __init__(self, env,
                 pop_size=40,
                 max_evaluation=40000,
                 activation=torch.relu,
                 num_layer=8,
                 hidden_dim=32,
                 init_sigma=0.3,
                 ):
        self.env = env
        self.pop_size = pop_size
        self.max_evaluation = max_evaluation
        self.activation = activation
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.init_sigma = init_sigma
        self.agent = Agent(id=0, in_dim=1, out_dim=1, num_layers=num_layer, hidden_dim=hidden_dim, activation=activation, lr=None)

    def get_setting_info(self):
        info = f'popsize({self.pop_size})'
        info += f'_evaluation({self.max_evaluation})'
        info += f'_layer({self.num_layer})'

        net_info = self.agent.net.get_setting()
        info += f'_{net_info}'
        info += f'_sigma({self.init_sigma})'
        return info

    def black_box_evaluate(self, net_params):
        self.agent.net.set_flat_params(net_params)
        loss = self.env.run(self.agent)
        return loss

    def train(self, seed, log_dir, save_log=False):
        print(self.get_setting_info(), f'curve_id:{self.env.curve_id}')

        init_vec = self.agent.net.get_flat_params()

        cma_opts = {
            'popsize': self.pop_size,
            'seed': seed,
        }

        opt = cma.CMAEvolutionStrategy(init_vec, self.init_sigma, cma_opts)

        total_params = len(init_vec)
        print(f"=== CMA-ES Neural Network Black-box Regression ===")
        print(f"parameter num: {total_params}")

        iter_cnt = 0
        loss_history = []

        num_evaluation = 0
        start_time = time.time()
        while num_evaluation < self.max_evaluation:
            candidate_param_vecs = opt.ask()

            losses = [self.black_box_evaluate(vec) for vec in candidate_param_vecs]
            opt.tell(candidate_param_vecs, losses)

            loss_history.append(opt.result.fbest)

            num_evaluation += len(candidate_param_vecs)
            iter_cnt += 1
            if iter_cnt % 100 == 0:
                best_vec = opt.result.xbest
                best_mse = opt.result.fbest
                time_elapsed = time.time() - start_time
                print(f"Iter {iter_cnt:3d} | Evals {num_evaluation} | Best Global MSE = {best_mse:.6f} | time elapsed: {time_elapsed:.2f} s")

        best_param_vec = opt.result.xbest
        min_mse = opt.result.fbest

        ##################### Save result #####################
        self.agent.net.set_flat_params(best_param_vec)
        loss = self.env.run(self.agent)
        logs = {"loss_history": loss_history,
                "x": self.agent.states,
                "y": self.env.curve_function(self.agent.states),
                "y_pred": self.agent.outs,
                "loss": self.agent.loss
                }

        setting = self.get_setting_info()
        if save_log:
            fname = f'{log_dir}/curve({self.env.curve_id})_seed({seed})_cma_es_{setting}_loss({loss:.5f}).np'
            with open(fname, 'wb') as f:
                f.write(pickle.dumps(logs))

        print(f"Mim MSE loss: {min_mse:.6f}")
        return loss


def fit_curve(curve_id, log_dir, init_sigma, save_log=False):
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
        cma_es = CMA_ES(
            env=env,
            pop_size=args.pop_size,
            max_evaluation=args.max_evaluation,
            activation=activation,
            num_layer=args.num_layer,
            hidden_dim=args.hidden_dim,
            init_sigma=init_sigma,
        )
        loss = cma_es.train(seed, log_dir, save_log=save_log)
        losses.append(loss)

    print(f'curve:{curve_id}')
    print(f"loss:{losses}")
    print(f"loss: mean:{np.mean(losses):.5f}, std:{np.std(losses):.5f}",)

    with open(f'{log_dir}/result_{args.activation}.txt', 'a') as f:
        print(f'curve:{curve_id}, max_evaluation:{args.max_evaluation}', file=f)
        print(f"loss:{losses}", file=f)
        print(f"loss: mean:{np.mean(losses):.5f}, std:{np.std(losses):.5f}", file=f)
        print('=======================================================================================', file=f, flush=True)

    return np.mean(losses)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    #parser.add_argument("--curve_id", type=id, default=0)
    parser.add_argument("--pop_size", type=int, default=40)
    parser.add_argument("--max_evaluation", type=int, default=40000)
    parser.add_argument("--activation", type=str, default="cos", choices=["cos", "relu", "tanh"])
    parser.add_argument("--num_layer", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--max_round", type=int, default=10)
    #parser.add_argument("--init_sigma", type=float, default=1.0)
    return parser


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    log_dir = f"./logs/cma/layer{args.num_layer}"
    os.makedirs(log_dir, exist_ok=True)

    init_sigma = 1.4
    losses = []
    for k in range(0, 8):
        loss = fit_curve(k, log_dir, init_sigma, save_log=False)
        losses.append(loss)

    print(f"init_sigma:{init_sigma}, mean loss:{np.mean(losses)}")
    print('========== finished =========')

