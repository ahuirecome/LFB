# cma_es_bbob_standard.py
import numpy as np
import cma
import ioh
import csv
import os
from agent import Agent
import random
import torch
from utils.utils_train import set_seed



class LFB_ES_BBOB(object):
    def __init__(self,
                 pop_size=40,
                 max_generation=10000,
                 activation=torch.cos,
                 min_layer=8,
                 max_layer=8,
                 hidden_dim=32,
                 lr=0.0001,
                 training_step=10,
                 bbob_dim=10):
        self.pop_size = pop_size
        self.max_generation = max_generation
        self.activation = activation
        self.min_layer = min_layer
        self.max_layer = max_layer
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.training_step = training_step
        self.bbob_dim = bbob_dim
        self.agents = self.init_agent(max_agent=self.pop_size, activation=self.activation, hidden_dim=self.hidden_dim)
        self.x = [np.ones(bbob_dim)]
        self.fbest = 0
        self.x_best = 0
        self.count = 0

    def stop(self):
        if self.evaluations >= self.max_evaluations:
            return True
        else:
            return False

    def get_setting_info(self):
        info = f'popsize({self.pop_size})'
        info += f'_generation({self.max_generation})'
        info += f'_layer({self.min_layer}_{self.max_layer})'

        net_info = self.agents[0].net.get_setting()
        info += f'_{net_info}'
        info += f'_lr({self.lr})'
        info += f'_steps({self.training_step})'
        return info

    def init_agent(self, max_agent, activation, hidden_dim):
        agents = []
        for k in range(max_agent):
            num_layer = random.randint(self.min_layer, self.max_layer)
            lr = random.choice([0.1, 0.01, 0.001, 0.0001])
            agent = Agent(id=k, in_dim=self.bbob_dim, out_dim=self.bbob_dim, num_layers=num_layer, activation=activation, hidden_dim=hidden_dim, lr=lr)
            agents.append(agent)

            parms = agent.net.get_flat_params()
            # print(f"agent {k} parameter num: {len(parms)}")
        return agents

    def ask(self):
        outs = []
        for k in range(self.pop_size):
            y_pred = self.agents[k].get_batch_out(self.x)[0]
            outs.append(y_pred)
        return outs

    def tell(self, solutions, function_values):
        best_index = np.argmin(function_values)
        self.fbest = function_values[best_index]

        # agent evolve
        targets = solutions[best_index]

        for k in range(len(self.agents)):
            if k != best_index:
                self.agents[k].learn(self.x, targets, lr=None, steps=self.training_step)

    def train(self, ):
        print(self.get_setting_info(), f'curve_id:{self.env.curve_id}')

        best_index = 0
        loss_history = []

        prev_best_loss = float('inf')
        start_time = time.time()
        for generation in range(self.max_generation):

            for k in range(self.pop_size):
                [states, outs, _, loss] = self.env.run(self.agents[k])
                self.agents[k].loss = loss

            losses = [x.loss for x in self.agents]
            best_index = np.argmin(losses)
            best_loss = losses[best_index]
            loss_history.append(best_loss)
            best_agent = self.agents[best_index]

            targets = self.agents[best_index].get_batch_out(self.env.states)

            self.agent_evolve(self.env.states, targets)

            time_cost = time.time() - start_time
            if generation % 100 == 0 or generation == self.max_generation - 1:
                print(
                    f'generation: {generation}, best agent:{best_index}, loss:{best_loss:.8f}, lr:{best_agent.lr}, time_cost:{time_cost:.1f}s')

        plt.pause(1)
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


def run_lfb_bbob_single_run(function_id: int, dim: int, instance: int,  max_fevals: int):
    """
    """
    problem = ioh.get_problem(
        function_id,
        instance=instance,
        dimension=dim,
        problem_class=ioh.ProblemClass.BBOB
    )
    fopt = problem.optimum.y
    #print(f"x: {problem.optimum.x}")
    print(f"Running | function_id={function_id}|instance={instance}|dim={dim} |fopt={fopt}| seed={seed:2d}")

    es = LFB_ES_BBOB(pop_size=40, min_layer=2, max_layer=8, hidden_dim=32, bbob_dim=dim, training_step=10)

    solutions = np.random.uniform(low=-5.0, high=5.0, size=(es.pop_size, dim))
    for k in range(es.pop_size):
        es.agents[k].learn(es.x, solutions[k], lr=0.01, steps=1000)

    # LFB
    round = 0
    while problem.state.evaluations < max_fevals:
        X = es.ask()
        f_vals = np.array([problem(x) for x in X])
        es.tell(X, f_vals)
        round += 1
        if round % 100 == 0:
            print(f'evals:{problem.state.evaluations}, fbest:{es.fbest:.8f}')

    best_f = es.fbest
    return best_f, problem.state.evaluations


if __name__ == "__main__":

    dim = 40
    func_list = range(1, 25)
    instance_list = list(range(1, 2))
    seeds_per_instance = list(range(1, 11))
    csv_filename = f"./bbob/lfb_bbob_hd32_dim{dim}.csv"


    if not os.path.exists(csv_filename):
        with open(csv_filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["func_id", "instance", "seed", "fbest", "evaluations"])

    #
    for fid in func_list:
        for inst in instance_list:
            for seed in seeds_per_instance:
                set_seed(seed)

                fbest, evals = run_lfb_bbob_single_run(fid, dim, inst, max_fevals=10000)

                with open(csv_filename, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([fid, inst, seed, fbest, evals])
                print(f" --> f{fid}, inst{inst}, seed{seed}, fbest={fbest:.6e}, evals={evals}\n")

    print("========== All BBOB experiments finished ==========")
    print(f"Result saved into {csv_filename}")
