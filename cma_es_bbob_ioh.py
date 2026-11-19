import ioh
import numpy as np
from cma import CMAEvolutionStrategy
import os
import csv
from utils.utils_train import set_seed


def run_cmaes_bbob(
    fid: int,
    iid: int,
    dim: int,
    budget: int,
    target_gap: float = 1e-8,
    sigma0: float = 2.0,
    seed: int = 42
) -> int | None:
    """
    Run CMA-ES on IOH BBOB problem, return ERT (evaluations to reach target gap).
    Returns None if budget exhausted without hitting target.
    Fixed seed for reproducibility.
    """

    problem = ioh.get_problem(
        fid,
        instance=iid,
        dimension=dim,
        problem_class=ioh.ProblemClass.BBOB
    )

    f_opt = problem.optimum.y

    x0 = np.random.uniform(-sigma0, sigma0, size=dim)
    es_opts = {
        "maxfevals": budget,
        "seed": seed,
        "verbose": 0,
        "popsize": 40,
    }
    es = CMAEvolutionStrategy(x0, sigma0, inopts=es_opts)

    ert = None
    while problem.state.evaluations < budget:
        xs = es.ask()
        fs = [problem(x) for x in xs]
        es.tell(xs, fs)

        current_best_y = problem.state.current_best.y

    print(f"F{fid} I{iid} D{dim} |found_best={current_best_y}| f_opt={f_opt} , evals={problem.state.evaluations}, ERT={ert}, seed={seed}")
    return current_best_y, es.result.evaluations


if __name__ == "__main__":
    fids = range(1, 25)
    dim = 40
    max_evals = 10000

    csv_filename = f"./bbob/cma_bbob_dim{dim}_eval{max_evals}_pos40.csv"

    # 写入表头
    if not os.path.exists(csv_filename):
        with open(csv_filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["func_id", "instance", "seed", "fbest", "evaluations"])

    for fid in fids:
        for iid in range(1, 2):
            for seed in range(1,11):
                set_seed(seed)

                [fbest, evals] = run_cmaes_bbob(fid=fid, iid=iid, dim=dim, budget=max_evals, target_gap=1e-8)
                with open(csv_filename, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([fid, iid, seed, fbest, evals])
                print('=====================================================================================')
