from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mealpy import FloatVar, SCA
from numba import njit
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

COLOR1 = "\033[91m"
COLOR2 = "\033[92m"
COLOR3 = "\033[93m"
COLOR4 = "\033[94m"
COLOR5 = "\033[95m"
COLOR6 = "\033[96m"
COLOR7 = "\033[97m"
ENDC = "\033[0m"

DATASETS = {
    17: "Breast Cancer",
    52: "Ionosphere",
    78: "Page Blocks",
    81: "Digits",
    94: "Spambase",
    101: "Tic-tac-toe",
    109: "Wine",
    336: "Kidney",
    728: "Toxicity",
    732: "Darwin",
}


@dataclass(frozen=True)
class RunConfig:
    dataset_id: int
    runs: int = 10
    epoch: int = 100
    pop_size: int = 10
    neighbors: int = 5
    cv_splits: int = 10
    seed: int = 42
    cv_jobs: int = -1
    run_jobs: int = 1


@njit(fastmath=True, cache=True)
def sstf(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-3.0 * (x - 0.5)))


class HHOSCA(SCA.DevSCA):
    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        super().__init__(epoch, pop_size, **kwargs)
        self.sort_flag = False

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        rand_pos = self.generator.uniform(self.problem.lb, self.problem.ub)
        return np.where(
            np.logical_and(self.problem.lb <= solution, solution <= self.problem.ub),
            solution,
            rand_pos,
        )

    def evolve(self, epoch: int) -> None:
        pop_new = []
        a = 2.0
        r1 = a * (1.0 - epoch / self.epoch)

        for idx in range(self.pop_size):
            pos = self.pop[idx].solution.copy()

            p_hho = 0.3 * (1.0 - epoch / self.epoch)
            use_hho = self.generator.uniform() < p_hho

            if use_hho:
                E0 = 2 * self.generator.uniform() - 1
                E = 2 * E0 * (1.0 - epoch / self.epoch)

                if abs(E) < 1:
                    pos_new = self.g_best.solution - E * np.abs(self.g_best.solution - pos)
                else:
                    rand_agent = self.pop[self.generator.integers(self.pop_size)]
                    pos_new = rand_agent.solution - self.generator.uniform() * np.abs(
                        rand_agent.solution - 2 * self.generator.uniform() * pos
                    )
            else:
                pos_new = pos.copy()
                for jdx in range(self.problem.n_dims):
                    r2 = 2 * np.pi * self.generator.uniform()
                    r3 = 2 * self.generator.uniform()
                    r4 = self.generator.uniform()

                    if r4 < 0.5:
                        pos_new[jdx] = pos_new[jdx] + r1 * np.sin(r2) * np.abs(
                            r3 * self.g_best.solution[jdx] - pos_new[jdx]
                        )
                    else:
                        pos_new[jdx] = pos_new[jdx] + r1 * np.cos(r2) * np.abs(
                            r3 * self.g_best.solution[jdx] - pos_new[jdx]
                        )

            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)

            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)

        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run BHHOSCA feature selection on one or more UCI datasets."
    )
    parser.add_argument(
        "--dataset-id",
        type=int,
        default=17,
        help="UCI dataset id to run. Default: 17 (Breast Cancer).",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Run every dataset listed in DATASETS.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Print the supported dataset ids and exit.",
    )
    parser.add_argument("--runs", type=int, default=10, help="Number of BHHOSCA runs.")
    parser.add_argument("--epoch", type=int, default=100, help="Number of optimizer epochs.")
    parser.add_argument("--pop-size", type=int, default=10, help="Population size.")
    parser.add_argument(
        "--neighbors",
        type=int,
        default=5,
        help="Number of neighbors for KNN evaluation.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=10,
        help="Number of StratifiedKFold splits.",
    )
    parser.add_argument(
        "--cv-jobs",
        type=int,
        default=-1,
        help="Parallel jobs used by cross_val_score. Use 1 to reduce CPU pressure.",
    )
    parser.add_argument(
        "--run-jobs",
        type=int,
        default=1,
        help="Parallel jobs used across repeated BHHOSCA runs.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser


def print_datasets() -> None:
    print("Available datasets:")
    for dataset_id, dataset_name in DATASETS.items():
        print(f"  {dataset_id}: {dataset_name}")


def load_dataset(dataset_id: int) -> tuple[np.ndarray, np.ndarray, int]:
    try:
        dataset = fetch_ucirepo(id=dataset_id)
    except Exception as exc:
        raise RuntimeError(
            "Failed to fetch the dataset from the UCI ML Repository. "
            "Check your internet connection and dataset id."
        ) from exc

    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    df = df.dropna()

    x = pd.get_dummies(df.iloc[:, :-1])
    y = LabelEncoder().fit_transform(df.iloc[:, -1])

    x_np = x.values.astype(np.float32)
    return x_np, y, x.shape[1]


def build_objective_function(
    x_np: np.ndarray,
    y: np.ndarray,
    num_features: int,
    neighbors: int,
    cv_splits: int,
    cv_jobs: int,
    seed: int,
):
    knn = KNeighborsClassifier(n_neighbors=neighbors)
    kfold = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    scorer = make_scorer(accuracy_score)
    uniform_random = np.random.default_rng(seed).uniform(0.0, 1.0, num_features)

    def objective_function(solution: np.ndarray) -> list[float]:
        sigmoid_solution = sstf(solution)
        boolean_solution = np.asarray(sigmoid_solution > uniform_random, dtype=np.bool_)

        if not np.any(boolean_solution):
            return [1.0, 1.0]

        x_selected = x_np[:, boolean_solution]
        scores = cross_val_score(
            knn,
            x_selected,
            y,
            cv=kfold,
            scoring=scorer,
            n_jobs=cv_jobs,
        )

        return [1.0 - float(np.mean(scores)), float(np.sum(boolean_solution) / num_features)]

    return objective_function


def run_single_optimization(config: RunConfig, objective_function, num_features: int) -> tuple[float, float, float]:
    algorithm = HHOSCA(epoch=config.epoch, pop_size=config.pop_size)
    problem = {
        "obj_func": objective_function,
        "bounds": FloatVar(lb=(-8.0,) * num_features, ub=(8.0,) * num_features),
        "minmax": "min",
        "obj_weights": [0.9, 0.1],
        "log_to": None,
        "name": "bHHOSCA",
    }
    g_best = algorithm.solve(problem)
    return (
        float(g_best.target.fitness),
        float(1.0 - g_best.target.objectives[0]),
        float(g_best.target.objectives[1] * num_features),
    )


def summarize_results(results: list[tuple[float, float, float]]) -> dict[str, float]:
    results_np = np.array(results, dtype=np.float64)
    fitnesses = results_np[:, 0]
    accuracies = results_np[:, 1]
    num_features_selected = results_np[:, 2]

    return {
        "avg_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "avg_num_features": float(np.mean(num_features_selected)),
        "avg_fitness": float(np.mean(fitnesses)),
        "std_fitness": float(np.std(fitnesses)),
        "best_fitness": float(np.min(fitnesses)),
        "worst_fitness": float(np.max(fitnesses)),
    }


def print_summary(dataset_id: int, summary: dict[str, float]) -> None:
    dataset_name = DATASETS.get(dataset_id, "Custom Dataset")
    print(f"Dataset: {dataset_name} ({dataset_id})")
    print("Algorithm: bHHOSCA")
    print(f"{COLOR1}Average accuracy: {summary['avg_accuracy']:.4f}{ENDC}")
    print(f"{COLOR2}Standard deviation of accuracy: {summary['std_accuracy']:.4f}{ENDC}")
    print(f"{COLOR3}Average number of features: {summary['avg_num_features']:.4f}{ENDC}")
    print(f"{COLOR4}Average fitness: {summary['avg_fitness']:.4f}{ENDC}")
    print(f"{COLOR5}Standard deviation of fitness: {summary['std_fitness']:.4f}{ENDC}")
    print(f"{COLOR6}Best fitness: {summary['best_fitness']:.4f}{ENDC}")
    print(f"{COLOR7}Worst fitness: {summary['worst_fitness']:.4f}{ENDC}")
    print("-" * 50)


def run_dataset(config: RunConfig) -> dict[str, float]:
    x_np, y, num_features = load_dataset(config.dataset_id)
    objective_function = build_objective_function(
        x_np=x_np,
        y=y,
        num_features=num_features,
        neighbors=config.neighbors,
        cv_splits=config.cv_splits,
        cv_jobs=config.cv_jobs,
        seed=config.seed,
    )

    if config.run_jobs == 1:
        results = [
            run_single_optimization(config, objective_function, num_features)
            for _ in range(config.runs)
        ]
    else:
        results = Parallel(n_jobs=config.run_jobs)(
            delayed(run_single_optimization)(config, objective_function, num_features)
            for _ in range(config.runs)
        )

    summary = summarize_results(results)
    print_summary(config.dataset_id, summary)
    return summary


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_datasets:
        print_datasets()
        return 0

    dataset_ids = list(DATASETS) if args.all_datasets else [args.dataset_id]

    for dataset_id in dataset_ids:
        run_dataset(
            RunConfig(
                dataset_id=dataset_id,
                runs=args.runs,
                epoch=args.epoch,
                pop_size=args.pop_size,
                neighbors=args.neighbors,
                cv_splits=args.cv_splits,
                seed=args.seed,
                cv_jobs=args.cv_jobs,
                run_jobs=args.run_jobs,
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
