# Vendored ~verbatim from NVARC's open-source notebook (cell `arc_decoder.py`).
# Original authors: Ivan Sorokin & Jean-François Puget (NVARC, ARC Prize 2025 winner).
import os
import bz2
import pickle
import numpy as np


def hashable(guess):
    return tuple(map(tuple, guess))


def score_sum(guesses, getter):
    guess_list = list(guesses.values())
    scores = {}
    for g in guess_list:
        h = hashable(g["solution"])
        x = scores[h] = scores.get(h, [[], g["solution"]])
        x[0].append(g)
    scores = [(getter(sc), o) for sc, o in scores.values()]
    scores = sorted(scores, key=(lambda x: x[0]), reverse=True)
    return [x[-1] for x in scores]


def getter_full_probmul_3(guesses, baseline=3):
    inf_score = np.sum([baseline - g["beam_score"] for g in guesses])
    aug_score = np.mean([np.sum([baseline - s for s in g["score_aug"]]) for g in guesses])
    return inf_score + aug_score


def score_full_probmul_3(guesses):
    return score_sum(guesses, getter_full_probmul_3)


def getter_kgmon(guesses):
    inf_score = len(guesses)
    aug_score = np.mean([np.mean(g["score_aug"]) for g in guesses])
    return inf_score - aug_score


def score_kgmon(guesses):
    return score_sum(guesses, getter_kgmon)


selection_algorithms = [score_full_probmul_3, score_kgmon]


class ArcDecoder:
    def __init__(self, dataset, n_guesses):
        self.dataset = dataset
        self.n_guesses = n_guesses
        self.decoded_results = {}

    def load_decoded_results(self, store, run_name=""):
        for key in os.listdir(store):
            with bz2.BZ2File(os.path.join(store, key)) as f:
                outputs = pickle.load(f)
            base_key = key.split(".")[0]
            self.decoded_results[base_key] = self.decoded_results.get(base_key, {})
            for i, sample in enumerate(outputs):
                self.decoded_results[base_key][f"{key}{run_name}.out{i}"] = sample

    def run_selection_algo(self, selection_algorithm=score_kgmon):
        return {bk: selection_algorithm({k: g for k, g in v.items()}) for bk, v in self.decoded_results.items()}

    def benchmark_selection_algos(self):
        print("*** Benchmark selection algorithms...")
        labels = {}
        num_tasks_per_puzzle = {}
        num_solved_keys = 0
        num_total_keys = 0
        for basekey, basevalues in self.decoded_results.items():
            mult_key, mult_sub = basekey.split("_")
            num_tasks_per_puzzle[mult_key] = max(num_tasks_per_puzzle.get(mult_key, 0), int(mult_sub) + 1)
            labels[basekey] = correct_solution = self.dataset.replies[basekey][0]
            for subkey, sample in basevalues.items():
                solution = sample["solution"]
                if np.shape(correct_solution) != np.shape(solution):
                    pass
                elif np.array_equal(correct_solution, solution):
                    num_solved_keys += 1
                num_total_keys += 1
        print(f" subkeys: {num_solved_keys}/{num_total_keys}")
        num_puzzles = len(num_tasks_per_puzzle)
        for selection_algorithm in selection_algorithms:
            name = selection_algorithm.__name__
            selected = self.run_selection_algo(selection_algorithm)
            correct_puzzles = {k for k, v in selected.items()
                               if any(np.array_equal(guess, labels[k]) for guess in v[:self.n_guesses])}
            score = sum(1 / num_tasks_per_puzzle[k.split("_")[0]] for k in correct_puzzles)
            print(f" acc: {score:5.1f}/{num_puzzles:3} ('{name}')")
