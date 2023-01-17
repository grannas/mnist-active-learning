import getopt
import json
import sys
from importlib import import_module

import numpy as np
from scipy import stats as st


class Experiment:
    """
    Repeats a specific training regime a set number of times and stores the performance
    metrics.

    Arguments:
        name (str) : name for the experimental setup
        method (str) : name of the MNISTTrainer sub-class that will be used
        params (dict) : parameter set used as input to the MNISTTrainer
        n (int) : number of times the training is repeated
        data_dir (str) : directory where the mnist data .jsons are stored
    """

    def __init__(self, name, method, params, n, data_dir):
        self.name = name
        self.method = getattr(import_module("active_learning"), method)
        params["data_dir"] = data_dir
        self.params = params
        self.n = n
        self.experiment_results = []

    def perform_experiment(self):
        for _ in range(self.n):
            self.experiment_results.append(
                self.method(**self.params).run_experiment()[0]
            )

    def return_stats(self, metric="test_acc"):
        vals = [results_dict[metric] for results_dict in self.experiment_results]
        return vals


def main(setup_file, data_dir):
    with open(setup_file, "r") as f:
        exp_setup = json.load(f)
        exp_results = []

        for exp_key in exp_setup:
            e = Experiment(
                name=exp_key,
                method=exp_setup[exp_key]["method"],
                params=exp_setup[exp_key]["train_params"],
                n=exp_setup[exp_key]["n"],
                data_dir=data_dir,
            )
            e.perform_experiment()
            stats = e.return_stats()
            exp_results.append([exp_key, stats])

        print()
        print("Experimental results")
        print("Experiment\tMean\tStd\tConf. Int. 99%")
        for exp_name, stats in exp_results:
            mean = np.mean(stats)
            std_dev = np.std(stats)
            c_int = st.t.interval(
                confidence=0.99, df=len(stats), loc=np.mean(stats), scale=st.sem(stats)
            )
            print(
                f"{exp_name}\t{mean:.4f}\t{std_dev:.5f}\t{c_int[0]:.4f}-{c_int[1]:.4f}"
            )


if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "s:d", ["setup-file=", "data-dir="])
    except getopt.GetoptError as e:
        print(e)

    setup_file = "experimental_setup.json"
    data_dir = "MNIST"
    verbose = False

    for opt, arg in opts:
        if opt in ("-s", "--setup-file"):
            setup_file = arg
        elif opt in ("-d", "--data-dir"):
            data_dir = arg

    main(setup_file, data_dir)
