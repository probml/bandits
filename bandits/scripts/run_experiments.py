import jax
import ml_collections
import glob
from datetime import datetime

from . import movielens_exp as movielens_run
from . import mnist_exp as mnist_run
from . import tabular_exp as tabular_run
from . import tabular_subspace_exp as tabular_sub_run

def make_config(filepath):
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()
  config.filepath = filepath
  config.ntrials = 10
  return config


def main():
    timestamp = datetime.timestamp(datetime.now())
    experiments = [tabular_run, mnist_run, movielens_run, tabular_sub_run]
    experiments_name = ["tabular", "mnist", "movielens", "tabular_subspace"]

    for experiment_run, experiment_name in zip(experiments, experiments_name):
        filename = f"./bandits/results/{experiment_name}_results_{timestamp}.csv"
        config = make_config(filename)
        experiment_run.main(config)
    

if __name__ == "__main__":
    main()