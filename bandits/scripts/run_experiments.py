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


def main(experiment=None):
    timestamp = datetime.timestamp(datetime.now())

    experiments = {
        "tabular": tabular_run,
        "mnist": mnist_run,
        "movielens": movielens_run,
        "tabular_subspace": tabular_sub_run
    }

    if experiment is not None:
        print(experiment)
        if experiment not in experiments:
          err = f"Experiment {experiment} not found. "
          err += f"Available experiments: {list(experiments.keys())}"
          raise ValueError(err)
        experiments = {experiment: experiments[experiment]}

    for experiment_name, experiment_run in experiments.items():
        filename = f"./bandits/results/{experiment_name}_results_{timestamp}.csv"
        config = make_config(filename)
        experiment_run.main(config)
    

if __name__ == "__main__":
    main()