import fire
from experiments import plot_results
from experiments import run_experiments


class Experiments:
    def plot(self):
        plot_results.main()
    
    def run(self):
        run_experiments.main()

if __name__ == "__main__":
    fire.Fire(Experiments)