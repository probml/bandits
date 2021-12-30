# Efficient Online Bayesian Inference for Neural Bandits

By [Gerardo Durán-Martín](http://github.com/gerdm), [Aleyna Kara](https://github.com/karalleyna), and [Kevin Murphy](https://github.com/murphyk)  
URL: https://arxiv.org/abs/2112.00195

<img width="907" alt="MNIST-experiment" src="https://user-images.githubusercontent.com/4108759/144386660-df6b83fa-992b-4de1-b5fd-f6f784bbb160.png">

-----

## Reproduce the results

There are two ways to reproduce the results from the paper

### Run the scripts

To reproduce the results, `cd` into the project folder and run

```bash
python bandits test
```

```bash
python bandits run_and_plot
```

### Step by step

If you only want to reproduce the results, run

```bash
python bandits run_experiments
```

If you have previously reproduced the results and want to reproduce the plots, run

```bash
python bandits plot_experiments
```

The results will be stored inside `bandits/figures/`.

### Execute the notebooks

An alternative way to reproduce the results is to simply open and run [`subspace_bandits.ipynb`](https://github.com/probml/bandits/blob/main/bandits/scripts/subspace_bandits.ipynb) 
