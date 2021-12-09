import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os
os.chdir("./bandits")

def plot_figure(data, x, y, filename, figsize=(24, 9), log_scale=False):   
    sns.set(font_scale=1.5)
    plt.style.use("seaborn-poster")

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    g = sns.barplot(x=x, y=y, hue="Method", data=data, errwidth=2, ax=ax, palette=colors)
    if log_scale:
        g.set_yscale("log")
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    plt.savefig(f"./figures/{filename}.png")
    plt.show()

def read_data(dataset_name):
    *_, filename = sorted(glob.glob(f"./results/{dataset_name}_results*.csv"))
    df = pd.read_csv(filename)
    if dataset_name=="mnist":
        linear_df = df[(df["Method"]=="Lin-KF") | (df["Method"]=="Lin")].copy()
        linear_df["Model"] = "MLP2"
        df = df.append(linear_df)
        linear_df["Model"] = "LeNet5"
        df = df.append(linear_df)

    by = ["Rank"] if dataset_name=="tabular" else ["Rank", "AltRank"]

    data_up = df.sort_values(by=by).copy()
    data_down = df.sort_values(by=by).copy()

    data_up["Reward"] = data_up["Reward"] + data_up["Std"]
    data_down["Reward"] = data_down["Reward"] - data_down["Std"]
    data = pd.concat([data_up, data_down])
    return data

def plot_subspace_figure(df, filename=None):
    df = df.reset_index().drop(columns=["index"])
    plt.style.use("seaborn-darkgrid")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(x="Subspace Dim", y="Reward", hue="Method", marker="o", data=df)
    lines, labels = ax.get_legend_handles_labels()
    for line, method in zip(lines, labels):
        data = df[df["Method"]==method]
        color = line.get_c()
        y_lower_bound =  data["Reward"] -  data["Std"]
        y_upper_bound = data["Reward"] + data["Std"]
        ax.fill_between(data["Subspace Dim"],  y_lower_bound, y_upper_bound, color=color, alpha=0.3)

    ax.set_ylabel("Reward", fontsize=16)
    plt.setp(ax.get_xticklabels(), fontsize=16) 
    plt.setp(ax.get_yticklabels(), fontsize=16) 
    ax.set_xlabel("Subspace Dimension(d)", fontsize=16)
    dataset = df.iloc[0]["Dataset"]
    ax.set_title(f"{dataset.title()} - Subspace Dim vs. Reward", fontsize=18)
    legend = ax.legend(loc="lower right", prop={'size': 16},frameon=1)
    frame = legend.get_frame()
    frame.set_color('white')
    frame.set_alpha(0.6)
    
    file_path = "./figures/"
    file_path = file_path + f"{dataset}_sub_reward.png" if filename is None else file_path + f"{filename}.png"
    plt.savefig(file_path)

method_ordering = {"EKF-Sub-SVD": 0,
                   "EKF-Sub-RND": 1,
                   "EKF-Sub-Diag-SVD": 2,
                   "EKF-Sub-Diag-RND": 3,
                   "EKF-Orig-Full": 4,
                   "EKF-Orig-Diag": 5,
                   "NL-Lim": 6,
                   "NL-Unlim": 7,
                   "Lin": 8,
                   "Lin-KF": 9,
                   "Lin-Wide": 9,
                   "Lim2": 10,
                   "NeuralTS": 11}
                   
colors = {k : sns.color_palette("Paired")[v]
          if k!="Lin-KF" else  sns.color_palette("tab20")[8]
          for k,v in method_ordering.items()}
    
dataset_info = {
    "mnist": {
        "elements": ["EKF-Sub-SVD", "EKF-Sub-RND", "EKF-Sub-Diag-SVD",
                    "EKF-Sub-Diag-RND", "EKF-Orig-Diag", "NL-Lim",
                    "NL-Unlim", "Lin"],
        "x": "Model",
    },

    "tabular": {
        "elements": ["EKF-Sub-SVD", "EKF-Sub-RND", "EKF-Sub-Diag-SVD",
                    "EKF-Sub-Diag-RND", "EKF-Orig-Diag", "NL-Lim",
                    "NL-Unlim", "Lin"],
        "x": "Dataset"
    },
    
    "movielens": {
        "elements": ["EKF-Sub-SVD", "EKF-Sub-RND", "EKF-Sub-Diag-SVD",
                    "EKF-Sub-Diag-RND", "EKF-Orig-Diag", "NL-Lim",
                    "NL-Unlim", "Lin"],
        "x": "Model"
        },
}


plot_configs = [
    {"metric": "Reward", "log_scale":False},
    {"metric": "Time", "log_scale":True},

]


def main():
    # Create reward / runnnig time experiments
    print("Plotting reward / running time")
    for dataset_name in dataset_info:
        print(dataset_name)
        info = dataset_info[dataset_name]
        methods = info["elements"]
        x = info["x"]

        df = read_data(dataset_name)
        df = df[df["Method"].isin(methods)]

        for config in plot_configs:
            metric = config["metric"]
            use_log_scale = config["log_scale"]

            filename = f"{dataset_name}_{metric.lower()}"
            plot_figure(df, x, metric, filename, log_scale=use_log_scale)


    # Plot subspace-dim v.s. reward
    print("Plotting subspace dim v.s. reward")
    *_, filename = sorted(glob.glob(f"./results/tabular_subspace_results*.csv"))
    tabular_sub_df = pd.read_csv(filename)

    datasets = ["shuttle", "adult", "covertype"]
    for dataset_name in datasets:
        print(dataset_name)
        subdf = tabular_sub_df.query("Dataset == @dataset_name")
        plot_subspace_figure(subdf)


if __name__ == "__main__":
    main()