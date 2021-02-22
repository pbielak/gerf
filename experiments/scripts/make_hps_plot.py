import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def make_hps_plot(hps_log_data: pd.DataFrame) -> plt.Figure:
    fig, axs = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(15, 5),
        sharex=True,
        sharey=True,
    )

    for col, (method_key, method_name) in enumerate(
        (
            ("line", "LINE"),
            ("n2v", "Node2vec"),
            ("sdne", "SDNE"),
        )
    ):
        data = hps_log_data[hps_log_data["emb_method"] == method_key].copy()
        data["val_auc"] = data["val_auc"] * 100.

        best = data[data["val_auc"] == data["val_auc"].max()]
        best_alpha = best["alpha"].values[0]
        best_beta = best["beta"].values[0]
        best_auc = best["val_auc"].values[0]

        alpha = data.groupby("alpha").agg(["mean", "std"])["val_auc"].fillna(0)
        axs[0, col].errorbar(alpha.index, alpha["mean"], yerr=alpha["std"])
        axs[0, col].set(
            title=(
                f"{method_name}\n"
                f"Best: $\lambda_G$ = {best_alpha}, $\lambda_X$ = {best_beta}, "
                f"AUC = {best_auc:.2f} [%]"
            ),
            xlabel="$\lambda_G$",
            ylabel="AUC [%]" if col == 0 else "",
            xticks=alpha.index,
        )

        beta = data.groupby("beta").agg(["mean", "std"])["val_auc"].fillna(0)
        axs[1, col].errorbar(beta.index, beta["mean"], yerr=beta["std"])
        axs[1, col].set(
            xlabel="$\lambda_X$",
            ylabel="AUC [%]" if col == 0 else "",
            xticks=beta.index,
        )

    fig.tight_layout()
    return fig


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", help="Name of the dataset", required=True)

    return parser.parse_args()


def main():
    args = get_args()

    sns.set("paper")

    hps_log_data = pd.read_csv(f"data/hps/GERF/{args.dataset}/log.csv")
    output_path = f"data/plots/hps_{args.dataset}.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig = make_hps_plot(hps_log_data=hps_log_data)
    fig.savefig(output_path)


if __name__ == "__main__":
    main()
