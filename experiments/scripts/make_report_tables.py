import json
import os
from typing import List

import numpy as np
import pandas as pd

from src import DATA_DIR


def make_metric_table(
    task: str,
    methods: List[str],
    datasets: List[str],
    metric_name: str,
) -> pd.DataFrame:
    def _select_metric(v):
        if metric_name in ('auc', 'accuracy'):
            return v[metric_name]
        elif "/" in metric_name:
            avg_type, metric = metric_name.split("/")
            return v[avg_type][metric]
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

    records = []

    for dataset in datasets:
        for method in methods:
            method_fname = method.replace("/", "_")
            metrics_path = os.path.join(
                DATA_DIR,
                f"metrics/{task}/{dataset}/{method_fname}.json",
            )

            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as fin:
                    metrics = json.load(fin)

                test_metrics = [
                    _select_metric(m["test"]) * 100.0
                    for m in metrics
                ]

                mean = np.mean(test_metrics)
                std = (
                    np.std(test_metrics, ddof=1)
                    if len(test_metrics) > 1
                    else 0.0
                )
                value = f"{mean:.2f} +/- {std:.2f}"
            else:
                value = np.nan

            records.append({
                "dataset": dataset,
                "method": method,
                "value": value,
            })

    df = (
        pd.DataFrame.from_records(records)
        .pivot(index="dataset", columns="method", values="value")
        .reindex(columns=methods, index=datasets)
        .transpose()
    )
    return df


def main():
    tasks = [
        "node_classification",
    ]

    methods = [
        # Features only
        "features",

        # Node2vec based
        "structural/n2v",
        "refined/naive/Concat/n2v",
        "refined/naive/ConcatPCA/n2v",
        "refined/naive/MLP/n2v",
        "refined/GERF/n2v",

        # LINE based
        "structural/line",
        "refined/naive/Concat/line",
        "refined/naive/ConcatPCA/line",
        "refined/naive/MLP/line",
        "refined/GERF/line",

        # SDNE based
        "structural/sdne",
        "refined/naive/Concat/sdne",
        "refined/naive/ConcatPCA/sdne",
        "refined/naive/MLP/sdne",
        "refined/GERF/sdne",

        # Attributed
        "attributed/tadw",
        "attributed/fscnmf",
        "attributed/dgi",
    ]

    datasets = [
        "WikiCS",
        "Amazon-CS",
        "Amazon-Photo",
        "Coauthor-CS",
    ]

    metric_names = [
        "auc",
    ]

    for task in tasks:
        report_path = os.path.join(DATA_DIR, f"report_{task}.txt")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        with open(report_path, "w") as fout:
            for metric_name in metric_names:
                df = make_metric_table(
                    task=task,
                    datasets=datasets,
                    methods=methods,
                    metric_name=metric_name,
                )
                fout.write(f"----- {metric_name} -----\n")
                fout.write(df.to_string())
                fout.write("\n\n")


if __name__ == "__main__":
    main()
