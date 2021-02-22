"""Script for evaluation of embeddings in node classification."""
import argparse
import json
import os
from typing import Generator

import torch
from tqdm import tqdm
import yaml

from src import DATA_DIR
from src.datasets import load_citation_dataset
from src.tasks.node_classification import evaluate_node_classification


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", help="Name of the dataset", required=True)

    return parser.parse_args()


def load_latents(
    method: str,
    dataset: str,
) -> Generator[torch.Tensor, None, None]:
    # Load dataset
    data = load_citation_dataset(name=dataset)

    if method == "features":  # Use features only
        z = data.x

        yield data, z
    else:  # Read embedding
        embedding_dir = os.path.join(
            DATA_DIR,
            f"embeddings/{method}/{dataset}/",
        )

        for fname in sorted(os.listdir(embedding_dir)):
            z = torch.load(f=os.path.join(embedding_dir, fname))
            yield data, z


def main():
    # Get params
    with open("experiments/configs/node_classification.yaml", "r") as fin:
        cfg = yaml.safe_load(fin)

    args = get_args()

    dataset = args.dataset

    for method in tqdm(cfg["embedding_methods"], desc="Embedding methods"):
        all_metrics = []

        for data, z in load_latents(method=method, dataset=dataset):
            all_metrics.append(evaluate_node_classification(z=z, data=data))

        # Save
        method_fname = method.replace("/", "_")
        metrics_path = os.path.join(
            DATA_DIR,
            f"metrics/node_classification/{dataset}/{method_fname}.json",
        )
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

        with open(metrics_path, "w") as fout:
            json.dump(obj=all_metrics, fp=fout, indent=4)


if __name__ == '__main__':
    main()
