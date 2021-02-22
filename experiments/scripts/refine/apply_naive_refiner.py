"""Apply refiner and encode features in embeddings."""
import argparse
import os

import torch
from tqdm import tqdm
import yaml

from src import DATA_DIR
from src.datasets import load_citation_dataset
from src.refiners import get_refiner


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        help="Name of the dataset",
        required=True,
    )
    parser.add_argument(
        "--method",
        help="Name of the refinement method",
        required=True,
    )

    return parser.parse_args()


def main():
    # Get params
    args = get_args()

    dataset = args.dataset
    method = args.method

    with open(f"experiments/configs/refine/naive/{method}.yaml", "r") as fin:
        cfg = yaml.safe_load(fin)

    # Load dataset
    data = load_citation_dataset(name=dataset)

    # Refine
    for emb_method in tqdm(
        iterable=cfg["structural_embedding_methods"],
        desc="Embedding method",
    ):
        # Read embedding
        embeddings_dir = os.path.join(
            DATA_DIR,
            f"embeddings/structural/{emb_method}/{dataset}/",
        )

        for fname in tqdm(
            iterable=sorted(os.listdir(embeddings_dir)),
            desc="Embedding",
            leave=False,
        ):
            z = torch.load(f=os.path.join(embeddings_dir, fname))

            # Refine embeddings
            refiner = get_refiner(name=method, config=cfg.get(dataset, {}))

            try:
                z_star = refiner.refine(
                edge_index=data.edge_index,
                emb=z,
                attr=data.x,
                )
            except Exception as e:
                __import__("pdb").set_trace()
                print(e)

            # Save embeddings
            refined_embedding_path = os.path.join(
                cfg["paths"]["output"]["embedding"]
                .replace("${dataset}", dataset)
                .replace("${emb_method}", emb_method),
                fname
            )
            os.makedirs(os.path.dirname(refined_embedding_path), exist_ok=True)

            torch.save(obj=z_star, f=refined_embedding_path)


if __name__ == "__main__":
    main()
