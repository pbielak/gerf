"""Apply graph retrofitting using node attributes."""
import argparse
import json
import os

import torch
from tqdm import tqdm
import yaml

from src import DATA_DIR
from src.datasets import load_citation_dataset
from src.retrofitting.train import retrofit


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        help="Name of the dataset",
        required=True,
    )

    return parser.parse_args()


def main():
    # Get params
    args = get_args()

    dataset = args.dataset

    with open("experiments/configs/refine/GERF.yaml", "r") as fin:
        cfg = yaml.safe_load(fin)

    best_params_path = (
        cfg["paths"]["input"]["best_params"]
        .replace("${dataset}", dataset)
    )
    with open(best_params_path, "r") as fin:
        best_params = json.load(fin)

    # Load dataset
    data = load_citation_dataset(name=dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        alpha = best_params[emb_method]["alpha"]
        beta = best_params[emb_method]["beta"]

        for fname in tqdm(
            iterable=sorted(os.listdir(embeddings_dir)),
            desc="Embedding",
            leave=False,
        ):
            z = torch.load(f=os.path.join(embeddings_dir, fname)).to(device)

            model = retrofit(
                data=data,
                embedding=z,
                alpha=alpha,
                beta=beta,
                lr=0.1,
                num_epochs=100,
            )
            z_star = model(z=z).cpu().detach()

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
