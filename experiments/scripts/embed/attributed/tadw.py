"""Run TADW method to obtain attributed node embeddings."""
import argparse
import os
from typing import Union

from karateclub import TADW
import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix
import torch
from torch_geometric.utils import to_networkx
from tqdm import trange
import yaml

from src import DATA_DIR
from src.datasets import load_citation_dataset


class _TADW(TADW):

    def fit(
        self,
        graph: nx.classes.graph.Graph,
        X: Union[np.array, coo_matrix],
    ):
        """This implementation adds a progress bar and more verbosity."""
        self._set_seed()
        self._check_graph(graph)

        print("Creating target matrix A...")
        self._A = self._create_target_matrix(graph)

        print("Creating reduced features T...")
        self._T = self._create_reduced_features(X)

        self._init_weights()

        for _ in trange(self.iterations, desc="Iterations"):
            self._update_W()
            self._update_H()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num-retrains", type=int, required=True)

    return parser.parse_args()


def main():
    # Read config
    with open("experiments/configs/embed/attributed/tadw.yaml", 'r') as fin:
        cfg = yaml.safe_load(fin)

    args = get_args()

    dataset_name = args.dataset
    num_retrains = args.num_retrains

    data = load_citation_dataset(name=dataset_name)
    params = cfg["params"][dataset_name]

    for idx in trange(num_retrains, desc="Retrain"):
        graph = to_networkx(data=data, to_undirected=True)

        # Build model
        tadw = _TADW(
            dimensions=params["emb_dim"],
            iterations=params["num_epochs"],
            alpha=params["lr"],
        )

        tadw.allow_disjoint = True

        # Train model
        tadw.fit(graph=graph, X=data.x.numpy())

        # Get embeddings
        z = torch.from_numpy(tadw.get_embedding()).float()

        # Save embeddings
        embedding_path = os.path.join(
            DATA_DIR,
            cfg["paths"]["output"]["embedding"]
            .replace("${name}", dataset_name)
            .replace("${idx}", str(idx)),
        )
        os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
        torch.save(obj=z, f=embedding_path)


if __name__ == '__main__':
    main()
