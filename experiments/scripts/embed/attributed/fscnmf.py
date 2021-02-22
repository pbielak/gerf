"""Run FSCNMF method to obtain attributed node embeddings."""
import argparse
import os
from typing import Union

from karateclub import FSCNMF
import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix
import torch
from torch_geometric.utils import to_networkx
from tqdm.auto import trange
import yaml

from src import DATA_DIR
from src.datasets import load_citation_dataset


class _FSCNMF(FSCNMF):

    def fit(
        self,
        graph: nx.classes.graph.Graph,
        X: Union[np.array, coo_matrix],
    ):
        """This implementation adds a progress bar."""
        self._set_seed()
        self._check_graph(graph)

        self._X = X
        self._A = self._create_base_matrix(graph)

        self._init_weights()

        for _ in trange(self.iterations, desc="Iterations"):
            self._update_B1()
            self._update_B2()
            self._update_U()
            self._update_V()

    def _create_D_inverse(self, graph):
        """
        Creating a sparse inverse degree matrix.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.

        Return types:
            * **D_inverse** *(Scipy array)* - Diagonal inverse degree matrix.
        """
        index = np.arange(graph.number_of_nodes())
        values = np.array([
            1.0/graph.degree[node] if graph.degree[node] > 0 else 0.0
            for node in range(graph.number_of_nodes())
        ])
        shape = (graph.number_of_nodes(), graph.number_of_nodes())
        D_inverse = coo_matrix((values, (index, index)), shape=shape)
        return D_inverse


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num-retrains", type=int, required=True)

    return parser.parse_args()


def main():
    # Read config
    with open("experiments/configs/embed/attributed/fscnmf.yaml", 'r') as fin:
        cfg = yaml.safe_load(fin)

    args = get_args()

    dataset_name = args.dataset
    num_retrains = args.num_retrains

    data = load_citation_dataset(name=dataset_name)
    params = cfg["params"][dataset_name]

    for idx in trange(num_retrains, desc="Retrain"):
        graph = to_networkx(data=data, to_undirected=True)

        # Build model
        fscnmf = _FSCNMF(
            dimensions=params["emb_dim"],
            iterations=params["num_epochs"],
        )

        fscnmf.allow_disjoint = True

        # Train model
        fscnmf.fit(graph=graph, X=data.x.numpy())

        # Get embeddings
        z = torch.from_numpy(fscnmf.get_embedding()).float()

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
