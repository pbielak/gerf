"""SDNE embedding using the GE (Graph Embedding) package."""
import argparse
import os
from typing import Optional

from ge import SDNE
import networkx as nx
import torch
import yaml
from torch_geometric.utils import to_networkx
from tqdm import trange

from src import DATA_DIR
from src.datasets import load_citation_dataset


class SDNEModel:

    def __init__(
        self,
        graph: nx.Graph,
        emb_dim: int,
        hidden_size: int,
        alpha: float,
        beta: float,
        nu1: float,
        nu2: float,
        batch_size: int,
    ):
        self._model = SDNE(
            graph=graph,
            hidden_size=[hidden_size, emb_dim],
            alpha=alpha,
            beta=beta,
            nu1=nu1,
            nu2=nu2,
        )
        self._batch_size = batch_size

    def fit(self, num_epochs: int) -> dict:
        hist = self._model.train(
            batch_size=self._batch_size,
            epochs=num_epochs,
            verbose=2,
        )

        losses = hist.history
        return losses

    def predict(
        self,
        nodes: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        emb = self._model.get_embeddings()

        if nodes is None:
            nodes = sorted(emb.keys())

        embeddings = torch.stack([
            torch.tensor(emb[node], dtype=torch.float)
            for node in nodes
        ], dim=0)

        return embeddings


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num-retrains", type=int, required=True)

    return parser.parse_args()


def main():
    # Read config
    with open("experiments/configs/embed/structural/sdne.yaml", 'r') as fin:
        cfg = yaml.safe_load(fin)

    args = get_args()

    dataset_name = args.dataset
    num_retrains = args.num_retrains

    data = load_citation_dataset(name=dataset_name)
    params = cfg["params"][dataset_name]

    for idx in trange(num_retrains, desc="Retrain"):
        graph = to_networkx(data=data, to_undirected=True)

        # Build model
        line = SDNEModel(
            graph=graph,
            emb_dim=params["emb_dim"],
            hidden_size=params["hidden_size"],
            alpha=params["alpha"],
            beta=params["beta"],
            nu1=params["nu1"],
            nu2=params["nu2"],
            batch_size=params["batch_size"],
        )

        # Train model
        line.fit(num_epochs=params["num_epochs"])

        # Get embeddings
        z = line.predict()

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
