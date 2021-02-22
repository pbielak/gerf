"""Run DeepGraphInfomax (based on the example in PyTorch-Geometric)."""
import argparse
import os

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, DeepGraphInfomax
from tqdm import trange
import yaml

from src import DATA_DIR
from src.datasets import load_citation_dataset


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=True)
        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        return x


def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index


class DGI:

    def __init__(
        self,
        num_node_features: int,
        emb_dim: int,
        lr: float,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = DeepGraphInfomax(
            hidden_channels=emb_dim,
            encoder=Encoder(num_node_features, emb_dim),
            summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
            corruption=corruption,
        ).to(device)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)

    def train(self, data: Data, num_epochs: int):
        self._model.train()

        pbar = trange(num_epochs, desc="Epochs", leave=False)
        losses = []

        for _ in pbar:
            self._optimizer.zero_grad()

            pos_z, neg_z, summary = self._model(data.x, data.edge_index)

            loss = self._model.loss(pos_z, neg_z, summary)
            losses.append(loss.item())

            loss.backward()
            self._optimizer.step()

            pbar.set_postfix({
                "first_loss": losses[0],
                "current_loss": losses[-1],
                "mean_loss": np.mean(losses),
            })

    @torch.no_grad()
    def predict(self, data: Data) -> torch.Tensor:
        self._model.eval()
        z, _, _ = self._model(data.x, data.edge_index)
        return z.cpu().float()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num-retrains", type=int, required=True)

    return parser.parse_args()


def main():
    # Read config
    with open("experiments/configs/embed/attributed/dgi.yaml", 'r') as fin:
        cfg = yaml.safe_load(fin)

    args = get_args()

    dataset_name = args.dataset
    num_retrains = args.num_retrains

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_citation_dataset(name=dataset_name).to(device)
    params = cfg["params"][dataset_name]

    for idx in trange(num_retrains, desc="Retrain"):
        # Build model
        dgi = DGI(
            num_node_features=data.num_node_features,
            emb_dim=params["emb_dim"],
            lr=params["lr"],
        )

        # Train model
        dgi.train(data=data, num_epochs=params["num_epochs"])

        # Get embeddings
        z = dgi.predict(data=data)

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
