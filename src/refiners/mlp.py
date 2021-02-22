import numpy as np
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

from src.refiners.base import BaseRefiner


class MLPRefiner(BaseRefiner):
    """Encode attributes by training MLP autoencoder."""

    def __init__(
        self,
        lr: float,
        batch_size: int,
        num_epochs: int,
    ):
        self._lr = lr
        self._batch_size = batch_size
        self._num_epochs = num_epochs

    def refine(
        self,
        edge_index: torch.Tensor,
        emb: torch.Tensor,
        attr: torch.Tensor,
    ) -> torch.Tensor:
        mlp = MLPAutoencoder(
            data_dim=emb.size(-1) + attr.size(-1),
            hidden_dim=emb.size(-1),
            lr=self._lr,
            batch_size=self._batch_size,
        )

        emb_with_attr = torch.cat([emb, attr], dim=-1)

        mlp.fit(data=emb_with_attr, num_epochs=self._num_epochs)

        return mlp.predict(emb_with_attr)


class MLPAutoencoder(nn.Module):

    def __init__(
        self,
        data_dim: int,
        hidden_dim: int,
        lr: float = 1e-3,
        batch_size: int = 128,
    ):
        super().__init__()
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        d = (data_dim + hidden_dim) // 2
        self._encoder = nn.Sequential(
            nn.Linear(in_features=data_dim, out_features=data_dim),
            nn.ReLU(),
            nn.Linear(in_features=data_dim, out_features=d),
            nn.ReLU(),
            nn.Linear(in_features=d, out_features=hidden_dim),
            nn.Tanh(),
        ).to(self._device)

        self._decoder = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=data_dim),
        ).to(self._device)

        self._batch_size = batch_size
        self._optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)
        self._loss_fn = torch.nn.MSELoss()

    def fit(self, data: torch.Tensor, num_epochs: int):
        self.train()
        loader = DataLoader(
            dataset=data.to(self._device),
            batch_size=self._batch_size,
            shuffle=True,
        )

        pbar = tqdm(range(num_epochs), desc="Epochs")
        losses = []

        for _ in pbar:
            epoch_loss = 0
            for batch in tqdm(loader, desc="Batches", leave=False):
                self._optimizer.zero_grad()

                pred = self._decoder(self._encoder(batch))

                loss = self._loss_fn(input=pred, target=batch)
                epoch_loss += loss.item()

                loss.backward()
                self._optimizer.step()

            losses.append(epoch_loss / len(loader))

            pbar.set_postfix({
                'first_loss': losses[0],
                'current_loss': losses[-1],
                'mean_loss': np.mean(losses),
            })

        return self

    def predict(self, data: torch.Tensor) -> torch.Tensor:
        self.eval()
        loader = DataLoader(
            dataset=data.to(self._device),
            batch_size=self._batch_size,
            shuffle=False,
        )

        z = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Batches", leave=False):
                z.append(self._encoder(batch).cpu())

        return torch.cat(z, dim=0)
