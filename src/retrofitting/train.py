import torch
from torch import nn
from torch_geometric.data import Data
from tqdm.auto import tqdm

from src.retrofitting.loss import GRLoss


def retrofit(
    data: Data,
    embedding: torch.Tensor,
    alpha: float,
    beta: float,
    lr: float,
    num_epochs: int,
) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RetrofittedEmbedding(z=embedding)

    model = model.to(device)
    embedding = embedding.to(device)

    loss_fn = GRLoss(
        edge_index=data.edge_index,
        embedding=embedding,
        attributes=data.x,
        alpha=alpha,
        beta=beta,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    pbar = tqdm(iterable=range(num_epochs), leave=False)

    for epoch in pbar:
        optimizer.zero_grad()

        z_star = model(z=embedding)

        loss = loss_fn(z_star=z_star)

        loss.backward(retain_graph=True)
        optimizer.step()

    return model


class RetrofittedEmbedding(nn.Module):

    def __init__(self, z: torch.Tensor):
        super().__init__()
        self.z_star = torch.nn.Parameter(z.clone())

    def forward(self, z: torch.Tensor):
        return self.z_star
