from sklearn.neighbors import NearestNeighbors
import torch
from tqdm.auto import tqdm


def get_neighbors(edge_index: torch.Tensor, attr: torch.Tensor):
    unique_nodes = edge_index.flatten().unique()

    # Fit KNN
    max_neighbors = (
        (edge_index == edge_index.mode().values.unsqueeze(-1))
        .sum(dim=1)
        .max()
        .item()
    )
    knn = NearestNeighbors(n_neighbors=max_neighbors + 1)
    knn.fit(attr)

    all_attribute_neighbors = knn.kneighbors(attr, return_distance=False)

    attribute_edge_index = []

    num_nodes = edge_index.max() + 1
    degrees = torch.zeros(num_nodes)

    for u in tqdm(unique_nodes, desc="Finding neighbors", leave=False):
        graph_neighbors = torch.cat([
            edge_index[:, edge_index[0] == u][1],  # Outgoing graph neighbors
            edge_index[:, edge_index[1] == u][0],  # Incoming graph neighbors
        ]).unique()

        num_neighbors = len(graph_neighbors)

        attr_neighbors = torch.tensor(
            all_attribute_neighbors[u, 1:(1 + num_neighbors)]
        )

        for an in attr_neighbors.tolist():
            attribute_edge_index.append((u.item(), an))

        degrees[u] = num_neighbors

    inv_degrees = 1. / degrees
    inv_degrees[torch.isinf(inv_degrees)] = 0
    attribute_edge_index = torch.tensor(attribute_edge_index).t()

    return edge_index, attribute_edge_index, inv_degrees
