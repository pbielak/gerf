import os
from torch_geometric.datasets import Amazon, Coauthor, WikiCS
from torch_geometric.data import Data
from torch_geometric import transforms as T


from src import REPO_ROOT


def load_citation_dataset(name: str) -> Data:
    if name == "WikiCS":
        data = load_WikiCS()
    elif name == "Amazon-CS":
        data = load_AmazonCS()
    elif name == "Amazon-Photo":
        data = load_AmazonPhoto()
    elif name == "Coauthor-CS":
        data = load_CoauthorCS()
    else:
        raise RuntimeError(f"Unknown dataset: {name}")

    return data


def load_WikiCS() -> Data:
    data = WikiCS(
        root=os.path.join(REPO_ROOT, "data/datasets/WikiCS/"),
        transform=T.NormalizeFeatures(),
    )[0]
    data.train_mask = data.train_mask[:, 0]
    data.val_mask = data.val_mask[:, 0]

    return data


def load_AmazonCS() -> Data:
    data = Amazon(
        root=os.path.join(REPO_ROOT, "data/datasets/AmazonCS/"),
        name="computers",
        transform=T.NormalizeFeatures(),
        pre_transform=T.AddTrainValTestMask(
            split="train_rest",
            num_splits=1,
            num_val=0.1,
            num_test=0.8,
        ),
    )[0]

    return data


def load_AmazonPhoto() -> Data:
    data = Amazon(
        root=os.path.join(REPO_ROOT, "data/datasets/AmazonPhoto/"),
        name="photo",
        transform=T.NormalizeFeatures(),
        pre_transform=T.AddTrainValTestMask(
            split="train_rest",
            num_splits=1,
            num_val=0.1,
            num_test=0.8,
        ),
    )[0]

    return data


def load_CoauthorCS() -> Data:
    data = Coauthor(
        root=os.path.join(REPO_ROOT, "data/datasets/CoauthorCS/"),
        name="cs",
        transform=T.NormalizeFeatures(),
        pre_transform=T.AddTrainValTestMask(
            split="train_rest",
            num_splits=1,
            num_val=0.1,
            num_test=0.8,
        ),
    )[0]

    return data
