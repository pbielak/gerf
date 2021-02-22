from src.refiners.base import BaseRefiner
from src.refiners.concat import ConcatRefiner, ConcatPCARefiner
from src.refiners.mlp import MLPRefiner


def get_refiner(name: str, config: dict) -> BaseRefiner:
    refiners = {
        "Concat": ConcatRefiner,
        "ConcatPCA": ConcatPCARefiner,
        "MLP": MLPRefiner,
    }

    if name not in refiners.keys():
        raise ValueError(f"Unknown refiner: \"{name}\"")

    return refiners[name](**config)
