from abc import ABC, abstractmethod

import torch


class BaseRefiner(ABC):

    @abstractmethod
    def refine(
        self,
        edge_index: torch.Tensor,
        emb: torch.Tensor,
        attr: torch.Tensor,
    ) -> torch.Tensor:
        pass
