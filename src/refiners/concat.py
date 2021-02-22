from sklearn import decomposition as sk_dec
import torch

from src.refiners.base import BaseRefiner


class ConcatRefiner(BaseRefiner):
    """Encode attributes by concatenation to embeddings."""

    def refine(
        self,
        edge_index: torch.Tensor,
        emb: torch.Tensor,
        attr: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([emb, attr], dim=-1)


class ConcatPCARefiner(BaseRefiner):
    """Encode attributes by concatenation followed by PCA."""

    def refine(
        self,
        edge_index: torch.Tensor,
        emb: torch.Tensor,
        attr: torch.Tensor,
    ) -> torch.Tensor:
        emb_dim = emb.size(-1)

        emb_with_attr = torch.cat([emb, attr], dim=-1)
        emb_with_attr_low_dim = (
            sk_dec
            .PCA(n_components=emb_dim)
            .fit_transform(emb_with_attr)
        )

        return torch.from_numpy(emb_with_attr_low_dim).type(emb.dtype)
