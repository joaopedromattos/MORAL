from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor
from torch_geometric.utils import coalesce

from datasets import Facebook, Google, German, Nba, Pokec_n, Pokec_z, Credit


def to_torch_sparse_tensor(
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    size: Optional[Union[int, Tuple[int, int]]] = None,
) -> Tensor:
    """Convert an edge index representation into a sparse COO tensor."""

    if size is None:
        num_nodes = int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
        size = (num_nodes, num_nodes)
    elif isinstance(size, int):
        size = (size, size)

    num_rows, num_cols = size
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_rows, num_cols)
    if edge_attr is None:
        edge_attr = torch.ones(edge_index.size(1), device=edge_index.device)

    sparse = torch.sparse_coo_tensor(edge_index, edge_attr, size=size, device=edge_index.device)
    return sparse.coalesce()


def get_dataset(dataset: str, splits_dir: Union[str, Path]) -> Tuple:
    """Load dataset tensors together with pre-computed edge splits."""

    splits_path = Path(splits_dir) / f"{dataset}.pt"
    if not splits_path.exists():
        raise FileNotFoundError(
            f"Could not find edge splits for '{dataset}'. Expected file at '{splits_path}'."
        )

    data, splits = torch.load(splits_path)

    dataset_map: Dict[str, object] = {
        "facebook": Facebook,
        "gplus": Google,
        "german": German,
        "nba": Nba,
        "pokec_n": Pokec_n,
        "pokec_z": Pokec_z,
        "credit": Credit,
    }
    try:
        dataset_obj = dataset_map[dataset]()
    except KeyError as exc:
        raise ValueError(f"Unknown dataset '{dataset}'.") from exc

    adj = to_torch_sparse_tensor(splits["train"]["edge"].t())
    features = dataset_obj.features()
    idx_train = dataset_obj.idx_train()
    idx_val = dataset_obj.idx_val()
    idx_test = dataset_obj.idx_test()
    labels = dataset_obj.labels()
    sens = dataset_obj.sens()
    sens_idx = dataset_obj.sens_idx()

    return adj, features, idx_train, idx_val, idx_test, labels, sens, sens_idx, data, splits
