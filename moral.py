"""Implementation of the MORAL model used in the AAAI submission.

The original code base shipped with a large number of experimental features
and partially implemented ideas.  This refactored version focuses on the core
training loop used in the paper: we learn one encoder/decoder pair per
sensitive group and optimise a binary cross entropy loss on balanced batches of
positive and negative edges.

The module is intentionally lightweight to make it easy for researchers to
understand, extend and import.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from loguru import logger
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import GCNConv, GINConv, SAGEConv

NUM_GROUPS = 3


class EdgeLabelDataset(Dataset):
    """PyTorch dataset returning edge indices and binary labels."""

    def __init__(self, edges: Tensor, labels: Tensor) -> None:
        if edges.size(0) != labels.size(0):  # pragma: no cover - sanity check
            raise ValueError("Edges and labels must contain the same number of rows")

        self.edges = edges.long()
        self.labels = labels.float()

    def __len__(self) -> int:  # pragma: no cover - trivial
        return int(self.edges.size(0))

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.edges[idx], self.labels[idx]


class TwoLayerGCN(nn.Module):
    """Small two layer GCN encoder."""

    def __init__(self, in_channels: int, hidden_channels: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


class TwoLayerGIN(nn.Module):
    """Simple GIN based encoder."""

    def __init__(self, in_channels: int, hidden_channels: int, dropout: float = 0.2) -> None:
        super().__init__()
        mlp1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        mlp2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.conv1 = GINConv(mlp1)
        self.conv2 = GINConv(mlp2)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


class TwoLayerSAGE(nn.Module):
    """Simple GraphSAGE encoder."""

    def __init__(self, in_channels: int, hidden_channels: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


class LinkPredictor(nn.Module):
    """Predicts link existence scores from node embeddings."""

    def __init__(self, in_channels: int, hidden_channels: Optional[int] = None) -> None:
        super().__init__()
        if hidden_channels is None:
            self.network = None
        else:
            self.network = nn.Sequential(
                nn.Linear(in_channels * 2, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, 1),
            )

    def forward(self, embeddings: Tensor, edge_index: Tensor) -> Tensor:
        src, dst = edge_index
        features = torch.cat([embeddings[src], embeddings[dst]], dim=-1)
        if self.network is None:
            scores = (embeddings[src] * embeddings[dst]).sum(dim=-1, keepdim=True)
        else:
            scores = self.network(features)
        return scores.view(-1)


def build_encoder(name: str, in_channels: int, hidden_channels: int) -> nn.Module:
    name = name.lower()
    if name == "gcn":
        return TwoLayerGCN(in_channels, hidden_channels)
    if name == "gin":
        return TwoLayerGIN(in_channels, hidden_channels)
    if name == "sage":
        return TwoLayerSAGE(in_channels, hidden_channels)
    raise ValueError(f"Unsupported encoder '{name}'.")


def build_predictor(name: str, hidden_channels: int) -> LinkPredictor:
    name = name.lower()
    if name in {"gae", "dot", "standard"}:
        return LinkPredictor(hidden_channels, hidden_channels=None)
    if name in {"mlp", "mlp_decoder"}:
        return LinkPredictor(hidden_channels, hidden_channels)
    raise ValueError(f"Unsupported decoder '{name}'.")


def normalize_features(features: Tensor) -> Tensor:
    """Column wise min-max normalisation with numerical safeguards."""

    min_values = features.min(dim=0).values
    max_values = features.max(dim=0).values
    denom = torch.where(max_values > min_values, max_values - min_values, torch.ones_like(max_values))
    return (features - min_values) / denom


class MORAL(nn.Module):
    """Main MORAL model.

    The model keeps one encoder/decoder pair for each of the three sensitive
    attribute combinations (00, 01, 11).  During training we sample balanced
    batches within each group which ensures that the model is not dominated by
    the majority group.
    """

    def __init__(
        self,
        adj: Tensor,
        features: Tensor,
        labels: Tensor,
        idx_train: Tensor,
        idx_val: Tensor,
        idx_test: Tensor,
        sens: Tensor,
        sens_idx: Tensor,
        edge_splits: Dict[str, Dict[str, Tensor]],
        dataset_name: str,
        num_hidden: int = 128,
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        encoder: str = "gcn",
        decoder: str = "gae",
        batch_size: int = 1024,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.criterion = nn.BCEWithLogitsLoss()

        self.features = normalize_features(features.float()).to(self.device)
        self.edge_index = adj.coalesce().indices().to(self.device)
        self.labels = labels.to(self.device)
        self.sens = sens.to(self.device)
        self.sens_idx = sens_idx
        self.sens_cpu = sens.cpu()
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test

        self.encoders = nn.ModuleList([
            build_encoder(encoder, self.features.size(1), num_hidden).to(self.device)
            for _ in range(NUM_GROUPS)
        ])
        self.predictors = nn.ModuleList([
            build_predictor(decoder, num_hidden).to(self.device)
            for _ in range(NUM_GROUPS)
        ])

        self.optimizers = [
            torch.optim.Adam(
                list(self.encoders[group].parameters()) + list(self.predictors[group].parameters()),
                lr=lr,
                weight_decay=weight_decay,
            )
            for group in range(NUM_GROUPS)
        ]

        self.train_loaders = self._build_group_loaders(edge_splits.get("train"), shuffle=True)
        self.valid_loaders = self._build_group_loaders(edge_splits.get("valid"), shuffle=False)

        test_edges, test_labels = self._prepare_edges(edge_splits.get("test"))
        self.test_edges = test_edges
        self.test_labels = test_labels

        self.original_sens_dist = self._compute_original_sens_dist()
        self.best_state: Optional[Dict[str, Tensor]] = None

    def _prepare_edges(self, split: Optional[Dict[str, Tensor]]) -> Tuple[Tensor, Tensor]:
        if split is None:
            return torch.empty(0, 2, dtype=torch.long), torch.empty(0)

        pos_edges = split["edge"].long()
        neg_edges = split["edge_neg"].long()
        edges = torch.cat([pos_edges, neg_edges], dim=0)
        labels = torch.cat(
            [torch.ones(pos_edges.size(0)), torch.zeros(neg_edges.size(0))], dim=0
        )
        return edges, labels

    def _build_group_loaders(self, split: Optional[Dict[str, Tensor]], shuffle: bool) -> List[Optional[DataLoader]]:
        if split is None:
            return [None] * NUM_GROUPS

        edges, labels = self._prepare_edges(split)
        sens_groups = self.sens_cpu[edges].sum(dim=1)
        loaders: List[Optional[DataLoader]] = []
        for group in range(NUM_GROUPS):
            mask = sens_groups == group
            if mask.sum() == 0:
                loaders.append(None)
                continue

            dataset = EdgeLabelDataset(edges[mask], labels[mask])
            batch = len(dataset) if self.batch_size <= 0 else self.batch_size
            loader = DataLoader(dataset, batch_size=batch, shuffle=shuffle, drop_last=False)
            loaders.append(loader)
        return loaders

    def _compute_original_sens_dist(self) -> Tensor:
        src, dst = self.edge_index
        edge_groups = (self.sens[src] + self.sens[dst]).long()
        counts = torch.bincount(edge_groups, minlength=NUM_GROUPS).float()
        return counts / counts.sum().clamp(min=1.0)

    def forward(self, group: int) -> Tensor:
        return self.encoders[group](self.features, self.edge_index)

    def _train_epoch(self) -> float:
        total_loss = 0.0
        total_batches = 0
        for group, loader in enumerate(self.train_loaders):
            if loader is None:
                continue

            encoder = self.encoders[group]
            predictor = self.predictors[group]
            optimizer = self.optimizers[group]

            encoder.train()
            predictor.train()

            for edges, labels in loader:
                optimizer.zero_grad()
                edges = edges.t().to(self.device)
                labels = labels.to(self.device)

                embeddings = encoder(self.features, self.edge_index)
                logits = predictor(embeddings, edges)
                loss = self.criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                total_batches += 1
        return total_loss / max(total_batches, 1)

    @torch.no_grad()
    def _evaluate(self, loaders: Sequence[Optional[DataLoader]]) -> Optional[float]:
        total_loss = 0.0
        total_batches = 0
        for group, loader in enumerate(loaders):
            if loader is None:
                continue

            self.encoders[group].eval()
            self.predictors[group].eval()

            embeddings = self.encoders[group](self.features, self.edge_index)
            for edges, labels in loader:
                edges = edges.t().to(self.device)
                labels = labels.to(self.device)
                logits = self.predictors[group](embeddings, edges)
                loss = self.criterion(logits, labels)
                total_loss += float(loss.item())
                total_batches += 1

        if total_batches == 0:
            return None
        return total_loss / total_batches

    def fit(self, epochs: int = 300) -> None:
        """Train the model and store the best validation checkpoint."""

        best_val = float("inf")
        best_state: Optional[Dict[str, Tensor]] = None

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch()
            val_loss = self._evaluate(self.valid_loaders)

            if val_loss is not None and val_loss < best_val:
                best_val = val_loss
                best_state = deepcopy(self.state_dict())

            if epoch % 10 == 0 or epoch == 1:
                if val_loss is None:
                    logger.info("Epoch %d | train=%.4f", epoch, train_loss)
                else:
                    logger.info("Epoch %d | train=%.4f | valid=%.4f", epoch, train_loss, val_loss)

        if best_state is not None:
            self.load_state_dict(best_state)
            self.best_state = best_state

    @torch.no_grad()
    def predict(self) -> Tensor:
        """Predict logits for the test edge split."""

        if self.test_edges.numel() == 0:
            raise RuntimeError("Test split is empty. Did you load the dataset correctly?")

        outputs = torch.zeros(self.test_edges.size(0), device=self.device)
        sens_groups = self.sens_cpu[self.test_edges].sum(dim=1)

        for group in range(NUM_GROUPS):
            mask = sens_groups == group
            if mask.sum() == 0:
                continue

            edges = self.test_edges[mask].t().to(self.device)
            self.encoders[group].eval()
            self.predictors[group].eval()
            embeddings = self.encoders[group](self.features, self.edge_index)
            logits = self.predictors[group](embeddings, edges)
            outputs[mask] = logits

        return outputs

    @staticmethod
    def fair_metric(pred: Tensor, labels: Tensor, sens: Tensor) -> Tuple[float, float]:
        """Return demographic parity and equality of opportunity gaps."""

        idx_s0 = sens == 0
        idx_s1 = sens == 1
        idx_s0_y1 = torch.logical_and(idx_s0, labels == 1)
        idx_s1_y1 = torch.logical_and(idx_s1, labels == 1)

        # Avoid division by zero
        parity = torch.tensor(0.0)
        equality = torch.tensor(0.0)
        if idx_s0.sum() > 0 and idx_s1.sum() > 0:
            parity = (pred[idx_s0].mean() - pred[idx_s1].mean()).abs()
        if idx_s0_y1.sum() > 0 and idx_s1_y1.sum() > 0:
            equality = (pred[idx_s0_y1].mean() - pred[idx_s1_y1].mean()).abs()

        return float(parity.item()), float(equality.item())

