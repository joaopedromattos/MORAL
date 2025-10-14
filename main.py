from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch_geometric.data import Data

from moral import MORAL
from utils import get_dataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_array_greedy_dkl(n: int, distribution: np.ndarray) -> torch.Tensor:
    """Generate indices that approximate a target distribution for every prefix."""

    actual_counts = np.zeros_like(distribution)
    result = []

    for i in range(n):
        if i == 0:
            choice = int(np.argmax(distribution))
        else:
            desired = distribution * i
            deficit = desired - actual_counts
            choice = int(np.argmax(deficit))
        result.append(choice)
        actual_counts[choice] += 1

    return torch.tensor(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MORAL on a selected dataset.")
    parser.add_argument("--dataset", type=str, default="facebook", help="Dataset identifier.")
    parser.add_argument("--model", type=str, default="gae", choices={"gae", "ncn"}, help="Base encoder/decoder setup.")
    parser.add_argument("--fair_model", type=str, default="moral", help="Name of the fairness method (for logging only).")
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch device string.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Training batch size per sensitive group.")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs.")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimensionality of the encoders.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay used by Adam.")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs with different seeds.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed used for reproducibility.")
    parser.add_argument(
        "--splits_dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory containing pre-computed edge splits (train/valid/test).",
    )
    parser.add_argument(
        "--ranking_loss",
        action="store_true",
        help="Deprecated option kept for compatibility with older scripts.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Deprecated option kept for compatibility with older scripts.",
    )
    return parser.parse_args()


def resolve_model_config(model_name: str) -> Dict[str, str]:
    mapping = {
        "gae": {"encoder": "gcn", "decoder": "gae"},
        "ncn": {"encoder": "gcn", "decoder": "mlp"},
    }
    try:
        return mapping[model_name]
    except KeyError as exc:  # pragma: no cover - safeguarded by argparse choices
        raise ValueError(f"Unsupported model '{model_name}'.") from exc


def run_single(args: argparse.Namespace, run: int) -> None:
    seed = args.seed + run
    seed_everything(seed)
    logger.info("Run %d/%d — seed=%d", run + 1, args.runs, seed)

    model_cfg = resolve_model_config(args.model)
    adj, features, idx_train, idx_val, idx_test, labels, sens, sens_idx, data, splits = get_dataset(
        args.dataset, args.splits_dir
    )

    labels = labels.cpu()
    sens = sens.cpu()

    model = MORAL(
        adj=adj,
        features=features,
        labels=labels,
        idx_train=idx_train.long(),
        idx_val=idx_val.long(),
        idx_test=idx_test.long(),
        sens=sens,
        sens_idx=sens_idx,
        edge_splits=splits,
        dataset_name=args.dataset,
        num_hidden=args.hidden_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        encoder=model_cfg["encoder"],
        decoder=model_cfg["decoder"],
        batch_size=args.batch_size,
        device=args.device,
    )

    logger.info("Training model…")
    model.fit(epochs=args.epochs)

    logger.info("Running inference on the test split…")
    outputs = model.predict().cpu()

    run_suffix = f"{args.dataset}_{args.fair_model.upper()}_{args.model.upper()}_{run}"
    torch.save(outputs, f"three_classifiers_{run_suffix}.pt")

    if not isinstance(data, Data):
        logger.warning("Unexpected data type returned by get_dataset; skipping ranking export.")
        return

    pi = (
        F.one_hot(data.y[data.edge_index].sum(0).long(), num_classes=3).float().sum(0)
        / data.edge_index.shape[1]
    ).cpu().numpy()

    K = 1000
    final_output = torch.zeros(size=(K,))
    final_labels = torch.zeros(size=(K,))
    output_positions = generate_array_greedy_dkl(K, pi)

    test_split = splits["test"]
    test_edges = torch.cat([test_split["edge"], test_split["edge_neg"]], dim=0)
    test_labels = torch.cat(
        [torch.ones(test_split["edge"].size(0)), torch.zeros(test_split["edge_neg"].size(0))],
        dim=0,
    )
    edge_sens_groups = sens[test_edges].sum(dim=1)

    for sens_value in range(3):
        mask = output_positions == sens_value
        if mask.sum() == 0:
            continue

        group_mask = edge_sens_groups == sens_value
        if group_mask.sum() == 0:
            continue

        group_outputs, indices = outputs[group_mask].sort(descending=True)
        group_labels = test_labels[group_mask][indices]
        final_output[mask] = group_outputs[: mask.sum()]
        final_labels[mask] = group_labels[: mask.sum()]

    torch.save((final_output, final_labels), f"three_classifiers_{run_suffix}_final_ranking.pt")
    logger.success("Finished run %d/%d", run + 1, args.runs)


def main() -> None:
    args = parse_args()
    logger.info("Processing dataset '%s' with model '%s'.", args.dataset, args.model)
    for run in range(args.runs):
        run_single(args, run)


if __name__ == "__main__":
    main()
