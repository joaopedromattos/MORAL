# MORAL - Breaking the Dyadic Barrier: Rethinking Fairness in Link Prediction Beyond Demographic Parity

MORAL (Multi-Output Ranking Aggregation for Link Fairness) is the model that
supports our accepted AAAI paper: Breaking the Dyadic Barrier: Rethinking Fairness in Link Prediction Beyond Demographic Parity. 

The repository keeps the training loop intentionally lightweight so that other researchers can import
the model, run experiments on the published benchmarks, and adapt the approach
to new datasets with minimal friction.

## Repository layout

| File | Description |
| --- | --- |
| `main.py` | Command line entry point that trains MORAL and exports the ranking artefacts used in the paper. |
| `moral.py` | Implementation of the MORAL model with per-sensitive-group encoders/decoders, validation checkpointing, and inference utilities. |
| `utils.py` | Dataset loader that converts the published edge splits into tensors and normalises sparse adjacency matrices. |
| `datasets.py` | Dataset definitions and download helpers for the benchmarks evaluated in the paper. |
| `moral.sh` | Example shell script that reproduces the paper runs across all datasets. |

## Installation

1. Create a Python environment (Python 3.9+ recommended).
2. Install PyTorch with CUDA support that matches your hardware by following the
   [official instructions](https://pytorch.org/).
3. Install the remaining dependencies:

   ```bash
   pip install torch-geometric loguru gdown requests pandas scipy networkx
   ```

   The dataset loaders will download raw files on demand into `./dataset/`.

## Getting the edge splits

The training script expects pre-computed train/validation/test edge splits saved
as `torch.save((data, splits), path)` files under `data/splits/`. Each file is
named after the dataset identifier, for example:

```
data/splits/facebook.pt
```

The `data` object stores the original PyG `Data` instance, and `splits` contains
three dictionaries (`"train"`, `"valid"`, `"test"`) with positive edges under
`"edge"` and negative samples under `"edge_neg"`.

If you wish to generate the splits yourself, follow the format above so that
`utils.get_dataset` can locate them.

## Quick start

Train MORAL on the Facebook dataset and export predictions:

```bash
python main.py --dataset facebook --model gae --device cuda:0 --epochs 300
```

The script performs three runs by default (controlled via `--runs`). For each
run it saves:

* `three_classifiers_<DATASET>_<FAIR_MODEL>_<MODEL>_<RUN>.pt` – raw logits for
  every edge in the test split.
* `three_classifiers_<...>_final_ranking.pt` – scores and labels after grouping
  edges according to the greedy KL routine described in the paper.

To reproduce the collection of experiments from the paper, adjust the CUDA
devices in `moral.sh` and execute:

```bash
bash moral.sh
```

## Configuration reference

`main.py` exposes a handful of arguments that cover the knobs discussed in the
paper:

* `--dataset`: dataset identifier (`facebook`, `german`, `nba`, `pokec_n`,
  `pokec_z`, `credit`, `gplus`).
* `--model`: base encoder/decoder setup. `gae` uses dot-product decoders, while
  `ncn` employs an MLP decoder.
* `--hidden_dim`: latent dimensionality of each per-group encoder.
* `--batch_size`: minibatch size per sensitive group (set `-1` to process entire
  splits at once).
* `--lr`, `--weight_decay`, `--epochs`: standard optimiser hyperparameters.
* `--device`: PyTorch device string (e.g. `cpu`, `cuda:0`).
* `--runs`, `--seed`: reproducibility controls.

Legacy flags such as `--ranking_loss` and `--baseline` remain for backwards
compatibility but do not alter the current training behaviour.

## Adapting MORAL to new datasets

MORAL assumes that each edge is associated with a binary sensitive attribute for
both incident nodes. To plug in a new dataset:

1. Implement a dataset wrapper in `datasets.py` (or a separate module) that
   exposes the following methods returning PyTorch tensors: `features()`,
   `labels()`, `idx_train()`, `idx_val()`, `idx_test()`, `sens()` (binary
   sensitive attribute per node), and `sens_idx()` (column index of the
   sensitive attribute in the raw features, if applicable).
2. Register the dataset class in the `dataset_map` inside `utils.get_dataset`.
3. Provide edge splits following the structure described in
   [Getting the edge splits](#getting-the-edge-splits).

At this point you can run `main.py --dataset <your_id>` and MORAL will download
any raw files required by your dataset class, normalise the features, and train
per-group encoders automatically.

## Extending the model

`moral.py` is designed to be self-contained:

* Encoders are built by `build_encoder` and can be extended with new GNN
  backbones (e.g. Graph Attention Networks) by adding an option to the helper.
* Decoders can be customised via `build_predictor`. Implementing a new decoder
  only requires subclassing `LinkPredictor` or providing a different module that
  consumes node embeddings and edge indices.
* The training loop maintains separate optimiser states per sensitive group,
  making it straightforward to plug in alternative sampling or loss functions.

If you prefer to import the model from another project, simply use:

```python
from moral import MORAL
```

and construct the module with the tensors provided by your own data pipeline.

## Citation

If you use this repository in your research, please cite the MORAL paper once it
is available. A BibTeX entry will be added after publication.
