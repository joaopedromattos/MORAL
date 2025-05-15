
import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    GINConv,
    SAGEConv,
    DeepGraphInfomax,
    JumpingKnowledge,
)
import os
from loguru import logger
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score
from torch.nn.utils import spectral_norm
from torch_geometric.utils import dropout_adj, convert
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import time
import argparse
import numpy as np
import scipy.sparse as sp
from torch_geometric.typing import SparseTensor

from torch_geometric.data import Data
from torch.utils.data import ConcatDataset, DataLoader, Dataset


from NCNC import CN0LinkPredictor, CN0LinkPredictorModded, CNLinkPredictor
from SEAL import SEALDataset, GCN_SEAL

## Scipy
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
import wandb

import numpy as np

def generate_array_greedy(n, p):
    """
    Greedy probabilistic approach to generate an array that maintains the 
    target distribution (p0, p1, p2) for any prefix length k.
    
    Args:
        n (int): Length of the array.
        p (tuple): Target proportions (p0, p1, p2).

    Returns:
        list: Generated array of length n with values in {0,1,2}.
    """
    target_counts = np.array([p_i * n for p_i in p])  # Expected counts for 0,1,2
    actual_counts = np.zeros(3)  # Current counts of 0s, 1s, 2s
    arr = []

    for i in range(n):
        # Compute probabilities to minimize deviation from target counts
        remaining = n - i
        probs = (target_counts - actual_counts) / remaining  # Adjusted probabilities
        
        # Avoid numerical errors
        probs = np.clip(probs, 0, 1)
        probs /= probs.sum()  # Normalize to sum to 1

        # Sample from adjusted probabilities
        choice = np.random.choice([0, 1, 2], p=probs)
        arr.append(choice)
        actual_counts[choice] += 1

    return arr
    

# Create a PyTorch Dataset for edge batches
class EdgeBatchDataset(Dataset):
    def __init__(self, edges, sample_value=1):
        self.edges = edges
        self.sample_value = sample_value
        

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, idx):
        edge_sampled = self.edges[idx]
        return (edge_sampled, self.sample_value)


'''Old and unoptimized version
def ranked_kl_divergence(edge_labels, scores, original_edge_dist, sens_labels, top_k=-1, noise=0.0):
    
    # Sort edge_labels based on scores in descending order
    # sorted_edge_labels = edge_labels[scores.argsort(descending=True), :]
    sorted_edge_labels = edge_labels

    if top_k == -1:
        top_k = len(sorted_edge_labels)

    # Precompute cumulative sum of one-hot encoded sorted edge labels
    cumulative_sum = sorted_edge_labels.cumsum(dim=0).float()
    
    # print(sorted_edge_labels, original_edge_dist)
    
    # import code
    # code.interact(local={**locals(), **globals()})

    # Precompute the normalization factors
    log_indices = torch.log2(torch.arange(1, top_k + 1).float() + 1).to(edge_labels.device)
    z_vals = 1 / log_indices

    # Compute the KL divergence for all positions at once
    # TODO FIND A WAY OF TRANSFORMING THE SOFTLABELS INTO LOG PROBABILITIES.
    kl_divs = torch.stack([F.kl_div(torch.log((cumulative_sum[i] / (i + 1)) + noise), original_edge_dist, reduction='batchmean') for i in range(top_k)])

    # Apply weighting using z_vals
    weighted_kl = kl_divs * z_vals

    # Sum the weighted KL divergences and normalize by Z
    Z = z_vals.sum()
    
    return weighted_kl.sum() / Z, torch.norm(sorted_edge_labels.float(), p=1, dim=1).mean()
'''


def ranked_kl_divergence(edge_labels, scores, original_edge_dist, sens_labels, top_k=-1, noise=1e-10):
    # scores = torch.tensor(scores)
    sorted_edge_labels = edge_labels
    
    if top_k == -1 or top_k > len(sorted_edge_labels):
        top_k = len(sorted_edge_labels)
    else:
        sorted_edge_labels = sorted_edge_labels[:top_k, :]
        
    cumulative_sum = sorted_edge_labels.cumsum(dim=0).float()
    log_indices = torch.log2(torch.arange(1, top_k + 1).float() + 1).to(edge_labels.device)
    z_vals = 1 / log_indices
    kl_divs = F.kl_div(torch.log((cumulative_sum / torch.arange(1, top_k + 1, device=cumulative_sum.device)[:, None]) + noise), original_edge_dist, reduction='none').sum(1) / cumulative_sum.shape[1]
    weighted_kl = kl_divs * z_vals
    Z = z_vals.sum()
    return weighted_kl.sum() / Z, torch.norm(sorted_edge_labels.float(), p=1, dim=1).mean()


def matching(alpha):
    # Negate the probability matrix to serve as cost matrix. This function
    # yields two lists, the row and colum indices for all entries in the
    # permutation matrix we should set to 1.
    row, col = linear_sum_assignment(-alpha)

    # Create the permutation matrix.
    permutation_matrix = coo_matrix((np.ones_like(row), (row, col))).toarray()
    return torch.from_numpy(permutation_matrix)


def log_sinkhorn(log_alpha, n_iter):
    """Performs incomplete Sinkhorn normalization to log_alpha.
    By a theorem by Sinkhorn and Knopp [1], a sufficiently well-behaved  matrix
    with positive entries can be turned into a doubly-stochastic matrix
    (i.e. its rows and columns add up to one) via the successive row and column
    normalization.

    [1] Sinkhorn, Richard and Knopp, Paul.
    Concerning nonnegative matrices and doubly stochastic
    matrices. Pacific Journal of Mathematics, 1967
    Args:
      log_alpha: 2D tensor (a matrix of shape [N, N])
        or 3D tensor (a batch of matrices of shape = [batch_size, N, N])
      n_iters: number of sinkhorn iterations (in practice, as little as 20
        iterations are needed to achieve decent convergence for N~100)
    Returns:
      A 3D tensor of close-to-doubly-stochastic matrices (2D tensors are
        converted to 3D tensors with batch_size equals to 1)
    """
    for _ in range(n_iter):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -2, keepdim=True)
    return log_alpha.exp()


def sample_gumbel(shape, device='cpu', eps=1e-20):
    """Samples arbitrary-shaped standard gumbel variables.
    Args:
      shape: list of integers
      eps: float, for numerical stability
    Returns:
      A sample of standard Gumbel random variables
    """
    u = torch.rand(shape, device=device)
    return -torch.log(-torch.log(u + eps) + eps)

def gumbel_sinkhorn(log_alpha, tau, n_iter, noise_factor=0.0):
    """ Sample a permutation matrix from the Gumbel-Sinkhorn distribution
    with parameters given by log_alpha and temperature tau.

    Args:
      log_alpha: Logarithm of assignment probabilities. In our case this is
        of dimensionality [num_pieces, num_pieces].
      tau: Temperature parameter, the lower the value for tau the more closely
        we follow a categorical sampling.
      n_iter: Number of iterations to run the gumbel_sinhorn operator.
        **This should be chosen carefully, in opposite way to `tau` to ensure
        numerical stabiity.**
      noise_factor: Enables scalling of the noise sampled. When noise_factor=0.0,
        outputs a sinkhorn operator, instead of a gumbel_sinkhorn.
    """
    # Sample Gumbel noise.
    gumbel_noise = sample_gumbel(log_alpha.shape, device=log_alpha.device) * noise_factor

    # Apply the Sinkhorn operator!
    sampled_perm_mat = log_sinkhorn((log_alpha + gumbel_noise)/tau, n_iter)
    
    return sampled_perm_mat


class Classifier(nn.Module):
    def __init__(self, ft_in, nb_classes, base_model="standard"):
        super(Classifier, self).__init__()
        
        self.base_model = base_model

        # Classifier projector
        if base_model == 'classifier':
            self.predictor = spectral_norm(nn.Linear(ft_in, nb_classes))
        elif base_model == 'gae':
            self.predictor = CN0LinkPredictor(in_channels=ft_in, hidden_channels=ft_in, out_channels=nb_classes, num_layers=2, dropout=0.5)
        elif base_model == 'ncn':
            self.predictor = CNLinkPredictor(in_channels=ft_in, hidden_channels=ft_in, out_channels=nb_classes, num_layers=2, dropout=0.5)
        elif base_model == 'seal':
            self.predictor = nn.Linear(ft_in, 2)

    def forward(self, x, adj, tar_ei):
        ret = None
        if self.base_model == 'standard':
            ret = self.predictor(x)
        elif self.base_model == 'gae':
            ret = self.predictor.multidomainforward(x, adj, tar_ei)
        elif self.base_model == 'ncn':
            adj = SparseTensor.from_edge_index(adj, sparse_sizes=(x.size(0), x.size(0)))
            ret = self.predictor.multidomainforward(x, adj, tar_ei)
        elif self.base_model == 'seal':
            ret = self.predictor(x)
            
        return ret


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(GCN, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        return x


class GIN(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(GIN, self).__init__()

        self.mlp1 = nn.Sequential(
            spectral_norm(nn.Linear(nfeat, nhid)),
            nn.ReLU(),
            nn.BatchNorm1d(nhid),
            spectral_norm(nn.Linear(nhid, nhid)),
        )
        self.conv1 = GINConv(self.mlp1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x


class JK(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(JK, self).__init__()
        self.conv1 = spectral_norm(GCNConv(nfeat, nhid))
        self.convx = spectral_norm(GCNConv(nhid, nhid))
        self.jk = JumpingKnowledge(mode="max")
        self.transition = nn.Sequential(
            nn.ReLU(),
        )

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        xs = []
        x = self.conv1(x, edge_index)
        x = self.transition(x)
        xs.append(x)
        for _ in range(1):
            x = self.convx(x, edge_index)
            x = self.transition(x)
            xs.append(x)
        x = self.jk(xs)
        return x


class SAGE(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(SAGE, self).__init__()

        # Implemented spectral_norm in the sage main file
        # ~/anaconda3/envs/PYTORCH/lib/python3.7/site-packages/torch_geometric/nn/conv/sage_conv.py
        self.conv1 = SAGEConv(nfeat, nhid, normalize=True)
        self.conv1.aggr = "mean"
        self.transition = nn.Sequential(
            nn.ReLU(), nn.BatchNorm1d(nhid), nn.Dropout(p=dropout)
        )
        self.conv2 = SAGEConv(nhid, nhid, normalize=True)
        self.conv2.aggr = "mean"

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.transition(x)
        x = self.conv2(x, edge_index)
        return x


class Encoder_DGI(nn.Module):
    def __init__(self, nfeat, nhid):
        super(Encoder_DGI, self).__init__()
        self.hidden_ch = nhid
        self.conv = spectral_norm(GCNConv(nfeat, self.hidden_ch))
        self.activation = nn.PReLU()

    def corruption(self, x, edge_index):
        # corrupted features are obtained by row-wise shuffling of the original features
        # corrupted graph consists of the same nodes but located in different places
        return x[torch.randperm(x.size(0))], edge_index

    def summary(self, z, *args, **kwargs):
        return torch.sigmoid(z.mean(dim=0))

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.activation(x)
        return x


class GraphInfoMax(nn.Module):
    def __init__(self, enc_dgi):
        super(GraphInfoMax, self).__init__()
        self.dgi_model = DeepGraphInfomax(
            enc_dgi.hidden_ch, enc_dgi, enc_dgi.summary, enc_dgi.corruption
        )

    def forward(self, x, edge_index):
        pos_z, neg_z, summary = self.dgi_model(x, edge_index)
        return pos_z


class Encoder(torch.nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, max_z:int, base_model="gcn", k: int = 2
    ):
        super(Encoder, self).__init__()
        self.base_model = base_model
        if self.base_model == "gcn":
            self.conv = GCN(in_channels, out_channels)
        elif self.base_model == "gin":
            self.conv = GIN(in_channels, out_channels)
        elif self.base_model == "sage":
            self.conv = SAGE(in_channels, out_channels)
        elif self.base_model == "infomax":
            enc_dgi = Encoder_DGI(nfeat=in_channels, nhid=out_channels)
            self.conv = GraphInfoMax(enc_dgi=enc_dgi)
        elif self.base_model == "jk":
            self.conv = JK(in_channels, out_channels)
        elif self.base_model == "seal":
            self.conv = GCN_SEAL(num_features=in_channels, hidden_channels=out_channels, num_layers=3, max_z=max_z)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, z=None, batch=None, edge_weight=None, node_id=None) -> torch.Tensor:
        if self.base_model == "seal":
            # print("Inside encoder")
            # import code
            # code.interact(local={**locals(), **globals()})
            return self.conv(z=z, edge_index=edge_index, batch=batch, x=x, edge_weight=edge_weight, node_id=node_id)
        else:
            return self.conv(x, edge_index)
        


class MORAL(torch.nn.Module):
    def __init__(
        self,
        adj,
        features,
        labels,
        idx_train,
        idx_val,
        idx_test,
        sens,
        sens_idx,
        edge_splits,
        dataset_name,
        num_hidden=16,
        num_proj_hidden=16,
        lr=0.0001,
        weight_decay=1e-5,
        drop_edge_rate_1=0.1,
        drop_edge_rate_2=0.1,
        drop_feature_rate_1=0.1,
        drop_feature_rate_2=0.1,
        encoder="gcn",
        decoder='standard',
        sim_coeff=0.5,
        nclass=1,
        batch_size = -1,
        device="cuda",
    ):
        super(MORAL, self).__init__()

        self.device = device
        self.batch_size = batch_size
        
        # Sinkhorn parameters
        self.tau = 0.5 # Temperature parameter.
        self.n_sink_iter = 20 # Number of iterations of Sinkhorn operator.
        
        # self.edge_index = convert.from_scipy_sparse_matrix(sp.coo_matrix(adj.to_dense().numpy()))[0]
        self.edge_index = adj.coalesce().indices()
        self.decoder_name = decoder
        self.encoder_name = encoder
        self.dataset_name = dataset_name
        
        self.sim_coeff = sim_coeff
        # self.encoder = encoder
        self.labels = labels

        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.sens = sens
        self.sens_idx = sens_idx
        self.lr = lr
        
        self.encoder = [Encoder(
            in_channels=features.shape[1], out_channels=num_hidden, base_model=encoder, max_z=features.shape[0]
        ).to(device) for i in range(3)]

        # Classifier
        self.c = [Classifier(ft_in=num_hidden, nb_classes=nclass, base_model=decoder).to(device) for i in range(3)]

        
        features = (features - features.min(dim=0).values)/(features.max(dim=0).values - features.min(dim=0).values)
        self.features = features.to(device)
        self.edge_index = self.edge_index.to(device)
        self.labels = self.labels.to(device)
        self.sens = self.sens.to(device)
        
        # Original distribution of each type of edge in the original graph
        sens_counts = F.one_hot(self.sens[self.edge_index].sum(0).long(), num_classes=3).sum(0)
        self.original_sens_dist = sens_counts / sens_counts.sum()
        
        logger.info(f"Creating batches... batch size: {batch_size}")
        
        # Adds all edges and labels as attributes
        for edge_set in edge_splits.keys():
            setattr(self, f'{edge_set}_edge_index', torch.cat([edge_splits[edge_set]['edge'].t(), edge_splits[edge_set]['edge_neg'].t()], dim=-1).to(self.device))
            setattr(self, f'{edge_set}_edge_labels', F.one_hot(torch.cat([torch.ones(edge_splits[edge_set]['edge'].size(0)), 
                                                                            torch.zeros(edge_splits[edge_set]['edge_neg'].size(0))]).long(), 
                                                                        num_classes=2).float().to(self.device))
            
            if edge_set != 'test':
                
                for cur_sens_value in range(3):
                    pos_mask = (self.sens[edge_splits[edge_set]['edge']].sum(1) == cur_sens_value).to(edge_splits[edge_set]['edge'].device)
                    neg_mask = (self.sens[edge_splits[edge_set]['edge_neg']].sum(1) == cur_sens_value).to(edge_splits[edge_set]['edge'].device)
            
                    pos_edge_dataset = EdgeBatchDataset(edge_splits[edge_set]['edge'][pos_mask], sample_value=1)
                    neg_edge_dataset = EdgeBatchDataset(edge_splits[edge_set]['edge_neg'][neg_mask], sample_value=0)
                    
                    edge_loader = DataLoader(ConcatDataset([pos_edge_dataset, neg_edge_dataset]), 
                                            batch_size= min(len(pos_edge_dataset), len(neg_edge_dataset)) if self.batch_size < 0 else self.batch_size, 
                                            shuffle=True if edge_set != 'test' else False, 
                                            num_workers=8, 
                                            pin_memory=True,)
            
                    setattr(self, f'{edge_set}_loader_{cur_sens_value}', edge_loader)
            
            else:
                pos_edge_dataset = EdgeBatchDataset(edge_splits[edge_set]['edge'], sample_value=1)
                neg_edge_dataset = EdgeBatchDataset(edge_splits[edge_set]['edge_neg'], sample_value=0)
        
    
                edge_loader = DataLoader(ConcatDataset([pos_edge_dataset, neg_edge_dataset]), 
                                            batch_size= min(len(pos_edge_dataset), len(neg_edge_dataset)) if self.batch_size < 0 else self.batch_size, 
                                            shuffle=True if edge_set != 'test' else False, 
                                            num_workers=8, 
                                            pin_memory=True,)
                
                setattr(self, f'{edge_set}_loader', edge_loader)

                            
        if encoder == 'seal' and decoder == 'seal':
            data = Data(x=self.features, edge_index=self.edge_index, y=torch.cat([torch.ones(edge_splits['train']['edge'].size(0)), 
                                                                          torch.zeros(edge_splits['train']['edge_neg'].size(0))]).long())
            self.seal_dataset = SEALDataset(root='data', data=data.cpu(), split_edge=edge_splits, num_hops=2, split='train', dataset_name=dataset_name)
            
            data_valid = Data(x=self.features, edge_index=self.edge_index, y=torch.cat([torch.ones(edge_splits['valid']['edge'].size(0)), 
                                                                          torch.zeros(edge_splits['valid']['edge_neg'].size(0))]).long())
            self.seal_dataset_valid = SEALDataset(root='data', data=data_valid.cpu(), split_edge=edge_splits, num_hops=2, split='valid', dataset_name=dataset_name)
            
            data_test = Data(x=self.features, edge_index=self.edge_index, y=torch.cat([torch.ones(edge_splits['test']['edge'].size(0)), 
                                                                          torch.zeros(edge_splits['test']['edge_neg'].size(0))]).long())
            self.seal_dataset_test = SEALDataset(root='data', data=data_test.cpu(), split_edge=edge_splits, num_hops=2, split='test', dataset_name=dataset_name)
            
            
        self.optimizer = [optim.Adam(list(self.c[i].parameters()) + list(self.encoder[i].parameters()), lr=lr, weight_decay=weight_decay) for i in range(3)]
        self = self.to(device)

        
    def _get_k_value(self, batch_dist):
        idx = batch_dist.argmin()
        return int((batch_dist[idx] * self.batch_size / self.original_sens_dist[idx]).ceil().item())

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, z=None, batch=None, edge_weight=None, node_id=None, sens_class:int=0) -> torch.Tensor:
        return self.encoder[sens_class](x, edge_index, z, batch, edge_weight, node_id)

    def classifier(self, x, adj=None, tar_ei=None, sens_class:int=0):
        return self.c[sens_class](x, adj, tar_ei)

    def normalize(self, x):
        val = torch.norm(x, p=2, dim=1).detach()
        x = x.div(val.unsqueeze(dim=1).expand_as(x))
        return x

    # TODO: CHECK FOR DELETION LATER
    def D(self, x1, x2):  # negative cosine similarity
        return -F.cosine_similarity(x1, x2.detach(), dim=-1).mean()

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor, e_1, e_2, idx):

        # projector
        p1 = self.projection(z1)
        p2 = self.projection(z2)

        # predictor
        h1 = self.prediction(p1)
        h2 = self.prediction(p2)

        # classifier
        c = self.classifier(z1)

        l1 = self.D(h1[idx], p2[idx]) / 2
        l2 = self.D(h2[idx], p1[idx]) / 2
        l3 = F.cross_entropy(c[idx], z3[idx].squeeze().long().detach())

        return self.sim_coeff * (l1 + l2), l3
    
    
    def sinkhorn_operator(self, emb, edge_index, batch_edge_index, scores):
        log_alphas = self.g(emb)
        idx = scores.argsort(descending=True)
        
        # Log_alphas was created from embeddings, so we normalize it to maintain numerical stability
        log_alphas = (log_alphas - log_alphas.min()) / (log_alphas.max() - log_alphas.min())
        
        # During training, we sample from the Gumbel-Sinkhorn distribution.
        if self.training:
            # permutation_matrices = gumbel_sinkhorn(F.log_softmax(log_alphas, dim=1), tau=self.tau, n_iter=self.n_sink_iter)
            permutation_matrices = gumbel_sinkhorn(log_alphas, tau=self.tau, n_iter=self.n_sink_iter)

        # During eval, we solve the linear assignment problem.
        else:
            permutation_matrices = torch.tensor(
                matching(log_alphas.cpu().detach().numpy())
            ).float().to(log_alphas.device)
            
        assert torch.allclose(permutation_matrices.sum(0), torch.tensor(1.0), atol=0.001), 'Columns do not sum to 1.0'
        assert torch.allclose(permutation_matrices.sum(1), torch.tensor(1.0), atol=0.01), 'Rows do not sum to 1.0'
            
            
        # TODO CHECK THIS... WHY IS THIS INTRODUCTING A VERY LARGE VALUE ON THE FINAL SOFT LABELS?
        # THIS IS THE 385 CASE
        soft_labels = permutation_matrices[idx, :] @ F.one_hot(self.sens[batch_edge_index].sum(0).long(), num_classes=3).float()[idx, :]
        
        assert torch.allclose(soft_labels.sum(), F.one_hot(self.sens[batch_edge_index].sum(0).long(), num_classes=3).float().sum(), atol=0.001), "Soft labels do not sum to original value."
        
        # import code
        # code.interact(local={**locals(), **globals()})

        return log_alphas, soft_labels
            
    def linear_eval(self, emb, labels, idx_train, idx_test):
        x = emb.detach()
        classifier = nn.Linear(in_features=x.shape[1], out_features=2, bias=True)
        classifier = classifier.to("cuda")
        optimizer = torch.optim.Adam(
            classifier.parameters(), lr=0.001, weight_decay=1e-4
        )
        for i in range(1000):
            optimizer.zero_grad()
            preds = classifier(x[idx_train])
            loss = F.cross_entropy(preds, labels[idx_train])
            loss.backward()
            optimizer.step()
            # if i%100==0:
            #     print(loss.item())
        classifier.eval()
        preds = classifier(x[idx_test]).argmax(dim=1)
        correct = (preds == labels[idx_test]).sum().item()
        return preds, correct / preds.shape[0]

    def ssf_validation_seal(self, x_1, edge_index_1, x_2, edge_index_2, y):
        batch_size = 1024
        valid_loader = DataLoader(self.seal_dataset_valid, batch_size=batch_size, shuffle=False, num_workers=0)
        sim_loss_accum = 0
        c_loss_accum = 0
        for _ in valid_loader:
            _ = _.to(self.device)
            edge_index_1 = dropout_adj(_.edge_index, p=self.drop_edge_rate_1)[0]
            edge_index_2 = dropout_adj(_.edge_index, p=self.drop_edge_rate_2)[0]
            x_1 = drop_feature(
                _.x,
                self.drop_feature_rate_1,
                self.sens_idx,
                sens_flag=False,
            )
            x_2 = drop_feature(
                _.x, self.drop_feature_rate_2, self.sens_idx
            )
            
            z1 = self.forward(x_1, edge_index_1, z=_.z, batch=_.batch, edge_weight=None, node_id=_.node_id)
            z2 = self.forward(x_2, edge_index_2, z=_.z, batch=_.batch, edge_weight=None, node_id=_.node_id)

            # projector
            p1 = self.projection(z1)
            p2 = self.projection(z2)

            # predictor
            h1 = self.prediction(p1)
            h2 = self.prediction(p2)

            l1 = self.D(h1, p2) / 2
            l2 = self.D(h2, p1) / 2
            sim_loss = self.sim_coeff * (l1 + l2)
            sim_loss_accum += sim_loss.item() / len(_)

            # classifier
            c = self.classifier(z1, edge_index_1, self.valid_edge_index)
            c2 = self.classifier(z2, edge_index_2, self.valid_edge_index)
            
            y_one_hot = F.one_hot(_.y, num_classes=2).float()

            # Binary Cross-Entropy
            l3 = (
                F.binary_cross_entropy_with_logits(
                    c, y_one_hot
                )
                / 2
            )
            l4 = (
                F.binary_cross_entropy_with_logits(
                    c2, y_one_hot
                )
                / 2
            )
            
            c_loss_accum += (1 - self.sim_coeff) * (l3.item() + l4.item()) / len(_)

        return sim_loss_accum, c_loss_accum
    
    
    def ssf_validation(self, x, edge_index, y):
        with torch.no_grad():
            total_loss = 0.0
            cl_loss = 0.0
            sim_loss = 0.0
            for sens_class in range(3):
                for batch_edge_index, batch_labels in tqdm(getattr(self, f'valid_loader_{sens_class}'), desc=f'Validation - {sens_class}'):
                    
                    # Skipping batches that do not match the pre-defined length.
                    if len(batch_edge_index) < self.batch_size:
                            continue
                    
                    batch_edge_index, batch_labels = batch_edge_index.t().to(self.device), batch_labels.to(self.device)
                    
                    z = self.forward(x, edge_index, sens_class=sens_class)
                        
                    c = self.classifier(z, edge_index, batch_edge_index)
                    
                    c = c.squeeze()
                    
                    ## For validation, we compute the actual NDKL with the labels.
                    sens_edge_labels = F.one_hot(self.sens[batch_edge_index].sum(0).long(), num_classes=3).long()[c.argsort(descending=True), :]
                    
                    ndkl_loss, sparsity_loss = ranked_kl_divergence(sens_edge_labels, c, self.original_sens_dist, F.one_hot(self.sens[batch_edge_index].sum(0).long(), num_classes=3).long()[c.argsort(descending=True), :], -1, noise=1e-20)
                    
                    # Binary Cross-Entropy
                    l = (F.binary_cross_entropy_with_logits(c.float(), batch_labels.float(),)/ 2)
                    
                    cl_loss += l.item()
                    
                    l = l + (self.sim_coeff * ndkl_loss)
                    
                    total_loss += l.item()
                    sim_loss += ndkl_loss.item()
                
        return total_loss, cl_loss, sim_loss
    

    def fit_GNN(self, epochs=300):
        best_loss = 100
        for epoch in range(epochs + 1):

            sim_loss = 0

            self.train()
            self.optimizer_2.zero_grad()
            edge_index_1 = self.edge_index
            x_1 = self.features

            # classifier
            z1 = self.forward(x_1, edge_index_1)
            c = self.classifier(z1)
        
            # Binary Cross-Entropy
            cl_loss = F.binary_cross_entropy_with_logits(
                c[self.idx_train],
                self.labels[self.idx_train].unsqueeze(1).float().to(self.device),
            )

            cl_loss.backward()
            self.optimizer_2.step()

            # Validation
            self.eval()
            z_val = self.forward(self.features, self.edge_index)
            c_val = self.classifier(z_val)
            val_loss = F.binary_cross_entropy_with_logits(
                c_val[self.idx_val],
                self.labels[self.idx_val].unsqueeze(1).float().to(self.device),
            )

            # if epoch % 100 == 0:
            #     logger.info(f"[Train] Epoch {epoch}: train_c_loss: {cl_loss:.4f} | val_c_loss: {val_loss:.4f}")

            if (val_loss) < best_loss:
                self.val_loss = val_loss.item()

                best_loss = val_loss
                if not os.path.exists("data"):
                    os.makedirs("data")
                torch.save(self.state_dict(), f"data/three_classifiers_weights_ssf_{self.dataset_name}_{self.encoder_name}_{self.decoder_name}.pt")
                
    def fit_seal(self, epochs=300):
        # Train model
        t_total = time.time()
        best_loss = 100
        best_acc = 0
        
        for epoch in tqdm(range(epochs + 1)):
            t = time.time()

            sim_loss = 0
            cl_loss = 0
            sim_loss_accum = 0
            train_loader = DataLoader(self.seal_dataset, batch_size=1024, shuffle=True, num_workers=0)
            for _ in tqdm(train_loader):
                _ = _.to(self.device)
                self.train()
                self.optimizer_1.zero_grad()
                self.optimizer_2.zero_grad()
                edge_index_1 = dropout_adj(_.edge_index, p=self.drop_edge_rate_1)[0]
                edge_index_2 = dropout_adj(_.edge_index, p=self.drop_edge_rate_2)[0]
                x_1 = drop_feature(
                    _.x,
                    self.drop_feature_rate_1,
                    self.sens_idx,
                    sens_flag=False,
                )
                x_2 = drop_feature(
                    _.x, self.drop_feature_rate_2, self.sens_idx
                )
                z1 = self.forward(x_1, edge_index_1, z=_.z, batch=_.batch, edge_weight=None, node_id=_.node_id)
                z2 = self.forward(x_2, edge_index_2, z=_.z, batch=_.batch, edge_weight=None, node_id=_.node_id)

                # print("INSIDE Sinkhorn")
                # import code
                # code.interact(local={**locals(), **globals()})
                # projector
                p1 = self.projection(z1)
                p2 = self.projection(z2)

                # predictor
                h1 = self.prediction(p1)
                h2 = self.prediction(p2)

                l1 = self.D(h1, p2) / 2
                l2 = self.D(h2, p1) / 2
                sim_loss = self.sim_coeff * (l1 + l2)
                sim_loss.backward()
                sim_loss_accum += sim_loss.item() / len(_)
                self.optimizer_1.step()



            train_loader = DataLoader(self.seal_dataset, batch_size=1024, shuffle=True, num_workers=0)
            c_loss_accum = 0 
            for _ in train_loader:
                _ = _.to(self.device)
                self.optimizer_1.zero_grad()
                self.optimizer_2.zero_grad()
                edge_index_1 = dropout_adj(_.edge_index, p=self.drop_edge_rate_1)[0]
                edge_index_2 = dropout_adj(_.edge_index, p=self.drop_edge_rate_2)[0]
                x_1 = drop_feature(
                    _.x,
                    self.drop_feature_rate_1,
                    self.sens_idx,
                    sens_flag=False,
                )
                x_2 = drop_feature(
                    _.x, self.drop_feature_rate_2, self.sens_idx
                )
                
                # classifier
                z1 = self.forward(x_1, edge_index_1, z=_.z, batch=_.batch, edge_weight=None, node_id=_.node_id)
                z2 = self.forward(x_2, edge_index_2, z=_.z, batch=_.batch, edge_weight=None, node_id=_.node_id)
                c = self.classifier(z1, edge_index_1, self.train_edge_index)
                c2 = self.classifier(z2, edge_index_2, self.train_edge_index)
                
                y_one_hot = F.one_hot(_.y, num_classes=2).float()

                # Binary Cross-Entropy
                l3 = (
                    F.binary_cross_entropy_with_logits(
                        c,
                        y_one_hot,
                    )
                    / 2
                )
                l4 = (
                    F.binary_cross_entropy_with_logits(
                        c2,
                        y_one_hot,
                    )
                    / 2
                )
                
                cl_loss = (1 - self.sim_coeff) * (l3 + l4)
                cl_loss.backward()
                self.optimizer_2.step()
                c_loss_accum += cl_loss.item() / len(_)
                
                
            loss = sim_loss_accum + c_loss_accum

            # Validation
            self.eval()
            val_s_loss, val_c_loss = self.ssf_validation_seal(
                self.val_x_1,
                self.val_edge_index_1,
                self.val_x_2,
                self.val_edge_index_2,
                self.labels,
            )
            # emb = self.forward(self.val_x_1, self.val_edge_index_1)
            # output = self.forwarding_predict(emb, self.val_edge_index_1, self.valid_edge_index)
            # preds = (output.squeeze() > 0).type_as(self.labels)
            # auc_roc_val = roc_auc_score(
            #     self.valid_edge_labels.detach().cpu().numpy(),
            #     output.detach().cpu().numpy(),
            # )

            if epoch % 50 == 0:
                print(f"[Train] Epoch {epoch}:train_s_loss: {sim_loss_accum:.4f} | train_c_loss: {c_loss_accum:.4f} | val_s_loss: {val_s_loss:.4f} | val_c_loss: {val_c_loss:.4f}")

            if (val_c_loss + val_s_loss) < best_loss:
                self.val_loss = val_c_loss + val_s_loss

                print(f'{epoch} | {val_s_loss:.4f} | {val_c_loss:.4f}')
                best_loss = val_c_loss + val_s_loss
                if not os.path.exists("data"):
                    os.makedirs("data")
                torch.save(self.state_dict(), f"data/three_classifiers_weights_ssf_{self.dataset_name}_{self.encoder_name}_{self.decoder_name}.pt")

    def fit(self, epochs=300):
        
        if self.encoder_name == 'seal':
            self.fit_seal(epochs=epochs)
        else:
            # Train model
            t_total = time.time()
            best_loss = 10e10
            best_acc = 0
            only_batch_index, only_batch_label = None, None
            
            for epoch in tqdm(range(epochs + 1), desc='Training'):
                t = time.time()
                
                x = self.features
                edge_index = self.edge_index

                for sens_class in range(3):
                    sim_loss = 0
                    cl_loss = 0
                    total_loss = 0
                    train_loader = getattr(self, f'train_loader_{sens_class}')
                    for batch_edge_index, batch_labels in train_loader:

                        
                        # if only_batch_index is None and only_batch_label is None:
                        #     only_batch_index, only_batch_label = batch_edge_index, batch_labels
                        # else:
                        #     batch_edge_index, batch_labels = only_batch_index, only_batch_label
                            
                        # print(batch_edge_index, batch_labels)
                        
                        self.train()
                        self.optimizer[sens_class].zero_grad()
                    
                        # Skipping batches that do not match the pre-defined length.
                        # if len(batch_edge_index) < self.batch_size:
                        #     continue
                        
                        batch_edge_index, batch_labels = batch_edge_index.t().to(self.device), batch_labels.to(self.device)
                        
                        # import code
                        # code.interact(local={**locals(), **globals()})
                        z = self.forward(x, edge_index, sens_class=sens_class)
                        
                        c = self.classifier(z, edge_index, batch_edge_index, sens_class=sens_class)
                    
                        c = c.squeeze()
                        
                        l = (F.binary_cross_entropy_with_logits(c.float(), batch_labels.float(),)/ 2)
                        
                        l.backward()
                
                        self.optimizer[sens_class].step()

                        total_loss += l.item()

                    print(f"[Train - sens attr {sens_class}] Epoch {epoch}: train_total_loss: {(total_loss / len(train_loader)):.4f} cl_loss: {(cl_loss / len(train_loader)):.4f} sim_loss: {(sim_loss / len(train_loader)):.4f}")

                # Validation
                # if epoch % 10 == 0:
                #     self.eval()
                #     val_loss, val_cl_loss, val_ndkl_loss = self.ssf_validation(
                #         x,
                #         edge_index,
                #         self.labels,
                #     )
                #     emb = self.forward(x, edge_index)
                #     output = self.classifier(emb, edge_index, self.valid_edge_index)
                #     output = output.squeeze()
                #     preds = (output.squeeze() > 0).type_as(self.labels)
                    
                #     accuracy = accuracy_score(self.valid_edge_labels.argmax(dim=1).cpu(), preds.cpu())
                #     f1 = f1_score(self.valid_edge_labels.argmax(dim=1).cpu(), preds.cpu())
                #     print(f"Validation Accuracy: {accuracy:.4f} | Validation F1 Score: {f1:.4f}")
                    
                #     if val_loss < best_loss:
                #         self.val_loss = val_loss

                #         logger.info(f'New best loss - epoch: {epoch} | val_loss : {val_loss:.4f} | val_cl_loss : {val_cl_loss:.4f} | val_ndkl_loss : {val_ndkl_loss:.4f}')
                #         best_loss = val_loss
                #         if not os.path.exists("data"):
                #             os.makedirs("data")
                            
                torch.save(self.state_dict(), f"data/three_classifiers_weights_ssf_{self.dataset_name}_{self.encoder_name}_{self.decoder_name}.pt")
                        
    def fit_hyperparam_search(self, config=None, epochs=100):
        # Initialize Weights and Biases
        wandb.init(config=config)
        self.sim_coeff = wandb.config.sim_coeff

        # Train model
        t_total = time.time()
        best_loss = 10e10
        best_acc = 0

        for epoch in tqdm(range(epochs + 1), desc='Training'):
            t = time.time()

            x = self.features
            edge_index = self.edge_index

            sim_loss = 0
            cl_loss = 0
            total_loss = 0

            for batch_edge_index, batch_labels in tqdm(self.train_loader):

                self.train()
                self.optimizer.zero_grad()

                # Skipping batches that do not match the pre-defined length.
                if len(batch_edge_index) < self.batch_size:
                    continue
            
                batch_edge_index, batch_labels = batch_edge_index.t().to(self.device), batch_labels.to(self.device)

                z = self.forward(x, edge_index)

                c = self.classifier(z, edge_index, batch_edge_index)

                c = c.squeeze()

                idx = c.argsort(descending=True)
                sens_matrix = F.one_hot(self.sens[batch_edge_index].sum(0).long(), num_classes=3).long()

                soft_labels = (c[:, None] * sens_matrix)[idx, :]
                soft_labels = torch.softmax(soft_labels, dim=-1)

                ndkl_loss, sparsity_loss = ranked_kl_divergence(soft_labels, c, self.original_sens_dist, sens_matrix, -1, noise=1e-20)

                # Binary Cross-Entropy
                l = (F.binary_cross_entropy_with_logits(c.float(), batch_labels.float(),) / 2)

                cl_loss += l.item()
                sim_loss += ndkl_loss.item()

                l = l + (self.sim_coeff * ndkl_loss)

                l.backward()

                self.optimizer.step()

                total_loss += l.item()

            print(f"[Train] Epoch {epoch}: train_total_loss: {(total_loss / len(train_loader)):.4f} cl_loss: {(cl_loss / len(train_loader)):.4f} sim_loss: {(sim_loss / len(train_loader)):.4f}")

            # Log metrics to Weights and Biases
            wandb.log({
                "train_total_loss": total_loss / len(train_loader),
                "train_cl_loss": cl_loss / len(train_loader),
                "train_sim_loss": sim_loss / len(train_loader),
                "epoch": epoch
            })

            # Validation
            if epoch % 10 == 0:
                self.eval()
                val_loss, val_cl_loss, val_ndkl_loss = self.ssf_validation(
                    x,
                    edge_index,
                    self.labels,
                )
                emb = self.forward(x, edge_index)
                output = self.classifier(emb, edge_index, self.valid_edge_index)
                output = output.squeeze()
                preds = (output.squeeze() > 0).type_as(self.labels)

                accuracy = accuracy_score(self.valid_edge_labels.argmax(dim=1).cpu(), preds.cpu())
                f1 = f1_score(self.valid_edge_labels.argmax(dim=1).cpu(), preds.cpu())
                print(f"Validation Accuracy: {accuracy:.4f} | Validation F1 Score: {f1:.4f}")

                # Log validation metrics to Weights and Biases
                wandb.log({
                    "val_loss": val_loss,
                    "val_cl_loss": val_cl_loss,
                    "val_ndkl_loss": val_ndkl_loss,
                    "val_accuracy": accuracy,
                    "val_f1_score": f1,
                    "epoch": epoch
                })

                if val_loss < best_loss:
                    self.val_loss = val_loss

                    logger.info(f'New best loss - epoch: {epoch} | val_loss : {val_loss:.4f} | val_cl_loss : {val_cl_loss:.4f} | val_ndkl_loss : {val_ndkl_loss:.4f}')
                    best_loss = val_loss

        wandb.finish()


    def predict_GNN(self):

        self.load_state_dict(torch.load(f"data/three_classifiers_weights_ssf_{self.dataset_name}_{self.encoder_name}_{self.decoder_name}.pt"))
        self.eval()
        emb = self.forward(
            self.features.to(self.device), self.edge_index.to(self.device)
        )
        output = self.forwarding_predict(emb)

        output_preds = (
            (output.squeeze() > 0)
            .type_as(self.labels)[self.idx_test]
            .detach()
            .cpu()
            .numpy()
        )

        labels = self.labels.detach().cpu().numpy()
        idx_test = self.idx_test

        F1 = f1_score(labels[idx_test], output_preds, average="micro")
        ACC = accuracy_score(
            labels[idx_test],
            output_preds,
        )
        try:
            AUCROC = roc_auc_score(labels[idx_test], output_preds)
        except:
            AUCROC = "N/A"

        ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1 = (
            self.predict_sens_group(output_preds, idx_test)
        )

        SP, EO = self.fair_metric(
            output_preds,
            self.labels[idx_test].detach().cpu().numpy(),
            self.sens[idx_test].detach().cpu().numpy(),
        )

        return (
            ACC,
            AUCROC,
            F1,
            ACC_sens0,
            AUCROC_sens0,
            F1_sens0,
            ACC_sens1,
            AUCROC_sens1,
            F1_sens1,
            SP,
            EO,
        )
        
    def predict_seal(self):
        global evaluating
        evaluating = True

        self.load_state_dict(torch.load(f"data/three_classifiers_weights_ssf_{self.dataset_name}_{self.encoder_name}_{self.decoder_name}.pt"))
        self.eval()
        
        outputs_accum = []
        test_loader = DataLoader(self.seal_dataset_test, batch_size=1024, shuffle=True, num_workers=0)
        for _ in tqdm(test_loader, desc='Testing'):
            
            _ = _.to(self.device)
            
            edge_index = _.edge_index
            
            x = _.x
            
            emb = self.forward(x, edge_index, z=_.z, batch=_.batch, edge_weight=None, node_id=_.node_id)
            
            output = self.forwarding_predict(emb, self.edge_index, self.test_edge_index)
            outputs_accum.append(output.detach().cpu())
            
        outputs_accum = torch.cat(outputs_accum)
        
        return outputs_accum, None, None

    def predict(self):
        
        if self.encoder_name == 'seal':
            return self.predict_seal()
        else:
            
            global evaluating
            evaluating = True

            with torch.no_grad():
                self.load_state_dict(torch.load(f"data/three_classifiers_weights_ssf_{self.dataset_name}_{self.encoder_name}_{self.decoder_name}.pt"))
                self.eval()
                outputs = torch.zeros(size=(self.test_edge_index.shape[-1],), device=self.device)
                for i in range(3):
                    
                    sens_mask = self.sens[self.test_edge_index].sum(0) == i
                    z = self.forward(self.features, self.edge_index, sens_class=i)
                            
                    c = self.classifier(z, self.edge_index, self.test_edge_index, sens_class=i)
                    
                    c = c.squeeze()
                    
                    outputs[sens_mask] = c[sens_mask]
                    
            return outputs


    def fair_metric(self, pred, labels, sens):

        idx_s0 = sens == 0
        idx_s1 = sens == 1
        idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
        idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)
        parity = abs(sum(pred[idx_s0]) / sum(idx_s0) - sum(pred[idx_s1]) / sum(idx_s1))
        equality = abs(
            sum(pred[idx_s0_y1]) / sum(idx_s0_y1)
            - sum(pred[idx_s1_y1]) / sum(idx_s1_y1)
        )
        return parity.item(), equality.item()

    def predict_sens_group(self, output, idx_test):
        # pred = self.lgreg.predict(self.embs[idx_test])
        pred = output
        result = []
        for sens in [0, 1]:
            F1 = f1_score(
                self.labels[idx_test][self.sens[idx_test] == sens]
                .detach()
                .cpu()
                .numpy(),
                pred[self.sens[idx_test] == sens],
                average="micro",
            )
            ACC = accuracy_score(
                self.labels[idx_test][self.sens[idx_test] == sens]
                .detach()
                .cpu()
                .numpy(),
                pred[self.sens[idx_test] == sens],
            )
            try:
                AUCROC = roc_auc_score(
                    self.labels[idx_test][self.sens[idx_test] == sens]
                    .detach()
                    .cpu()
                    .numpy(),
                    pred[self.sens[idx_test] == sens],
                )
            except:
                AUCROC = "N/A"
            result.extend([ACC, AUCROC, F1])

        return result


def drop_feature(x, drop_prob, sens_idx, sens_flag=True):
    drop_mask = (
        torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1)
        < drop_prob
    )

    x = x.clone()
    drop_mask[sens_idx] = False

    x[:, drop_mask] += torch.ones(1).normal_(0, 1).to(x.device)

    # Flip sensitive attribute
    if sens_flag:
        x[:, sens_idx] = 1 - x[:, sens_idx]

    return x
