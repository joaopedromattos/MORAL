from torch_geometric.data import Data
from utils import get_dataset
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.utils import convert
from torch_sparse import SparseTensor
from loguru import logger
from moral import MORAL
import torch.nn.functional as F
import random

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mask_graphair(old_split_edge, new_edge_index):
    logger.info("Masking edges...")
    mask = torch.zeros(new_edge_index.size(1), dtype=torch.bool).cpu()
    for edge in tqdm(old_split_edge):
        mask |= ((edge[0] == new_edge_index).any(0) & (edge[1] == new_edge_index).any(0))           
    return mask

def generate_array_greedy_dkl(n, p, dist_size=3):
    """
    Greedy approach that tries to minimize a KL-like deviation as positions fill.
    """
    actual_counts = np.zeros(dist_size)
    arr = []

    for i in range(n):
        # Avoid division by zero when i == 0
        if i == 0:
            choice = int(np.argmax(p))
        else:
            # Prefer the group that's currently most underrepresented relative to p
            desired = p * i
            deficit = desired - actual_counts
            choice = int(np.argmax(deficit))
        arr.append(choice)
        actual_counts[choice] += 1

    return torch.tensor(arr)


@logger.catch
def main():
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='facebook')
    parser.add_argument('--model', type=str, default='gae')
    parser.add_argument('--fair_model', type=str, default='fair_walk')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--exposure_coeff', type=float, default=50_000)
    parser.add_argument('--sim_coeff', type=float, default=0.5)
    parser.add_argument('--ranking_loss', action='store_true')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=500)
    
    args = parser.parse_args()

    runs = 3
    for run in range(0, runs):
        logger.info(f"Processing {args.dataset} dataset with {args.fair_model} model.")
        
        model = {
            'gae' : {
                'encoder': 'gcn',
                'decoder': 'gae',
            },
            'ncn' : {
                'encoder': 'gcn',
                'decoder': 'ncn',
            },
            'seal':{
                'encoder': 'seal',
                'decoder': 'seal',            
            }
        }[args.model]
        
        fair_model_class = MORAL

        adj, features, idx_train, idx_val, idx_test, labels, sens, sens_idx, data, splits = get_dataset(args.dataset)

        logger.info("Initializing model...")
        
        # Initiate the model (with default parameters).
        seed_everything(run)
    
        try:
        
            params = {
                'adj':adj,
                'features':features,
                'labels':labels,
                'idx_train':idx_train.long(),
                'idx_val':idx_val.long(),
                'idx_test':idx_test.long(),
                'sens':sens,
                'sens_idx': sens_idx,
                'num_hidden':128,
                'num_proj_hidden':128,
                'encoder': model['encoder'],
                'decoder': model['decoder'],
                'edge_splits': splits,
                'dataset_name': args.dataset,
                'device': args.device,
                'batch_size': args.batch_size,
                'sim_coeff': args.sim_coeff,
                'lr': args.lr
            }
            
            fair_model = fair_model_class(**params)
            
            fair_model.fit(epochs=args.epochs)

            output = fair_model.predict()
            torch.save((output), f'three_classifiers_{args.dataset}_{args.fair_model.upper()}_{args.model.upper()}_{run}.pt')
            
            pi = F.one_hot(data.y[data.edge_index].sum(0).long(), num_classes=3).float().sum(0) / data.edge_index.shape[1]
            
            K = 1000 # Predicting 1000 edges
            final_output = torch.zeros(size=(K,))
            final_labels = torch.zeros(size=(K,))
            output_array_positions = generate_array_greedy_dkl(K, pi.numpy())

            # WARNING: The snippet below combines the edges in the test set assuming
            # there are enough samples from each sensitive type for the chosen $K$.
            for sens_value in range(3):
                mask = output_array_positions == sens_value
                sens_mask = sens == sens_value
                sens_val_outputs_sorted, idx = output[sens_mask].sort(descending=True)
                final_output[mask] = sens_val_outputs_sorted[:mask.sum()]
                
                # Selecting the labels of the current sensitive values in the order obtained when sorting by score.
                final_labels[mask] = labels[sens_mask][idx][:mask.sum()]
                
            torch.save((final_output, final_labels), f'three_classifiers_{args.dataset}_{args.fair_model.upper()}_{args.model.upper()}_{run}_final_ranking.pt')
            
        except KeyboardInterrupt:
            logger.warning("Training interrupted... running inference with current model weights")
            
        finally:
            output = fair_model.predict()
            torch.save((output), f'three_classifiers_{args.dataset}_{args.fair_model.upper()}_{args.model.upper()}_{run}.pt')
            

if __name__ == '__main__':
    main()
