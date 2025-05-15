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
# from sinkhorn import Sinkhorn
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
        
        except KeyboardInterrupt:
            logger.warning("Training interrupted... running inference with current model weights")
            
        finally:
            output = fair_model.predict()
            torch.save((output), f'three_classifiers_{args.dataset}_{args.fair_model.upper()}_{args.model.upper()}_{run}.pt')
            

if __name__ == '__main__':
    main()
