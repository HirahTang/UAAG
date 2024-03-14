from rdkit import Chem
import json
import numpy as np
from utils.dataset import retrieve_dataloader
from train_test import train_epoch
import argparse
import os
import shutil
import sys
sys.path.append('.')
import numpy as np
import torch
import wandb
import time
import pickle
from tqdm.auto import tqdm
from models import get_model, get_optim

def main(args):
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    dtype = torch.float32
    
    if not args.no_wandb:
        wandb.login()
        wandb.init(project="UAAG")
        wandb.config.update(args)
        
    train_loader, val_loader, test_loader = retrieve_dataloader(args)
    
    # Define the model (EGNN Flow)
    
    # Retrieve Dataloader
    # Define the dataloader for Protein, and amino acids

    # Define the Dataloader
    model, node_dist = get_model(args, device, train_loader)
    
    model = model.to(device)
    
    optim = get_optim(args, model)
    
    best_nll_val = 1e8
    best_nll_test = 1e8
    
    for epoch in range(args.start_epoch, args.n_epochs):
        start_epoch = time.time()
        train_epoch()
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='egnn_dynamics',
                    help='our_dynamics | schnet | simple_dynamics | '
                         'kernel_dynamics | egnn_dynamics |gnn_dynamics')
    
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--start_epoch', type=int, default=0,
                    help='')
    
    parser.add_argument('--no_wandb', action="store_false", help="Disable wandb logging")
    parser.add_argument('--no_cuda', action="store_false", help="Disable CUDA")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of workers")
    parser.add_argument('--data_path', type=str, default="data/uaag_data_small.json", help="Path to the data")
    parser.add_argument('--train_size', type=float, default=0.8, help="Train size")
    parser.add_argument('--valid_size', type=float, default=0.1, help="Validation size")
    parser.add_argument('--test_size', type=float, default=0.1, help="Test size")
    parser.add_argument('--lr', type=float, default=2e-4)
    
    parser.add_argument('--diffusion_steps', type=int, default=500)
    parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                        help='learned, cosine')
    parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5,
                        )
    parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                        help='vlb, l2')
    
    parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 1],
                    help='normalize factors for [x, categorical, integer]')
    # EGNN args -->
    parser.add_argument('--n_layers', type=int, default=6,
                        help='number of layers')
    parser.add_argument('--inv_sublayers', type=int, default=1,
                        help='number of layers')
    parser.add_argument('--nf', type=int, default=128,
                        help='number of layers')
    parser.add_argument('--tanh', type=eval, default=True,
                        help='use tanh in the coord_mlp')
    parser.add_argument('--attention', type=eval, default=True,
                        help='use attention in the EGNN')
    parser.add_argument('--norm_constant', type=float, default=1,
                        help='diff/(|diff| + norm_constant)')
    parser.add_argument('--sin_embedding', type=eval, default=False,
                        help='whether using or not the sin embedding')
    
    parser.add_argument('--normalization_factor', type=float, default=1,
                    help="Normalize the sum aggregation of EGNN")
    parser.add_argument('--aggregation_method', type=str, default='sum',
                        help='"sum" or "mean"')
    args = parser.parse_args()
    main(args)
    
    
    