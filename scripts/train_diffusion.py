from rdkit import Chem
import json
import numpy as np
from utils.dataset import retrieve_dataloader
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


def main():
    # Set up wandb

    # Transforms
    
    # Protein Featurizer
    
    
    # Retrieve Dataloader
    # Define the dataloader for Protein, and amino acids

    # Define the Dataloader
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    dtype = torch.float32
    
    if not args.no_wandb:
        wandb.login()
        wandb.init(project="UAAG")
        wandb.config.update(args)
        
    train_loader, val_loader, test_loader = retrieve_dataloader(args)
    
    # Define the model
    
    
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    main(args)
    
    
    