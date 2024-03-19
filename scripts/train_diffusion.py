import sys
sys.path.append('.')
from rdkit import Chem
import json
import numpy as np
from utils import bond_analyze
from utils.dataset import retrieve_dataloader
from train_test import train_epoch, analyze_and_save, test
import argparse
import os
import shutil

from configs.datasets_config import uaag_configs

from equivariant_diffusion import en_diffusion

import numpy as np
import torch
import wandb
import time
import pickle
from tqdm.auto import tqdm
from models import get_model, get_optim

from configs.datasets_config import uaag_configs

from rdkit import Chem
from rdkit.Chem import AllChem

def molecule_visualize(mol_list):
    output_mol = []
    output_file_mol = []
    for mol in range(len(mol_list)):
        pos, atom_type = mol_list[mol]
        x = pos[:, 0]
        y = pos[:, 1]
        z = pos[:, 2]
        atoms = [uaag_configs['atom_decoder'][i] for i in atom_type]
        
        edge_list = []
        
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                p1 = np.array([x[i], y[i], z[i]])
                p2 = np.array([x[j], y[j], z[j]])
                dist = np.sqrt(np.sum((p1 - p2) ** 2))
                atom1, atom2 = atoms[i], atoms[j]


                draw_edge_int = bond_analyze.get_bond_order(atom1, atom2, dist)
                if draw_edge_int:
                    edge_list.append(([i, j], draw_edge_int))
        molB = Chem.RWMol()
    
        for atom in atoms:
            molB.AddAtom(Chem.Atom(atom))
            
        for bond in edge_list:
            molB.AddBond(bond[0][0], bond[0][1], Chem.BondType.SINGLE)
        conf = Chem.Conformer()
        for idx, (x_pos, y_pos, z_pos) in enumerate(list(zip(x, y, z))):

            conf.SetAtomPosition(idx, (float(x_pos), float(y_pos), float(z_pos)))
            
        molB.AddConformer(conf)
        
        final_mol = molB.GetMol()
        mol_block = Chem.MolToMolBlock(final_mol)
        output_file_mol.append(mol_block)
        try:
            mol_wandb = wandb.Molecule.from_rdkit(final_mol)
            output_mol.append(mol_wandb)
        except:
            print(f"Sampled molecule {mol} is not valid")
            continue
        
        
    return output_mol, output_file_mol
        
        

def main(args):
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    dtype = torch.float32
    
    if not args.no_wandb:
        wandb.login()
        wandb.init(project="UAAG")
        wandb.config.update(args)
    
    print("Loading Dataloader")
    
    train_loader, val_loader, test_loader = retrieve_dataloader(args)
    
    print("Dataloader Setup")
    # Define the model (EGNN Flow)
    
    # Retrieve Dataloader
    # Define the dataloader for Protein, and amino acids

    # Define the Dataloader
    model, node_dist = get_model(args, device)
    
    model = model.to(device)
    
    optim = get_optim(args, model)
    
    best_nll_val = 1e8
    best_nll_test = 1e8
    
    for epoch in range(args.start_epoch, args.n_epochs):
        print(f"Epoch: {epoch}")
        start_epoch = time.time()
        train_epoch(args, train_loader, epoch, model, device, dtype, optim, node_dist)
        print(f"Epoch took {time.time() - start_epoch:.1f} seconds.")
        
        if epoch % args.test_epochs == 0:
            print("Testing")
            
            if isinstance(model, en_diffusion.EnVariationalDiffusion):
                wandb.log(model.log_info(), commit=True)
            
            _, random_mol_list = analyze_and_save(epoch, model, node_dist, args, device, uaag_configs, node_dist, n_samples=args.n_stability_samples)
            
            mol_list, MolBlock_list = molecule_visualize(random_mol_list)
            print(mol_list)
            # Test the model
            nll_val = test(args, val_loader, epoch, model, device, dtype, node_dist, partition='Val')
            nll_test = test(args, test_loader, epoch, model, device, dtype, node_dist, partition='Test')
            
            for mol_idx in range(len(MolBlock_list)):
                with open(f'outputs/{args.exp_name}_sampled_mol_{epoch}_{mol_idx}.mol', 'w') as f:
                    f.write(MolBlock_list[mol_idx])
            
            if nll_val < best_nll_val:
                best_nll_val = nll_val
                best_nll_test = nll_test
                
                # Save the model
                if nll_val < best_nll_val or nll_test < best_nll_test:
                    save_path = f"outputs/{args.exp_name}_best_model_{epoch}.pth"
                    torch.save(model.state_dict(), save_path)
                    with open('outputs/%s/args.pickle' % args.exp_name, 'wb') as f:
                        pickle.dump(args, f)
                        
            print('Val loss: %.4f \t Test loss:  %.4f' % (nll_val, nll_test))
            print('Best val loss: %.4f \t Best test loss:  %.4f' % (best_nll_val, best_nll_test))
                
            if not args.no_wandb:
                wandb.log({'Sampled Molecules': mol_list})
                wandb.log({"Val loss ": nll_val}, commit=True)
                wandb.log({"Test loss ": nll_test}, commit=True)
                wandb.log({"Best cross-validated test loss ": best_nll_test}, commit=True)
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='uaag_unconditional',)
    parser.add_argument('--model', type=str, default='egnn_dynamics',
                    help='our_dynamics | schnet | simple_dynamics | '
                         'kernel_dynamics | egnn_dynamics |gnn_dynamics')
    
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--start_epoch', type=int, default=0,
                    help='')
    parser.add_argument('--n_report_steps', type=int, default=100, help="reprt loss steps")
    parser.add_argument('--no_wandb', type=int, default=0, help="wandb setup")
    parser.add_argument('--no_cuda', action="store_false", help="Disable CUDA")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of workers")
    parser.add_argument('--test_epochs', type=int, default=10, help="Test epochs")
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
    parser.add_argument('--ode_regularization', type=float, default=1e-3)
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
    
    parser.add_argument('--n_stability_samples', type=int, default=500,
                    help='Number of samples to compute the stability')
    
    parser.add_argument('--normalization_factor', type=float, default=1,
                    help="Normalize the sum aggregation of EGNN")
    parser.add_argument('--aggregation_method', type=str, default='sum',
                        help='"sum" or "mean"')
    args = parser.parse_args()
    main(args)
    
    
    