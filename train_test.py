import wandb
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask
import numpy as np
from utils.analyze import analyze_stability_for_molecules
import time
import torch
from sampling import sample



def analyze_and_save(epoch, model_sample, nodes_dist, args, device, dataset_info, prop_dist,
                     n_samples=1000, batch_size=100):
    print(f'Analyzing molecule stability at epoch {epoch}...')
    batch_size = min(batch_size, n_samples)
    assert n_samples % batch_size == 0
    molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    for i in range(int(n_samples/batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, node_mask = sample(args, device, model_sample,
                                                nodesxsample=nodesxsample)

        molecules['one_hot'].append(one_hot.detach().cpu())
        molecules['x'].append(x.detach().cpu())
        molecules['node_mask'].append(node_mask.detach().cpu())

    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    validity_dict, rdkit_tuple, random_mol = analyze_stability_for_molecules(molecules, dataset_info)

    wandb.log(validity_dict)
    if rdkit_tuple is not None:
        wandb.log({'Validity': rdkit_tuple[0][0], 'Uniqueness': rdkit_tuple[0][1], 'Novelty': rdkit_tuple[0][2]})
    return validity_dict, random_mol


def compute_loss_and_nll(args, generative_model, nodes_dist, x, h, node_mask, edge_mask, context):
    
    bs, n_nodes, n_dims = x.size()
    
    edge_mask = edge_mask.view(bs, n_nodes * n_nodes)
    
    assert_correctly_masked(x, node_mask)
    
    nll = generative_model(x, h, node_mask, edge_mask, context)
    
    N = node_mask.squeeze(2).sum(1).long()
    
    log_pN = nodes_dist.log_prob(N)

    assert nll.size() == log_pN.size()
    nll = nll - log_pN

    # Average over batch.
    nll = nll.mean(0)

    reg_term = torch.tensor([0.]).to(nll.device)
    mean_abs_z = 0.
    
    return nll, reg_term, mean_abs_z

def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:

            assert_correctly_masked(variable, node_mask)

def train_epoch(args, loader, epoch, model, device, dtype, optim, nodes_dist):
    
    model.train()
    n_iterations = len(loader)
    nll_epoch = []
    for i, data in enumerate(loader):
        x = data['ligand_atom_pos'].to(device, dtype)
        node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
        edge_mask = data['edge_mask'].to(device, dtype)
        one_hot = data['ligand_one_hot'].to(device, dtype)
        charges = data['ligand_atom_type'].to(device, dtype)
        
        x = remove_mean_with_mask(x, node_mask)
        
        # one_hot is a tensor of [a,b,1,c] shape, make it [a, b, c]
        one_hot = one_hot.reshape([one_hot.shape[0], one_hot.shape[1], one_hot.shape[-1]])
        check_mask_correct([x, one_hot, charges], node_mask)
        
        assert_mean_zero_with_mask(x, node_mask)
        
        h = {'categorical': one_hot, 'integer': charges}
        
        context = None
        
        optim.zero_grad()
        
        # transform batch through flow
        
        nll, reg_term, mean_abs_z = compute_loss_and_nll(args, model, nodes_dist, x, h, node_mask, edge_mask, context)
        
        loss = nll + args.ode_regularization * reg_term
        loss.backward()
        
        optim.step()
        
        if i % args.n_report_steps == 0:
            print(f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, "
                  f"Loss {loss.item():.2f}, NLL: {nll.item():.2f}, "
                  f"RegTerm: {reg_term.item():.1f}")
            
        nll_epoch.append(nll.item())
        
        wandb.log({"Batch NLL": nll.item()}, commit=True)
        
    wandb.log({"Train Epoch NLL": np.mean(nll_epoch)}, commit=False)
        
def test(args, loader, epoch, eval_model, device, dtype, nodes_dist, partition='Test'):
    eval_model.eval()
    with torch.no_grad():
        nll_epoch = 0
        n_samples = 0

        n_iterations = len(loader)

        for i, data in enumerate(loader):
            x = data['ligand_atom_pos'].to(device, dtype)
            batch_size = x.size(0)
            node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
            edge_mask = data['edge_mask'].to(device, dtype)
            one_hot = data['ligand_one_hot'].to(device, dtype)
            charges = data['ligand_atom_type'].to(device, dtype)

            x = remove_mean_with_mask(x, node_mask)
            one_hot = one_hot.reshape([one_hot.shape[0], one_hot.shape[1], one_hot.shape[-1]])
            check_mask_correct([x, one_hot, charges], node_mask)
            assert_mean_zero_with_mask(x, node_mask)
            
            h = {'categorical': one_hot, 'integer': charges}
            
            context = None
            
            nll, _, _ = compute_loss_and_nll(args, eval_model, nodes_dist, x, h,
                                                    node_mask, edge_mask, context)
            # standard nll from forward KL

            nll_epoch += nll.item() * batch_size
            n_samples += batch_size
            if i % args.n_report_steps == 0:
                print(f"\r {partition} NLL \t epoch: {epoch}, iter: {i}/{n_iterations}, "
                      f"NLL: {nll_epoch/n_samples:.2f}")

    return nll_epoch/n_samples



        