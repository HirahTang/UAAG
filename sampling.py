import numpy as np
import torch
import torch.nn.functional as F
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked
from utils import bond_analyze

def sample(args, device, model, nodesxsample=torch.tensor([10])):
    
    max_n_nodes = 14
    
    assert int(torch.max(nodesxsample)) <= max_n_nodes
    batch_size = len(nodesxsample)
    
    node_mask = torch.zeros(batch_size, max_n_nodes)
    for i in range(batch_size):
        node_mask[i, 0:nodesxsample[i]] = 1
        
     # Compute edge_mask

    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1).to(device)
    node_mask = node_mask.unsqueeze(2).to(device)
    
    context = None
    
    x, h = model.sample(batch_size, max_n_nodes, node_mask, edge_mask, context, fix_noise=False)
    
    assert_correctly_masked(x, node_mask)
    assert_mean_zero_with_mask(x, node_mask)
    
    one_hot = h['categorical']
    charges = h['integer']

    assert_correctly_masked(one_hot.float(), node_mask)
    
    assert_correctly_masked(charges.float(), node_mask)
    
    return one_hot, charges, x, node_mask