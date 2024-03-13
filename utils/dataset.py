from torch.utils.data import DataLoader, Dataset, Subset
import json
import torch
import os

class UAAGDataset(Dataset):
    def __init__(self, data_path):
        
        uaag_data = json.load(open(data_path, 'rb'))
        for name in uaag_data:
            uaag_data[name]['index'] = name
        
        self.data = []
        for sample in list(uaag_data.values()):
            # Discard the amino acid with less than 5 atoms (only if it's a glycine, which has 4 atoms)
            if len(sample['ligand_atom_type']) > 4:
                self.data.append(sample)
            elif len(sample['ligand_atom_type']) == 4 and sample['ligand_aa'] == 6:
                self.data.append(sample)
                
            else:
                continue
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample
    

def batch_stack(props):
    """
    Stack a list of torch.tensors so they are padded to the size of the
    largest tensor along each axis.

    Parameters
    ----------
    props : list of Pytorch Tensors
        Pytorch tensors to stack

    Returns
    -------
    props : Pytorch tensor
        Stacked pytorch tensor.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return torch.tensor(props)
    elif props[0].dim() == 0:
        return torch.stack(props)
    else:
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)
    
    
def drop_zeros(props, to_keep):
    """
    Function to drop zeros from batches when the entire dataset is padded to the largest molecule size.

    Parameters
    ----------
    props : Pytorch tensor
        Full Dataset


    Returns
    -------
    props : Pytorch tensor
        The dataset with  only the retained information.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return props
    elif props[0].dim() == 0:
        return props
    else:
        return props[:, to_keep, ...]

def collate_fn(batch):
    batch = {key: batch_stack([torch.tensor(sample[key]) for sample in batch]) for key in ['ligand_atom_type', 'ligand_atom_pos']}
    
    to_keep = batch['ligand_atom_type'].sum(0) > 0
    
    batch = {key: drop_zeros(prop, to_keep) for key, prop in batch.items()}
    
    atom_mask = batch['ligand_atom_type'] > 0
    batch['atom_mask'] = atom_mask
    
    # Obtain edges
    batch_size, n_nodes = atom_mask.size()
    edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
    
    # mask diagonal
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    
    batch['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)
    batch['ligand_atom_type'] = batch['ligand_atom_type'].unsqueeze(2)
    return batch

def retrieve_dataloader(cfg):
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    
    uaag_data = UAAGDataset(cfg.data_path)
    
    full_size = len(uaag_data)
    train_size = int(cfg.train_size * full_size)
    val_size = int(cfg.valid_size * full_size)
    test_size = full_size - train_size - val_size
    
    train_data, val_data, test_data = torch.utils.data.random_split(uaag_data, [train_size, val_size, test_size])
    train_set = Subset(train_data.dataset, train_data.indices)
    val_set = Subset(val_data.dataset, val_data.indices)
    test_set = Subset(test_data.dataset, test_data.indices)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader
    
    
def obtain_histogram(data_path):
    
    uaag_data = UAAGDataset(data_path)
    num_node = []
    for name in uaag_data:
        num_node.append(len(name['ligand_atom_type']))
    
    # Counter the histogram of num_node
    histogram = {}
    for i in num_node:
        if i in histogram:
            histogram[i] += 1
        else:
            histogram[i] = 1
    return histogram
    