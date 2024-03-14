import wandb
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask
import numpy as np
import utils
import time
import torch

def train_epoch(args, loader, epoch, model, device, dtype. optim, nodes_dist):
    pass