from typing import Optional
import random
import os
import numpy as np
import torch
from torch.backends import cudnn


def reseed_everything(seed: Optional[int]):
    """
    Set random seed for reproducibility across all random number generators.
    
    Args:
        seed: Random seed. If None, no seeding is performed.
    """
    if seed is None:
        return

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
