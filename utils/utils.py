import torch


clamp = lambda x: torch.clamp(x, min=-5, max=5)
