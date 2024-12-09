from .dataset import SimDataset
from .sim_cond_dataset import SimCondDataset

vae_datasets = {
    "sim": SimDataset,
    "sim_cond": SimCondDataset,
}