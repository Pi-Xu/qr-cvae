from .dataset import SimDataset
from .sim_cond_dataset import SimCondDataset
from .utils import clamp

vae_datasets = {
    "sim": SimDataset,
    "sim_cond": SimCondDataset,
}