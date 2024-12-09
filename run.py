import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from utils import vae_datasets
from models import *
from experiment import VAEXperiment
import pytorch_lightning as pl
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from pytorch_lightning.strategies import DDPStrategy

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/sim_vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

save_dir = os.path.join(config['logging_params']['save_dir'], config['exp_params']['name'])
tb_logger =  TensorBoardLogger(save_dir=save_dir,
                               name=config['model_params']['name'],
                               version=f"seed_{config['exp_params']['manual_seed']}")
csv_logger = CSVLogger(save_dir=save_dir,
                          name=config['model_params']['name'],
                          version=f"seed_{config['exp_params']['manual_seed']}")

# For reproducibility
pl.seed_everything(config['exp_params']['manual_seed'], True)

model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment(model,
                          config['exp_params'])

data = vae_datasets[config["data_params"]['name']](**config["data_params"], pin_memory=len(config['trainer_params']['devices']) != 0)

data.setup()

runner = Trainer(logger=[tb_logger, csv_logger],
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=2, 
                                     dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                     monitor= "val_loss",
                                     save_last= True),
                 ],
                 strategy=DDPStrategy(find_unused_parameters=False),
                 **config['trainer_params'])


Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)