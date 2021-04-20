import pytorch_lightning as pl
import torch

from torch.nn.functional import interpolate 
from pathlib import Path

from util import arguments
from dataset.dataset import ImplicitDataset
from model.model import Network

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

import os
import numpy as np

class ImplicitTrainer(pl.LightningModule):

    def __init__(self, kwargs):
        super(ImplicitTrainer, self).__init__()
        self.hparams = kwargs
        self.model = Network()
        self.dataset = lambda split: ImplicitDataset(split, self.hparams.datasetdir, self.hparams.splitsdir, self.hparams)

    #Here you could set different learning rates for different layers
    def configure_optimizers(self):
        opt_g = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr':self.hparams.lr}
            ], lr=self.hparams.lr)

        return [opt_g], []
    
    def train_dataloader(self):
        dataset = self.dataset('train')
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, drop_last=True, pin_memory=True)
    """
    def val_dataloader(self):
        dataset = self.dataset('val')
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, drop_last=False)

    def test_dataloader(self):
        dataset = self.dataset('test')
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, drop_last=False)"""
    
    def forward(self, batch):
        out = self.model(batch)
        return out

    def training_step(self, batch, batch_idx):
        out = self.forward(batch)

        loss = self.losses_and_logging(batch, 'train')
        
        return {'loss': loss}
    """
    def validation_step(self, batch, batch_idx):
        logits, depthmap, point_cloud = self.forward(batch)
        
        # additional supervision
        if self.hparams.subsample_points == 0:
            occupancies = batch['occupancies']
        else:
            _, occupancies_pointcloud = determine_occupancy(batch['mesh'], point_cloud.cpu().detach().numpy())
            occupancies_pointcloud = occupancies_pointcloud.to(logits.device)     
            occupancies = torch.cat((occupancies_pointcloud, batch['occupancies']), axis=1)

        if self.hparams.visualize:
            self.visualize_intermediates(batch, depthmap, point_cloud)

        #losses and logging
        loss = self.losses_and_logging(batch, depthmap, logits, occupancies, 'val')
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        _, depthmap, point_cloud = self.forward(batch)
        self.visualize_intermediates(batch, depthmap, point_cloud)

        return {'loss': 0}"""
    
    def losses_and_logging(self, batch, mode):

        mse_loss = torch.nn.functional.mse_loss(batch['sample_input'], batch['sample_target'], reduction='mean')
        loss = mse_loss

        self.log(f'{mode}_loss', loss)

        return loss

    def visualize_intermediates(self, batch, depthmap, point_cloud):
        pass
    
    #uncomment to log gradients
    """
    def on_after_backward(self):
    # example to inspect gradient information in tensorboard
        if self.trainer.global_step % 25 == 0:  # don't make the tf file huge
            print(self.project.sigma.grad)
            params = self.state_dict()
            for k, v in params.items():
                grads = v
                name = k
                self.logger.experiment.add_histogram(tag=name, values=grads, global_step=self.trainer.global_step)"""


def train_scene_net(args):
    model = ImplicitTrainer(args)
    seed_everything(args.seed)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join("runs", args.experiment, 'checkpoints'),
        filename='{epoch}-{loss:.2f}',
        save_top_k=2, save_last=True, monitor='loss', verbose=False, period=args.save_epoch
        )
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join("runs", 'logs/'), name=args.experiment)
    trainer = Trainer(
        gpus=args.gpu , num_sanity_val_steps=args.sanity_steps, checkpoint_callback=checkpoint_callback, max_epochs=args.max_epoch, 
        limit_val_batches=args.val_check_percent, val_check_interval=min(args.val_check_interval, 0.5), 
        check_val_every_n_epoch=max(1, args.val_check_interval), resume_from_checkpoint=args.resume, logger=tb_logger, benchmark=True, 
        profiler=args.profiler, precision=args.precision
        )

    trainer.fit(model)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning) 
    _args = arguments.parse_arguments()
    train_scene_net(_args)
