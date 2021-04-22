import pytorch_lightning as pl
import torch

from torch.nn.functional import interpolate 
from pathlib import Path

from util import arguments
from dataset.dataset import ImplicitDataset
from model.model import Network, implicit_to_img, SimpleImplicitScene
from util.visualize_rgb import visualize_implicit_rgb

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

import os
import numpy as np

class ImplicitTrainer(pl.LightningModule):

    def __init__(self, kwargs):
        super(ImplicitTrainer, self).__init__()
        self.hparams = kwargs
        #self.model = Network()
        self.model = SimpleImplicitScene()
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
    
    def val_dataloader(self):
        dataset = self.dataset('val')
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, drop_last=False)
    
    def forward(self, batch):
        out = self.model(batch['points'])
        return out

    def training_step(self, batch, batch_idx):
        out = self.forward(batch) # (BS, C, 224, 224)
        loss = torch.nn.functional.l1_loss(out, batch['target'])
        #self.log('loss', loss)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss = torch.nn.functional.mse_loss(out, batch['target'])

        output_vis_path = Path("runs") / self.hparams.experiment / f"vis" / f'{(self.global_step // 100):05d}'
        output_vis_path.mkdir(exist_ok=True, parents=True)  
        visualize_implicit_rgb(out[0], output_vis_path / 'cat_out.png')

        #self.log('val_loss', loss)        
        return {'val_loss': loss}


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
        gpus=[args.gpu] , num_sanity_val_steps=args.sanity_steps, checkpoint_callback=checkpoint_callback, max_epochs=args.max_epoch, 
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
