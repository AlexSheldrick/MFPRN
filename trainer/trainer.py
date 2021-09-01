import pytorch_lightning as pl
import torch

import torch.nn as nn
from pathlib import Path
import numpy as np
import pyexr

from util import arguments
from dataset.dataset import ImplicitDataset
from model.model import SimpleNetwork, implicit_to_mesh, implicit_rgb_only
from util.visualize import render_mesh, render_shapenet
from util.bookkeeping import list_model_ids

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

import os

class ImplicitTrainer(pl.LightningModule):

    def __init__(self, kwargs):
        super(ImplicitTrainer, self).__init__()
        self.save_hyperparameters(kwargs)
        #self.hparams = kwargs
        self.model = SimpleNetwork(self.hparams)

        self.dataset = lambda split: ImplicitDataset(split, self.hparams.datasetdir, self.hparams.splitsdir, self.hparams)
        if self.hparams.fieldtype == 'occupancy':
            self.level = 0.5
            self.lossfunc = torch.nn.functional.binary_cross_entropy_with_logits
            
        elif self.hparams.fieldtype =='sdf':
            self.level = 0
            self.lossfunc = torch.nn.functional.l1_loss
        else:
            self.level = 0.001
            self.lossfunc = torch.nn.functional.l1_loss
        print('running in config:',self.hparams.fieldtype,'\n', self.level, '\n')

    #Here you could set different learning rates for different layers
    def configure_optimizers(self):
        #divide parameters in normal params and rgb layer params
        rgb_param_list = ['lin_rgb.weight', 'lin_rgb.bias']
        rgb_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in rgb_param_list, self.model.named_parameters()))))
        base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in rgb_param_list, self.model.named_parameters()))))

        opt_g = torch.optim.Adam([
            {'params': base_params, 'lr':self.hparams.lr},
            {'params': rgb_params, 'lr':self.hparams.lr}
            ], lr=self.hparams.lr)
        
        return [opt_g], []
    
    def train_dataloader(self):
        dataset = self.dataset('train')
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, drop_last=True, pin_memory=True)
    
    def val_dataloader(self):
        dataset = self.dataset('val')
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, drop_last=False)
    
    def test_dataloader(self):
        dataset = self.dataset('test')
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, drop_last=False)
    

    def forward(self, batch):
        sdf, rgb = self.model(batch)
        return sdf, rgb

    def training_step(self, batch, batch_idx):

        sdf, rgb = self.forward(batch)
        loss = self.losses_and_logging(batch, sdf, rgb, 'train')

        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):

        sdf, rgb = self.forward(batch)

        loss = self.losses_and_logging(batch, sdf, rgb, 'val')
        if batch_idx == 0: self.visualize(batch)      
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):

        sdf, rgb = self.forward(batch)

        loss = self.losses_and_logging(batch, sdf, rgb, 'test')
        self.visualize(batch, test = True)      
        return {'val_loss': loss}
    
    def losses_and_logging(self, batch, sdf, rgb, mode):

        sdf_loss = self.lossfunc(sdf, batch['sdf'])
        rgb_loss = torch.nn.functional.l1_loss(rgb, batch['rgb']) #roughly the same scale as clamped sdf
        loss = sdf_loss + rgb_loss

        #self.log_dict({f'{self.hparams.fieldtype}_loss': sdf_loss})    (f'{self.hparams.fieldtype}_loss', sdf_loss)
        
        self.log(f'{mode}_{self.hparams.fieldtype}_loss', sdf_loss)
        self.log(f'{mode}_rgb_loss', rgb_loss)
        self.log(f'{mode}_loss', loss)
        
        """if mode == 'val':
            self.logger.experiment.add_scalars('train_loss', {f'{mode}': loss}, global_step = self.global_step)
            self.logger.experiment.add_scalars('train_rgb_loss', {f'{mode}': rgb_loss}, global_step = self.global_step)
            self.logger.experiment.add_scalars(f'train_{self.hparams.fieldtype}_loss', {f'{mode}': sdf_loss}, global_step = self.global_step)
            self.log('val_loss', loss)         """                                                         

        return loss

    def visualize(self, batch, test = False):
        output_vis_path = Path("runs") / self.hparams.fieldtype / self.hparams.experiment / "vis" / f'{(self.global_step // 100):05d}'
        if test:
            id_dict = list_model_ids(save=False)
            output_vis_path = Path("runs") / self.hparams.fieldtype / self.hparams.experiment / str(self.hparams.test).split('/')[-1] / batch['name'][0].split('/')[-1]

        output_vis_path.mkdir(exist_ok=True, parents=True)      
        #images = torch.cat((data[0], batch['target'][0]), dim=-1)
        #self.logger.experiment.add_image('output/target', images, self.current_epoch)
        implicit_to_mesh(self.model, batch, (self.hparams.res, self.hparams.res, self.hparams.res), self.level, output_vis_path/'mesh.obj')
        
        try:
            images = render_mesh(output_vis_path/'mesh.obj')
            images = images.detach().cpu().numpy().reshape(-1,images.shape[2],4).transpose(2,0,1)

            mesh_idx = 'mesh idx: ' + batch["name"][0].split('/')[-1]
            if test:
                shapenet_idx = int(batch["name"][0].split('/')[-1].lstrip('0'))

                #Do only RGB inference on GT vertices and render

                obj_path = str(batch["name"][0] + '/norm.obj')
                sample_idx = str(batch["sample_idx"][0]).zfill(3)
                
                image_input = np.ones_like(images)
                pyexr_array = pyexr.read(str(batch['name'][0] + f'/_r_{sample_idx}.exr')).transpose(2, 0, 1)[:4]

                #correct would be to normalize but i'm lazy
                pyexr_array[:3] = pyexr_array[:3] / 3
                image_input[:, :224, :224] = pyexr_array
                
                implicit_rgb_only(self.model, batch, obj_path=(obj_path), output_path=output_vis_path/'mesh_rgb.obj')
                #(network, batch, obj_path, output_path)
                images_rgb_inference = render_mesh(output_vis_path/'mesh_rgb.obj')
                images_rgb_inference = images_rgb_inference.detach().cpu().numpy().reshape(-1,images_rgb_inference.shape[2],4).transpose(2,0,1)

                #render shapenet original as a comparison
                #render shapenet broke
                image_shapenet = np.ones_like(images_rgb_inference)
                #image_shapenet, _ = render_shapenet(id_dict, outpath=None, idx=shapenet_idx, batchsize=4, distance=1.7, verbose=False)
                #image_shapenet = image_shapenet.detach().cpu().numpy().reshape(-1,image_shapenet.shape[2],4).transpose(2,0,1)
                
                #concat everything for tensorboard
                #tensorboard expects (batch_size, height, width, channels)
                #images is 4, 672, 224(*4)
                images = np.concatenate((image_input, images, images_rgb_inference, image_shapenet), axis=2)
            
            self.logger.experiment.add_image(mesh_idx, images[:3], self.current_epoch)
        except: FileNotFoundError

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def train_scene_net(args):
    model = ImplicitTrainer(args)
    seed_everything(args.seed)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join("runs", f'{model.hparams.fieldtype}/{model.hparams.experiment}','checkpoints'),
        filename='{epoch}-{val_loss:.4f}',
        save_top_k=2, save_last=True, monitor='val_loss', verbose=False, period=args.save_epoch
        )
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join("runs", 'logs/'), name=f'ablation/{model.hparams.fieldtype}/{model.hparams.experiment}', )
    trainer = Trainer(
        gpus=[args.gpu] , num_sanity_val_steps=args.sanity_steps, checkpoint_callback=True, callbacks=[checkpoint_callback], max_epochs=args.max_epoch, 
        limit_val_batches=args.val_check_percent, val_check_interval=min(args.val_check_interval, 1.0), 
        check_val_every_n_epoch=max(1, args.val_check_interval), resume_from_checkpoint=args.resume, logger=tb_logger, benchmark=True, 
        profiler=args.profiler, precision=args.precision#, accumulate_grad_batches=4
        )
    if args.test is not None:
        model = ImplicitTrainer.load_from_checkpoint(args.test, strict = False)
        #model.hparams.num_points = 50000
        #model.level = 1-0.999
        model.hparams.batch_size = 1
        model.hparams.res = 256
        model.hparams.split = args.splitsdir
        model.hparams.test = args.test
        trainer.test(model)
        #trainer.fit(model)
    
    else: 
        trainer.fit(model)
        model.hparams.batch_size = 1
        model.hparams.res = 256
        #trainer.test(model)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning) 
    _args = arguments.parse_arguments()
    #_args.batch_size = 4
    train_scene_net(_args)

    """
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # create grid of images
    img_grid = torchvision.utils.make_grid(images)
    """