import pytorch_lightning as pl
import torch

import torch.nn as nn
from pathlib import Path
import numpy as np
import pyexr
from torchvision import transforms



from util import arguments
from dataset.dataset import ImplicitDataset
from model.model import SimpleNetwork, implicit_to_mesh, implicit_rgb_only, gradient, rotate_world_to_view, determine_implicit_surface
from util.visualize import render_mesh, render_shapenet
from util.bookkeeping import list_model_ids

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

from skimage.color import rgba2rgb
from PIL import ImageFont, ImageDraw
font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 16)


import os

class ImplicitTrainer(pl.LightningModule):

    def __init__(self, kwargs):
        super(ImplicitTrainer, self).__init__()
        self.save_hyperparameters(kwargs)
        #self.hparams = kwargs
        self.model = SimpleNetwork(self.hparams)

        my_transforms = self.hparams.transforms

        self.dataset = lambda split: ImplicitDataset(split, self.hparams.datasetdir, self.hparams.splitsdir, self.hparams, transform=my_transforms)
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
        sdf_surface, rgb_surface, sdf_grad = determine_implicit_surface(self.model, batch)
        loss = self.losses_and_logging(batch, sdf, rgb, sdf_surface, rgb_surface, sdf_grad, 'train')

        #if batch_idx == 5: self.visualize(batch, mode='train')      

        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):

        sdf, rgb = self.forward(batch)
        sdf_surface, rgb_surface, sdf_grad = determine_implicit_surface(self.model, batch)

        loss = self.losses_and_logging(batch, sdf, rgb, sdf_surface, rgb_surface, sdf_grad,'val')
        if batch_idx == 0: self.visualize(batch, mode='val')      
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):

        sdf, rgb = self.forward(batch)

        loss = self.losses_and_logging(batch, sdf, rgb, sdf_surface=None, rgb_surface=None, sdf_grad=None, mode = 'test')
        self.visualize(batch, mode='test')      
        return {'val_loss': loss}
    
    def losses_and_logging(self, batch, sdf, rgb, sdf_surface, rgb_surface, sdf_grad, mode):
        
        sdf_loss = self.lossfunc(sdf, batch['sdf'])
        rgb_loss = torch.nn.functional.l1_loss(rgb, batch['rgb']) #roughly the same scale as clamped sdf
        loss = sdf_loss + rgb_loss 

        if sdf_surface is not None and self.hparams.fieldtype == 'sdf':

            # reproject points or render pointcloud --> l2 loss on visible points
            # // OR // compare rgb to pixel values of subsampled indices (if the assumption holds that we stay on same raster)
            #try option2 here:
            #subsampled_pixels = batch['image'].permute(0, 2, 3, 1).reshape(batch['image'].shape[0], -1, 3) #swap points and channels
            #subsampled_pixels = subsampled_pixels[batch['depth_idx'], 0]#.reshape(-1,2000,3)

            #rgb_surface_loss = torch.nn.functional.l1_loss(rgb_surface, subsampled_pixels)
            # enforce norm grad = 1
            #rgb loss needs pointcloud rendering
            #sdf should be zero on surface
            sdf_surface_loss = torch.nn.functional.l1_loss(sdf_surface, torch.zeros_like(sdf_surface))
            # eikonal loss
            grad_lambda = 0.1 #this is a hyperparameter
            grad_loss = ((sdf_grad.norm(2, dim=-1) - 1) ** 2).mean()
            self.log(f'{mode}_grad_loss', grad_loss)
            self.log(f'{mode}_sdf_surface_loss', sdf_surface_loss)
            if 'train' in mode:
                loss = loss + grad_lambda*grad_loss
                loss = loss + grad_lambda*sdf_surface_loss
                #loss = loss + grad_lambda*rgb_surface_loss

        """if self.hparams.fieldtype == 'sdf' and 'train' in mode:
            # eikonal loss
            grad_lambda = 0.1 #this is a hyperparameter
            sdf_gradient = gradient(batch['points'], sdf)            
            grad_loss = ((sdf_gradient.norm(2, dim=-1) - 1) ** 2).mean()
            self.log(f'{mode}_grad_loss', grad_loss)
            loss += grad_lambda*grad_loss"""
        
        #self.log_dict({f'{self.hparams.fieldtype}_loss': sdf_loss})    (f'{self.hparams.fieldtype}_loss', sdf_loss)
        self.log(f'{mode}_{self.hparams.fieldtype}_loss', sdf_loss)
        self.log(f'{mode}_rgb_loss', rgb_loss)
        
        self.log(f'{mode}_loss', loss)                                                

        return loss

    def visualize(self, batch, mode):
        output_vis_path = Path("runs") / self.hparams.fieldtype / self.hparams.experiment / "vis" / f'{mode}' / f'{(self.global_step // 100):05d}'
        if 'test' in mode:
            id_dict = list_model_ids(save=False)
            output_vis_path = Path("runs") / self.hparams.fieldtype / self.hparams.experiment / str(self.hparams.test).split('/')[-1] / batch['name'][0].split('/')[-1]

        output_vis_path.mkdir(exist_ok=True, parents=True)      
        #images = torch.cat((data[0], batch['target'][0]), dim=-1)
        #self.logger.experiment.add_image('output/target', images, self.current_epoch)
        implicit_to_mesh(self.model, batch, (self.hparams.res, self.hparams.res, self.hparams.res), self.level, output_vis_path/'mesh.obj')
        
        try:
            sample_idx = str(batch["sample_idx"][0]).zfill(3)
            images = render_mesh(output_vis_path/'mesh.obj', viewangle = int(sample_idx))
            images = images.detach().cpu().numpy().reshape(-1,images.shape[2],4).transpose(2,0,1)

            if (('val' in mode) or ('train' in mode)):
                images = images[:3]
                tmp_images = batch['image'][0].clone().detach().cpu().numpy()
                tmp_images = np.concatenate((tmp_images, images), axis=1)
                images = np.empty((3,672,448))
                #images[...,:224] = tmp_images[:,:672]
                #images[...,224:] = tmp_images[:,672:]
                #6 images in 224 intervalls, GT; GTrender, 1,2,3,4
                images[:,:224, :224] = tmp_images[:,:224]
                images[:,:224,224:] = tmp_images[:,224:448]
                images[:,224:,:224] = tmp_images[:,448:896]
                images[:,224:,224:] = tmp_images[:,896:]

                

            mesh_idx = 'mesh idx: ' + batch["name"][0].split('/')[-1]
            if 'test' in mode:
                shapenet_idx = int(batch["name"][0].split('/')[-1].lstrip('0'))

                #Do only RGB inference on GT vertices and render

                obj_path = str(batch["name"][0] + '/norm.obj')
                
                
                image_input = np.ones_like(images)
                #pyexr_array = pyexr.read(str(batch['name'][0] + f'/_r_{sample_idx}.exr'))
                #pyexr_array[...,:3] = pyexr_array[...,:3]/3
                
                #pyexr_array = rgba2rgb(pyexr_array)
                #pyexr_array = torch.from_numpy(pyexr_array).permute(2, 0, 1)
                pyexr_array = batch['image'].detach().cpu().numpy()

                image_input[:3, :224, :224] = pyexr_array
                
                implicit_rgb_only(self.model, batch, obj_path=(obj_path), output_path=output_vis_path/'mesh_rgb.obj')
                #(network, batch, obj_path, output_path)
                images_rgb_inference = render_mesh(output_vis_path/'mesh_rgb.obj', viewangle = int(sample_idx))
                images_rgb_inference = images_rgb_inference.detach().cpu().numpy().reshape(-1, images_rgb_inference.shape[2], 4).transpose(2,0,1)

                #render shapenet original as a comparison
                #render shapenet broke
                image_shapenet = np.ones_like(images_rgb_inference)
                """#image_shapenet, _ = render_shapenet(id_dict, outpath=None, idx=shapenet_idx, batchsize=4, distance=1.7, verbose=False)
                #image_shapenet = image_shapenet.detach().cpu().numpy().reshape(-1,image_shapenet.shape[2],4).transpose(2,0,1)
                
                #concat everything for tensorboard
                #tensorboard expects (batch_size, height, width, channels)
                #images is 4, 672, 224(*4)"""
                images = np.concatenate((image_input, images, images_rgb_inference, image_shapenet), axis=2)
                """
                images = images.transpose(1,2,0)
                draw = ImageDraw.Draw(images)
                draw.text((0,0), "Input image", (0,0,0), font=font)
                draw.text((224,0), "Infernce Occupancy & Color", (0,0,0), font=font)
                draw.text((448,0), "Infernce Color only", (0,0,0), font=font)
                images = images.transpose(2,0,1)
                print('Im drawing')
                """

                
                #images = rgba2rgb(images.transpose(1, 2, 0)).transpose(2, 0, 1)
                #012 -> 201
            
            self.logger.experiment.add_image(mesh_idx, images[:3], self.current_epoch)
        except FileNotFoundError:
            pass

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
    if args.resume:
        #model.hparams.freeze_pretrained = None 
        #model.hparams.lr /= 5
        model.hparams.batch_size = 2
        model.hparams.num_points = 2000
        print(model.hparams.lr, model.hparams.freeze_pretrained)
        trainer.fit(model)
        model.hparams.batch_size = 1
        model.hparams.res = 256
        trainer.test(model)
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
        trainer.test(model)


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