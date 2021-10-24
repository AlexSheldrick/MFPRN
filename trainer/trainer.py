import pytorch_lightning as pl
import torch

import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np

from util import arguments
from dataset.dataset import ImplicitDataset
from model.model import SimpleNetwork, implicit_to_mesh, implicit_rgb_only, rotate_world_to_view, project_3d_to_2d_gridsample
from util.visualize import render_mesh
#from util.bookkeeping import list_model_ids

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers

#from PIL import ImageFont, ImageDraw
#font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 16)

import os

class ImplicitTrainer(pl.LightningModule):

    def __init__(self, kwargs):
        super(ImplicitTrainer, self).__init__()
        self.save_hyperparameters(kwargs)
        self.model = SimpleNetwork(self.hparams)
        if self.hparams.test is not None:
            self.splitlen = len([x.strip() for x in (Path("../data/splits") / kwargs.splitsdir / f"train.txt").read_text().split("\n") if x.strip() != ""])
        else:
            self.splitlen = 2150
        print(self.splitlen)

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
        opt_g = torch.optim.AdamW([
            {'params': self.model.parameters(), 'lr':self.hparams.lr},
            ], lr=self.hparams.lr, weight_decay = self.hparams.lr_decay)
        """opt_g = torch.optim.AdamW([
            {'params': base_params, 'lr':self.hparams.lr},
            {'params': rgb_params, 'lr':self.hparams.lr}
            ], lr=self.hparams.lr, weight_decay = self.hparams.lr_decay)"""
        #lr is decided by finder, aka the maximum
        scheduler = torch.optim.lr_scheduler.OneCycleLR(opt_g, max_lr=self.hparams.lr, steps_per_epoch= (self.splitlen // (self.hparams.batch_size)), epochs=self.hparams.max_epoch)
        
        sched = {
            'scheduler': scheduler,
            'interval' : 'step',
            }
        
        #max 0.009, min0.0001
        opt_g.zero_grad(set_to_none=True)
        
        return [opt_g], [sched] #
    
    def train_dataloader(self):
        dataset = self.dataset('train')
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, pin_memory = True, drop_last=True)
    
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
        if self.hparams.implicit_grad and 'sdf' in self.hparams.fieldtype:
            #does 2 forward steps to determine gradients
            sdf, rgb, sdf_surface, rgb_surface, sdf_grad, surface_points = determine_implicit_surface(self.model, batch)
        else:
            sdf, rgb = self.forward(batch) 
            sdf_surface, rgb_surface, sdf_grad, surface_points = None, None, None, None

        loss = self.losses_and_logging(batch, sdf, rgb, sdf_surface, rgb_surface, sdf_grad, surface_points, 'train') 

        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        if self.hparams.implicit_grad and 'sdf' in self.hparams.fieldtype:
            #does 2 forward steps to determine gradients
            sdf, rgb, sdf_surface, rgb_surface, sdf_grad, surface_points = determine_implicit_surface(self.model, batch)
        else:
            sdf, rgb = self.forward(batch) 
            sdf_surface, rgb_surface, sdf_grad, surface_points = None, None, None, None

        loss = self.losses_and_logging(batch, sdf, rgb, sdf_surface, rgb_surface, sdf_grad, surface_points, 'val')
        #if batch_idx == 0: self.visualize(batch, mode='val')      
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):

        sdf, rgb = self.forward(batch)

        loss = self.losses_and_logging(batch, sdf, rgb, sdf_surface=None, rgb_surface=None, sdf_grad=None, surface_points=None, mode = 'test')
        self.visualize(batch, mode='test')      
        return {'val_loss': loss}
    
    def losses_and_logging(self, batch, sdf, rgb, sdf_surface, rgb_surface, sdf_grad, surface_points, mode):
        
        loss = 0        
        sdf_loss = self.lossfunc(sdf, batch['sdf'])
        rgb_loss = torch.nn.functional.l1_loss(rgb, batch['rgb']) #roughly the same scale as clamped sdf
        #sdf_loss = torch.nn.functional.mse_loss(sdf.float(), batch['sdf'].float(), reduction='mean')
        #rgb_loss = torch.nn.functional.mse_loss(rgb.float(), batch['rgb'].float(), reduction='mean')
        
        if sdf_surface is not None and self.hparams.fieldtype == 'sdf':
            
            #R, T = batch['camera'][...,:3,:3], batch['camera'][...,:3,3]
            #points = rotate_world_to_view(surface_points, R, T)
            #projected_points = project_3d_to_2d_gridsample(points, 245, 112, 112).unsqueeze(1) #could load/use an intrinsics file here
            #sampled_rgb = F.grid_sample(batch['image'], projected_points) #(BS, 3, 1, 1) sampled_rgb: #BS,C, 1,numpoints -> 2, 3, 1, numpoints
            #sampled_rgb = sampled_rgb.squeeze(2).transpose(2,1) #BS, numpoints, C
            #rgb_surface_loss = torch.nn.functional.l1_loss(rgb_surface, sampled_rgb)
            
            rgb_surface_loss = torch.nn.functional.l1_loss(rgb_surface, batch['rgb'][:,-500:])
            sdf_surface_loss = torch.nn.functional.l1_loss(sdf_surface, torch.zeros_like(batch['sdf'][:,-500:]))
            #if you wanted a hinge loss:
            #sdf_surface_loss = torch.maximum(sdf_surface_loss, 1e-4)

            # enforce norm grad = 1
            # eikonal loss
            grad_lambda = 0.01 #this is a hyperparameter
            grad_loss = ((sdf_grad.norm(2, dim=-1) - 1) ** 2).mean()
            self.log(f'{mode}_grad_loss', grad_loss)
            self.log(f'{mode}_sdf_surface_loss', sdf_surface_loss)
            self.log(f'{mode}_rgb_surface_loss', rgb_surface_loss)
            if 'train' in mode:
                pass
                loss = loss + grad_lambda*grad_loss
                #loss = loss + 5*sdf_surface_loss

                #loss = loss + 2*rgb_surface_loss
                
        
       #if self.hparams.implicit_grad is None: rgb_loss_lambda = 0.1
        #else: rgb_loss_lambda = 0.1 #we rely only on surface RGB as guidance now
        if self.hparams.fieldtype == 'sdf': loss = loss + 10*sdf_loss + 0.5*rgb_loss
        else: loss = loss + sdf_loss + 0.5*rgb_loss 
        self.log(f'{mode}_{self.hparams.fieldtype}_loss', sdf_loss)
        self.log(f'{mode}_rgb_loss', rgb_loss)
        
        self.log(f'{mode}_loss', loss)                                           

        return loss

    def visualize(self, batch, mode):
        output_vis_path = Path("runs") / self.hparams.fieldtype / self.hparams.experiment / "vis" / f'{mode}' / f'{(self.global_step // 100):05d}'
        if 'test' in mode:
            #id_dict = list_model_ids(save=False)
            output_vis_path = Path("runs") / self.hparams.fieldtype / self.hparams.experiment / str(self.hparams.test).split('/')[-1] / batch['name'][0].split('/')[-1]

        output_vis_path.mkdir(exist_ok=True, parents=True)      
        implicit_to_mesh(self.model, batch, (self.hparams.res, self.hparams.res, self.hparams.res), self.level, output_vis_path/'mesh.obj')
        
        try:
            sample_idx = str(batch["sample_idx"][0]).zfill(3)
            images = render_mesh(output_vis_path/'mesh.obj', viewangle = int(sample_idx))
            images = images.detach().cpu().numpy().reshape(-1,images.shape[2],4).transpose(2,0,1)

            if (('val' in mode)):
                images = images[:3]
                tmp_images = batch['image'][0].clone().detach().cpu().numpy()
                tmp_images = np.concatenate((tmp_images, images), axis=1)
                images = np.empty((3,672,448))
                #6 images in 224 intervalls, GT; GTrender, 1,2,3,4
                images[:,:224, :224] = tmp_images[:,:224]
                images[:,:224,224:] = tmp_images[:,224:448]
                images[:,224:,:224] = tmp_images[:,448:896]
                images[:,224:,224:] = tmp_images[:,896:]

            mesh_idx = 'mesh idx: ' + batch["name"][0].split('/')[-1]
            if 'test' in mode:   
                #Do only RGB inference on GT vertices and render
                obj_path = str(batch["name"][0] + '/norm.obj')              
                image_input = np.ones_like(images)
                pyexr_array = batch['image'].detach().cpu().numpy()
                image_input[:3, :224, :224] = pyexr_array                
                #implicit_rgb_only(self.model, batch, obj_path=(obj_path), output_path=output_vis_path/'mesh_rgb.obj')
                #(network, batch, obj_path, output_path)
                #images_rgb_inference = render_mesh(output_vis_path/'mesh_rgb.obj', viewangle = int(sample_idx))
                #images_rgb_inference = images_rgb_inference.detach().cpu().numpy().reshape(-1, images_rgb_inference.shape[2], 4).transpose(2,0,1)
                #images_rgb_inference = np.ones((4,1120,896))

                #render shapenet original as a comparison
                #render shapenet broke
                #image_shapenet = np.ones_like(images_rgb_inference)
                """#image_shapenet, _ = render_shapenet(id_dict, outpath=None, idx=shapenet_idx, batchsize=4, distance=1.7, verbose=False)
                #image_shapenet = image_shapenet.detach().cpu().numpy().reshape(-1,image_shapenet.shape[2],4).transpose(2,0,1)
                
                #concat everything for tensorboard
                #tensorboard expects (batch_size, height, width, channels)
                #images is 4, 672, 224(*4)"""
                #images = np.concatenate((image_input, images, images_rgb_inference, image_shapenet), axis=2)
                images = np.concatenate((image_input, images), axis=2)
                
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

#### render-implicit-surface with 1 gradient step
# reproject depth map +/- epsilon
# evaluate sdf on grid around points. Central gradient 7 points need to be evaluated (p, p+x, p-x, p+z, p-z, p+y, p-y)
# evaluate gradient-field via finite differences, from each point pi, take 1 step
# step: p_i_surface = p_i - sdf(p_i) * norm(grad(p_i, p_i+x ....))
# evaluate network on all points p_i_surface for RGB (and sdf for debugging)
# reproject pointcloud into image plane or render this pointcloud differentially
# l2 loss for all pixels that contain projected surface point
# for debugging: load network with checkpoint for forward passes etc.

def determine_implicit_surface(model, batch, mode = 'central', epsilon = 1e-4): #Could do this in forward pass already but don't have supervision
    #is cloning necessary? What is happening to the underlying memory
    probe = {}
    probe['image'] = batch['image']
    probe['voxels'] = batch['voxels']
    probe['camera'] = batch['camera']
    batch['depthpoints'] = batch['points'][:,-500:].clone().detach()
    batch['depthpoints'] = batch['depthpoints'] + batch['depthpoints'] * torch.normal(mean=0, std=1e-4, size = (batch['depthpoints'].shape[1], 1), device=batch['depthpoints'].device)
    # add x,y,z deviations for finite differences (central or forward)
    shifted_points, num_points, spacing = shift_points_finite_difference(batch['depthpoints'], epsilon, mode)

    ## split points to do everythin in 2 forward passes
    #batch_num_points = batch['points'].shape[1]
    #pointbatch_1 = batch['points'][:,:(batch_num_points // spacing)].clone().detach() #1 (first eigth or fifth)
    #pointbatch_2 = batch['points'][:,(batch_num_points // spacing):].clone().detach() #2 (rest)
    #probe['points'] = torch.cat((pointbatch_1, shifted_points), axis=1)
    probe['points'] = shifted_points

    #first forward pass
    
    sdf_gradpoints, _ = model(probe)

    #second forward pass
    ## gradient 
    sdf_grad = finite_differences_gradient(sdf_gradpoints, epsilon, mode)
    # take 1 gradient step, should land on surface
    surface_points = shifted_points[:,:num_points] - sdf_gradpoints[:,:num_points,None] * sdf_grad / sdf_grad.norm(2, keepdim=True, dim=-1)
    # re-evaluate sdf on surface & forwardpass rest of points
    probe['points'] = torch.cat((batch['points'], surface_points), axis=1)
    sdf_batch_2, rgb_batch_2 = model(probe)
    sdf, rgb = sdf_batch_2[:,:-num_points], rgb_batch_2[:,:-num_points]
    sdf_surface, rgb_surface = sdf_batch_2[:,-num_points:], rgb_batch_2[:,-num_points:]
    # bookkeeping   
    #sdf, rgb = torch.cat((sdf_1, sdf_2), axis=1), torch.cat((rgb_1, rgb_2), axis=1)
    
    return sdf, rgb, sdf_surface, rgb_surface, sdf_grad, surface_points

def finite_differences_gradient(points, epsilon, mode):

    if 'forward' in mode:
        #structure is points(0:num_points --> f(x-e), num_points:2*num_points --> x_shifted points etc., ..., 4num_points: --> base points)
        assert points.shape[1] % 4 == 0, 'forward difference gradient points not divisible by 4 --> 4-stencil input required'
        num_points = points.shape[1] // 4
        grad_x =(points[...,num_points:2*num_points] - points[...,:num_points]) / epsilon
        grad_y =(points[...,2*num_points:3*num_points] - points[...,:num_points]) / epsilon
        grad_z =(points[...,3*num_points:4*num_points] - points[...,:num_points]) / epsilon
        gradxyz = torch.cat((grad_x.unsqueeze(-1), grad_y.unsqueeze(-1), grad_z.unsqueeze(-1)), axis=-1)

    if 'central' in mode:
        #structure is points(0:num_points --> f(x-e), num_points:2*num_points --> x_shifted points etc., ..., 4num_points: --> base points)
        assert points.shape[1] % 7 == 0, 'central difference gradient points not divisible by 7 --> 7-stencil input required'
        num_points = points.shape[1] // 7
        grad_x =(points[...,num_points:2*num_points] - points[...,2*num_points:3*num_points]) / (2*epsilon)
        grad_y =(points[...,3*num_points:4*num_points] - points[...,4*num_points:5*num_points]) / (2*epsilon)
        grad_z =(points[...,5*num_points:6*num_points] - points[...,6*num_points:7*num_points]) / (2*epsilon)
        gradxyz = torch.cat((grad_x.unsqueeze(-1), grad_y.unsqueeze(-1), grad_z.unsqueeze(-1)), axis=-1)

    return gradxyz

def shift_points_finite_difference(points, epsilon, mode):
    with torch.no_grad():
        if 'central' in mode:
            num_stencilpoints = 7
            num_points = points.shape[1]
            x_shifted_positive = points + torch.tensor([epsilon,0,0], device=points.device)[None,None,:]
            x_shifted_negative = points - torch.tensor([epsilon,0,0], device=points.device)[None,None,:]
            y_shifted_positive = points + torch.tensor([0,epsilon,0], device=points.device)[None,None,:]
            y_shifted_negative = points - torch.tensor([0,epsilon,0], device=points.device)[None,None,:]
            z_shifted_positive = points + torch.tensor([0,0,epsilon], device=points.device)[None,None,:]      
            z_shifted_negative = points - torch.tensor([0,0,epsilon], device=points.device)[None,None,:]
            
            shifted_points = torch.cat((
                points, x_shifted_positive, x_shifted_negative, y_shifted_positive, y_shifted_negative,
                z_shifted_positive, z_shifted_negative              
                ), axis=1)

        if 'forward' in mode:
            num_stencilpoints = 4
            num_points = points.shape[1]
            x_shifted = points + torch.tensor([epsilon,0,0], device=points.device)[None,None,:]
            y_shifted = points + torch.tensor([0,epsilon,0], device=points.device)[None,None,:]
            z_shifted = points + torch.tensor([0,0,epsilon], device=points.device)[None,None,:]
            shifted_points = torch.cat((points, x_shifted, y_shifted, z_shifted), axis=1)

    return shifted_points, num_points, (num_stencilpoints+1)

def train_scene_net(args):
    model = ImplicitTrainer(args)
    seed_everything(args.seed)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join("runs", f'{model.hparams.fieldtype}/{model.hparams.experiment}','checkpoints'),
        filename='{epoch}-{val_loss:.4f}',
        save_top_k=2, save_last=True, monitor='val_loss', verbose=False, every_n_epochs=args.save_epoch
        )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join("runs", 'logs/'), name=f'ablation/{model.hparams.fieldtype}/{model.hparams.experiment}', )
    #args.profiler = pl.profiler.AdvancedProfiler(filename='profile')
    trainer = Trainer(
        gpus=[args.gpu],num_sanity_val_steps=args.sanity_steps, checkpoint_callback=True, callbacks=[checkpoint_callback, lr_monitor], max_epochs=args.max_epoch, 
        limit_val_batches=args.val_check_percent, val_check_interval=min(args.val_check_interval, 1.0), 
        check_val_every_n_epoch=max(1, args.val_check_interval), resume_from_checkpoint=args.resume, logger=tb_logger, benchmark=True, 
        profiler=args.profiler, precision=args.precision, #accumulate_grad_batches=6#, 
        )
    if args.resume:
        #model.hparams.freeze_pretrained = None 
        #model.hparams.lr /= 4
        #model.hparams.implicit_grad = True
        #model.hparams.batch_size = 7
        #model.hparams.num_points = 4500
        print(model.hparams.lr, model.hparams.freeze_pretrained)
        trainer.fit(model)
        #model.hparams.batch_size = 1
        #model.hparams.res = 256
        #trainer.test(model)
    elif (args.test is not None) and (args.resume is None):
        model = ImplicitTrainer.load_from_checkpoint(args.test, strict = False)
        #model.hparams.num_points = 50000
        #model.level = 1-0.999
        #model.hparams.num_surface_points = 3000
        model.hparams.num_points *= 3*model.hparams.batch_size
        model.hparams.batch_size = 1
        model.hparams.res = 256
        model.hparams.split = args.splitsdir
        model.hparams.test = args.test
        trainer.test(model)
        #trainer.fit(model)
    
    else: 
        
        # Run learning rate finder
        #model_checkpoint = ''
        #model = ImplicitTrainer.load_from_checkpoint(args.test, strict = False)
        if args.autoLR is not None:
            if args.precision == 16:
                trainer.scaler = torch.cuda.amp.GradScaler()        
            lr_finder = trainer.tuner.lr_find(model, min_lr = 1e-8, max_lr=1e2, num_training=100)
            # Results can be found in
            print(lr_finder.results)

            # Plot with
            #fig = lr_finder.plot(suggest=True)
            #fig.show()

            # Pick point based on plot, or get suggestion
            new_lr = lr_finder.suggestion()
            print(new_lr)       

            # update hparams of the model
            model.hparams.lr = new_lr

        # Fit model      
        trainer.fit(model)

        model.hparams.num_points *= 3*model.hparams.batch_size
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