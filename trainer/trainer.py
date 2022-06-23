from doctest import testfile
from pyclbr import Function
import pytorch_lightning as pl
import torch

import torch.nn as nn
from pathlib import Path
import numpy as np

from util import arguments
from dataset.dataset import ImplicitDataset
from model.model import SimpleNetwork, DeepSDFDecoder, PixelAligned2D3DEncoder, GlobalFeature2D3DEncoder, implicit_to_mesh, implicit_to_verts_faces, evaluate_network_on_grid, visualize_sdf, GaussFourierEmbedding
from util.visualize import render_mesh

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers

from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.io import load_obj, load_objs_as_meshes
import trimesh

import subprocess
try:
    from subprocess import DEVNULL  # Python 3.
except ImportError:
    DEVNULL = open(os.devnull, 'wb')


import os

class ImplicitTrainer(pl.LightningModule):

    def __init__(self, kwargs):
        super(ImplicitTrainer, self).__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(kwargs)
        #self.model = SimpleNetwork(self.hparams)
        self.encoder = PixelAligned2D3DEncoder(self.hparams)
        #self.encoder = GlobalFeature2D3DEncoder(self.hparams)
        self.decoder = DeepSDFDecoder(self.hparams)
        self.embedding = GaussFourierEmbedding(3, self.hparams.coordinate_embedding_size[0], self.hparams.coordinate_embedding_scale[0])

        #if self.hparams.test is not None:
        self.splitlen = len([x.strip() for x in (Path("../data/splits") / self.hparams.splitsdir / f"train.txt").read_text().split("\n") if x.strip() != ""])
        #else:
        #    self.splitlen = 2150
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
        """
        rgb_param_list = ['lin_rgb.weight', 'lin_rgb.bias']
        rgb_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in rgb_param_list, self.model.named_parameters()))))
        base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in rgb_param_list, self.model.named_parameters()))))
        opt_g = torch.optim.AdamW([
            {'params': base_params, 'lr':self.hparams.lr},
            {'params': rgb_params, 'lr':self.hparams.lr}
            ], lr=self.hparams.lr, weight_decay = self.hparams.lr_decay)
        """
        #model_params = self.model.parameters()
        model_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        opt_g = torch.optim.AdamW([
            {'params': model_params, 'lr':self.hparams.lr},
            ], lr=self.hparams.lr, weight_decay = self.hparams.lr_decay)
        
        #lr is decided by finder, aka the maximum
        num_steps = (self.splitlen // (self.hparams.batch_size))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(opt_g, max_lr=self.hparams.lr, steps_per_epoch= num_steps, epochs=self.hparams.max_epoch)
        
        sched = {
            'scheduler': scheduler,
            'interval' : 'step',
            }
        
        #opt_g.zero_grad(set_to_none=True)
        
        return [opt_g], [sched] #sched
    
    def train_dataloader(self):
        dataset = self.dataset('train')
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, pin_memory = True, drop_last=True)
    
    def val_dataloader(self):
        dataset = self.dataset('val')
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, drop_last=True)
    
    def test_dataloader(self):
        dataset = self.dataset('test')
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, drop_last=False)
    
    def forward(self, batch):
        latent_features = self.encoder(batch)
        #sizes = latent_features.size()
        #features = latent_features.unsqueeze(1).expand(sizes[0], batch['points'].size()[1], sizes[1])
        points = self.embedding(batch['points'])
        #features = torch.cat((features, points), axis=2)
        features = torch.cat((latent_features, points), axis=2)
        sdf, rgb = self.decoder(features) 
        #sdf, rgb = self.model(batch)              
        return sdf, rgb, latent_features

    def training_step(self, batch, batch_idx):
        #start of original model & loss
        opt = self.optimizers()
        opt.zero_grad()
        sdf, rgb, features = self.forward(batch) 
        
        loss_backward = None
        sdf_loss = self.lossfunc(sdf, batch['sdf'])
        loss = 0
        rgb_loss = torch.nn.functional.l1_loss(rgb, batch['rgb'])
        loss = 10*sdf_loss + 0.5*rgb_loss
        self.manual_backward(loss)

        #'''
        #quote1        
        if (self.current_epoch + 1 ) > 50 :
            try:
                #opt.zero_grad()
                model = (self.encoder, self.decoder)        
                v,f = zip(*[implicit_to_verts_faces(model, batch, (self.hparams.res, self.hparams.res, self.hparams.res), self.embedding,
                                    self.level, None, res_increase=self.hparams.inf_res, batch_idx=i) for i in range(len(batch['name']))]) #3sec per loop
                
                v, f= list(v), list(f)
                #check for min-indices
                min_length = min([len(v[i]) for i in range(len(v))])
                if min_length > 4000:
                    min_length = 4000
                    
                indices = [torch.randperm(len(v[i]))[:min_length] for i in range(len(v))]  
                gt_mesh_samples = batch['surface_points']
                exp_mesh = Meshes(v, f)                
                exp_mesh_samples = sample_points_from_meshes(exp_mesh, num_samples=2000) #- 0.5 #normalize [-0.5, 0.5]
                chamfer_loss, _ = chamfer_distance(exp_mesh_samples, gt_mesh_samples) #ignore normals
                self.manual_backward(chamfer_loss)

                #store upstream gradients        
                dL_dx_i =  [v[i].grad[indices[i]] for i in range(len(v))]   
                dL_dx_i = torch.stack(dL_dx_i)
                dL_dx_i[torch.isnan(dL_dx_i)] = 0           
                
                # use vertices to compute full backward pass
                opt.zero_grad()        
                verts = [v[i][indices[i]].clone().detach() for i in range(len(v))]  
                verts = torch.stack(verts)
                xyz = torch.tensor(verts, requires_grad = True,dtype=torch.float32, device=torch.device('cuda:0'))

                #sizes = features.size()
                #features = features.unsqueeze(1).expand(sizes[0], xyz.size()[1], sizes[1])
                #features = torch.cat((features, xyz), axis=2)
                sdf_verts = forward_pass_vertices(model, batch, self.embedding, xyz)
                loss_normals = torch.sum(sdf_verts) #/ torch.prod(torch.tensor([*sdf_verts.shape], device = 'cuda:0', requires_grad = False))
                self.manual_backward(loss_normals, retain_graph = True)        
                normals = xyz.grad/torch.norm(xyz.grad, 2, 2).unsqueeze(-1)

                # now assemble inflow derivative
                opt.zero_grad()            
                dL_ds_i = -torch.matmul(dL_dx_i.unsqueeze(2), normals.unsqueeze(-1)).squeeze(-1).squeeze(-1)

                # finally assemble full backward pass
                intermediate_loss = (dL_ds_i * sdf_verts)        
                loss_backward = torch.sum(intermediate_loss)
                self.manual_backward(loss_backward + rgb_loss)

                self.log(f'train_loss_backward', loss_backward) 
                self.log(f'train_chamfer_loss', chamfer_loss)   
            except RuntimeError as R:
                print('no vertices could be produced for {}'.format([i.split('/')[-1] for i in batch['name']]))
                print(R)
            except ValueError as V:
                print(V,'mesh was empty')

        #loss = chamfer_loss
        #quote2
        #'''        

        self.log(f'train_{self.hparams.fieldtype}_loss', sdf_loss)
        self.log(f'train_rgb_loss', rgb_loss)        
        self.log(f'train_loss', loss)
        
        self.log_dict({"loss": loss}, prog_bar=True)

        opt.step()
        self.lr_schedulers().step()
        return {'loss': loss}

    
    def validation_step(self, batch, batch_idx):
        
        sdf, rgb, features = self.forward(batch) 
        sdf_surface, rgb_surface, sdf_grad, surface_points = None, None, None, None

        loss = self.losses_and_logging(batch, sdf, rgb, sdf_surface, rgb_surface, sdf_grad, surface_points, 'val')
        #if batch_idx == 0: self.visualize(batch, mode='val')      #validate qualitative training progress
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):

        sdf, rgb, features = self.forward(batch)

        loss = self.losses_and_logging(batch, sdf, rgb, sdf_surface=None, rgb_surface=None, sdf_grad=None, 
                                        surface_points=None, mode = 'test')
        self.visualize(batch, mode='test')      
        return {'val_loss': loss}
    
    def losses_and_logging(self, batch, sdf, rgb, sdf_surface, rgb_surface, sdf_grad, surface_points, mode):
        
        loss = 0        
        sdf_loss = self.lossfunc(sdf, batch['sdf'])
        rgb_loss = torch.nn.functional.l1_loss(rgb, batch['rgb'])
                
        if self.hparams.fieldtype == 'sdf': loss = loss + 10*sdf_loss + 0.5*rgb_loss
        else: loss = loss + sdf_loss + 0.5*rgb_loss 

        self.log(f'{mode}_{self.hparams.fieldtype}_loss', sdf_loss)
        self.log(f'{mode}_rgb_loss', rgb_loss)        
        self.log(f'{mode}_loss', loss)                                   

        return loss

    def visualize(self, batch, mode):
        output_vis_path = Path("runs") / self.hparams.fieldtype / self.hparams.experiment / "vis" / f'{mode}' / f'{(self.global_step // 100):05d}'
        if 'test' in mode:
            output_vis_path = Path("runs") / self.hparams.fieldtype / self.hparams.experiment / str(self.hparams.test).split('/')[-1] / batch['name'][0].split('/')[-1]

        output_vis_path.mkdir(exist_ok=True, parents=True)
        model = (self.encoder, self.decoder)      
        implicit_to_mesh(model, batch, (self.hparams.res, self.hparams.res, self.hparams.res), self.embedding, self.level, output_vis_path/'mesh.obj')
        
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
                obj_path = str(batch["name"][0] + '/norm.obj')              
                image_input = np.ones_like(images)
                pyexr_array = batch['image'].detach().cpu().numpy()
                image_input[:3, :224, :224] = pyexr_array  
                images = np.concatenate((image_input, images), axis=2)
            
            self.logger.experiment.add_image(mesh_idx, images[:3], self.current_epoch)
        except FileNotFoundError:
            pass

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

## function diffable_3d_loss
## in: features
## out: 


#### render-implicit-surface with 1 gradient step
# reproject depth map +/- epsilon
# evaluate sdf on grid around points. Central gradient 7 points need to be evaluated (p, p+x, p-x, p+z, p-z, p+y, p-y)
# evaluate gradient-field via finite differences, from each point pi, take 1 step
# step: p_i_surface = p_i - sdf(p_i) * norm(grad(p_i, p_i+x ....))
# evaluate network on all points p_i_surface for RGB (and sdf for debugging)
# reproject pointcloud into image plane or render this pointcloud differentially
# l2 loss for all pixels that contain projected surface point
# for debugging: load network with checkpoint for forward passes etc.

class chamfer_dist(Function):
    @staticmethod
    def forward(ctx, i):
        result = i



def forward_pass_vertices(model, batch, embedding, vertices):
    encoder, decoder = model
    probe = {}
    probe['image'] = batch['image']
    probe['voxels'] = batch['voxels']
    probe['camera'] = batch['camera']
    probe['points'] = vertices

    #first forward pass
    features = encoder(probe)
    points = embedding(vertices)    
    features = torch.cat((features, points), axis=2)
    sdf_vertices, _ = decoder(features)
    return sdf_vertices

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
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join("runs", 'logs/'), name=f'diffmc/{model.hparams.fieldtype}/{model.hparams.experiment}', )
    
    trainer = Trainer(
        gpus=[args.gpu],num_sanity_val_steps=args.sanity_steps, enable_checkpointing=True, callbacks=[checkpoint_callback, lr_monitor], max_epochs=args.max_epoch, 
        limit_val_batches=args.val_check_percent, val_check_interval=min(args.val_check_interval, 1.0), 
        check_val_every_n_epoch=max(1, args.val_check_interval),  logger=tb_logger, benchmark=True, 
        profiler=args.profiler, precision=args.precision, log_every_n_steps=1, #resume_from_checkpoint=args.resume, #accumulate_grad_batches=6#, 
        )

    if args.resume:
        #model.hparams.max_epoch = args.max_epoch
        #model.hparams.lr = args.lr
        model = model.load_from_checkpoint(args.resume)
        model.hparams.batch_size = 6
        #model.encoder.requires_grad_(False)
        #model.hparams.splitsdir = 'overfit'
        trainer.fit(model)

    elif (args.test is not None) and (args.resume is None):
        model = ImplicitTrainer.load_from_checkpoint(args.test, strict = False)
        model.hparams.num_points *= 2*model.hparams.batch_size
        model.hparams.batch_size = 1
        model.hparams.res = 256
        model.hparams.splitsdir = args.splitsdir
        model.hparams.test = args.test
        print(model.hparams.splitsdir)
        trainer.test(model)
    
    else:         
        # Run learning rate finder
        if args.autoLR is not None:
            if args.precision == 16:
                trainer.scaler = torch.cuda.amp.GradScaler()        
            lr_finder = trainer.tuner.lr_find(model, min_lr = 1e-8, max_lr=1e2, num_training=100)
            # Results can be found in
            print(lr_finder.results)
            new_lr = lr_finder.suggestion()
            print(new_lr)       

            # update hparams of the model
            model.hparams.lr = new_lr

        # Fit model
        #model = ImplicitTrainer.load_from_checkpoint(args.resume, lr = args.lr, max_epoch = args.max_epoch)      
        
        trainer.fit(model)

        # Test with single batch, higher resolution & increase points for faster inference
        model.hparams.num_points *= model.hparams.batch_size
        model.hparams.batch_size = 1
        model.hparams.res = 256
        #trainer.test(model)
        #subprocess.run(['python','-test', args.resume.replace('checkpoints/','')], stdout=DEVNULL, stderr=DEVNULL)#, capture_output=True)#,    ])
    


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning) 
    _args = arguments.parse_arguments()
    train_scene_net(_args)