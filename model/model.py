import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.autograd import grad
"""
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
)"""

from util import arguments
from util.visualize import visualize_sdf, make_col_mesh

from pytorch_lightning import LightningModule

import math
import trimesh

from torch.nn.init import _calculate_correct_fan
import numpy as np
#from batchrenorm import BatchRenorm1d


args = arguments.parse_arguments()

## Highlevel: 
## Some (pretrained, multiscale) 2D feature extraction on images
## Some MLP conditioned on (coordinate + multiscale-, possibly projected- latent code) generates (SDF/UDF, Color)
## How loss? Want to regularize SDF via Sitzmann/Lipman [Normals, Eikonal, Zero surface]
## Try regress an image via mlp first?  
## [Image --> CNN --> Multiscale features --> [F_i , x_i] --> MLP --> RGB --> (3x224x224) image --> L2 loss on Minibatch of pixels]

class SimpleNetwork(LightningModule):

    def __init__(self, hparams, hidden_dim=512):
        super(SimpleNetwork, self).__init__()
        self.save_hyperparameters(hparams)
        self.freeze_pretrained = hparams.freeze_pretrained

        self.embedding = GaussFourierEmbedding(3, self.hparams.coordinate_embedding_size[0], self.hparams.coordinate_embedding_scale[0])
        self.embedding_sdf = GaussFourierEmbedding(1, int(self.hparams.coordinate_embedding_size[1]), self.hparams.coordinate_embedding_scale[1])

        if (self.hparams.coordinate_embedding_size[0] > 0):  
            point_embedding_size = int(2*self.hparams.coordinate_embedding_size[0])
        if (self.hparams.coordinate_embedding_size[1] > 0):
            sdf_embedding = int(2*self.hparams.coordinate_embedding_size[1])
        if self.hparams.coordinate_embedding_size[0] == 0: 
            point_embedding_size = 3
        if self.hparams.coordinate_embedding_size[1] == 0: 
            sdf_embedding = 1

        if self.hparams.encoder == 'conv3d': 
            self.feature_extractor = Conv3dFeatureExtractor(hparams, 16,32,64,128)
            feature_size = point_embedding_size + 128
        
        if self.hparams.encoder == 'conv2d_pretrained': 
            self.feature_extractor = PreTrainedFeatureExtractor2D()
            feature_size = point_embedding_size + 256

        if self.hparams.encoder == 'ifnet':
            self.feature_extractor = IFNetFeatureExtractor(32, 64, 128, 128)
            feature_size = (1 + 64 + 128 + 128) * 7

        if self.hparams.encoder == 'conv2d_pretrained_projective':
            self.feature_extractor = PreTrainedFeatureExtractor2D_projective(self.freeze_pretrained)
             
            feature_size = 512 + point_embedding_size

        if self.hparams.encoder == 'hybrid':
            self.feature_extractor_2d = PreTrainedFeatureExtractor2D_projective(self.freeze_pretrained)
            self.feature_extractor_3d = Conv3dFeatureExtractor(hparams, 16,32,64,128)
            feature_size = point_embedding_size + 128 + 256
        
        if self.hparams.encoder == 'hybrid_surface':
            self.feature_extractor = PreTrainedFeatureExtractor2D_projective(self.freeze_pretrained)             
            feature_size = 512 + point_embedding_size + 128 #sampled points embedded, surface points embedded, rgb
            self.pointnet = SimplePointnet()
            #self.surfaceFC1 = nn.Linear(point_embedding_size + 3, 256)
            #self.surfaceFC2 = nn.Linear(256, 128)
            #for layer in [self.surfaceFC1, self.surfaceFC1]:
            #    init_weights_relu(layer)
            #    layer = torch.nn.utils.weight_norm(layer)


        if self.hparams.encoder == 'hybrid_depthproject':
            self.feature_extractor_2d = PreTrainedFeatureExtractor2D_projective(self.freeze_pretrained)
            self.feature_extractor_3d = Conv3d_multiscale_FeatureExtractor(hparams, 64,128,128,128)
            if 'colored_density' in self.hparams.voxel_type: f0_channels = 4
            else: f0_channels = 1
            feature_size = point_embedding_size + 124 + 256 + f0_channels #256 local 2d + 256 global 2d + 124 local 3d + 64global3d + 3col + embedding
            

        if self.hparams.encoder == 'hybrid_ifnet':
            self.feature_extractor_2d = PreTrainedFeatureExtractor2D_projective(self.freeze_pretrained)
            self.feature_extractor_3d = IFNetFeatureExtractor(32, 64, 128, 128)
            feature_size = point_embedding_size + (1 + 64 + 128 + 128) + 256

        #3,4,6 added
        self.lin1 = nn.Linear(feature_size, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        #self.lin3 = nn.Linear(hidden_dim , hidden_dim)
        #self.lin4 = nn.Linear(hidden_dim, hidden_dim)
        self.lin5 = nn.Linear(feature_size + hidden_dim, hidden_dim)
        #self.lin6 = nn.Linear(hidden_dim, hidden_dim)
        self.lin7 = nn.Linear(hidden_dim, hidden_dim)
        self.lin8 = nn.Linear(hidden_dim, hidden_dim)
        self.lin9 = nn.Linear(feature_size + sdf_embedding, hidden_dim)
        self.lin_sdf = nn.Linear(hidden_dim, 1)
        self.lin_rgb = nn.Linear(hidden_dim, 3)
        #layers = [self.lin1, self.lin2, self.lin3, self.lin4, self.lin5, self.lin6, self.lin7, self.lin8, self.lin9, self.lin_sdf, self.lin_rgb]
        #layers = [self.lin1, self.lin2, self.lin3, self.lin4, self.lin5, self.lin6, self.lin7, self.lin8, self.lin9]
        layers = [self.lin1, self.lin2, self.lin5, self.lin7, self.lin8, self.lin9]
        #layers = [self.lin1, self.lin2, self.lin5, self.lin7, self.lin8, self.lin9, self.lin_sdf, self.lin_rgb]
        
        for layer in layers:
            #pass
            #
            #sal_init(layer)
            init_weights_relu(layer)
            layer = torch.nn.utils.weight_norm(layer)
            
        #sal_init_last_layer(self.lin_sdf)
        init_weights_symmetric(self.lin_sdf)
        self.lin_sdf = torch.nn.utils.weight_norm(self.lin_sdf)
        init_weights_symmetric(self.lin_rgb)
        self.lin_rgb = torch.nn.utils.weight_norm(self.lin_rgb)
        """self.BN1 = torch.nn.BatchNorm1d(hidden_dim)
        self.BN2 = torch.nn.BatchNorm1d(hidden_dim)
        self.BN5 = torch.nn.BatchNorm1d(hidden_dim)
        self.BN7 = torch.nn.BatchNorm1d(hidden_dim)
        self.BN8 = torch.nn.BatchNorm1d(hidden_dim)
        self.BN9 = torch.nn.BatchNorm1d(hidden_dim)"""

        """self.BN1 = torch.nn.LayerNorm(hidden_dim)
        self.BN2 = torch.nn.LayerNorm(hidden_dim)
        self.BN5 = torch.nn.LayerNorm(hidden_dim)
        self.BN7 = torch.nn.LayerNorm(hidden_dim)
        self.BN8 = torch.nn.LayerNorm(hidden_dim)
        self.BN9 = torch.nn.LayerNorm(hidden_dim)"""

        """self.BN1 = BatchRenorm1d(hidden_dim)
        self.BN2 = BatchRenorm1d(hidden_dim)
        self.BN5 = BatchRenorm1d(hidden_dim)
        self.BN7 = BatchRenorm1d(hidden_dim)
        self.BN8 = BatchRenorm1d(hidden_dim)
        self.BN9 = BatchRenorm1d(hidden_dim)"""
        
        # Linear: 
        # Input(N, *, H_in) - H_in: input features.
        # Output(N, *, H_out)
        self.actvn = nn.ReLU()
        # consider using nn.ReLU6()

    def forward(self, batch):
        
        voxels = batch['voxels']
        image = batch['image']

        if self.hparams.encoder == 'ifnet':
            features = self.feature_extractor(batch['points'], voxels) # features: (B, features, 1,7,sample_num) 
            
            shape = features.shape
            features = torch.reshape(features, 
                                    (shape[0], shape[1] * shape[3], shape[4]))
            features = features.transpose(-1,-2) # features: (B, num_points, features*7)

        elif self.hparams.encoder == 'conv2d_pretrained_projective':
            features = self.feature_extractor(batch['points'], image, batch['camera'])  #(B, num_points, features) 
            points = self.embedding(batch['points'])
            features = torch.cat((features, points), axis=-1) #(bs, num_points, features + 2*embedding_size)

        elif self.hparams.encoder == 'hybrid_surface':
            features = self.feature_extractor(batch['points'], image, batch['camera'])  #(B, num_points, features) 
            points = self.embedding(batch['points'])
            surface_points = self.embedding(batch['voxels'][...,:3]) #pseudovoxels. These are actually surface points
            surface_rgb = batch['voxels'][...,3:]
            surface_features = torch.cat((surface_points, surface_rgb), axis=-1)
            surface_features = self.pointnet(surface_features)
            surface_features = surface_features.unsqueeze(1)
            #surface_features = surface_features.transpose(-1,-2)
            surface_features = surface_features.expand(points.shape[0], points.shape[1], -1)
            features = torch.cat((features, points, surface_features), axis=-1) #(bs, num_points, features + 2*embedding_size)
        
        elif self.hparams.encoder == 'hybrid':
            features_2d = self.feature_extractor_2d(batch['points'], image, batch['camera'])  #(B, num_points, features) 
            points = self.embedding(batch['points'])
            features_3d = self.feature_extractor_3d(voxels)
            features_3d = features_3d.transpose(-1,-2)
            features_3d = features_3d.expand(-1, points.shape[1], -1)
            features = torch.cat((features_2d, features_3d, points), axis=-1) #(bs, num_points, features_2d(256) + features_3d(128) + 2*embedding_size)

        elif self.hparams.encoder == 'hybrid_depthproject':
            features_2d = self.feature_extractor_2d(batch['points'], image, batch['camera'])  #(B, num_points, features)
            features_3d = self.feature_extractor_3d(batch['points'], voxels) 
            points = self.embedding(batch['points'])
            #print(features_2d.shape, features_3d.shape, points.shape)            
            features = torch.cat((features_2d, features_3d, points), axis=-1) #(bs, num_points, features_2d(256) + features_3d(128) + 2*embedding_size)
            #print(features.shape)

        elif self.hparams.encoder == 'hybrid_ifnet':
            features_2d = self.feature_extractor_2d(batch['points'], image, batch['camera'])  #(B, num_points, features)

            features_3d = self.feature_extractor_3d(batch['points'], voxels) # features: (B, features, 1,7,sample_num) 
            
            shape = features_3d.shape
            features_3d = torch.reshape(features_3d, 
                                    (shape[0], shape[1] * shape[3], shape[4]))
            features_3d = features_3d.transpose(-1,-2) # features: (B, num_points, features*7)

            points = self.embedding(batch['points'])

            #features_3d = features_3d.transpose(-1,-2)
            #features_3d = features_3d.expand(-1, points.shape[1], -1)

            features = torch.cat((features_2d, features_3d, points), axis=-1) #(bs, num_points, features_2d(256) + features_3d(128) + 2*embedding_size)

        else:
            points = self.embedding(batch['points'])
            features = self.feature_extractor(points, image, voxels) 
            features = features.transpose(-1,-2) 
            features = features.expand(-1, points.shape[1], -1) #(bs, num_points, 64)
            features = torch.cat((features, points), axis=-1) #(bs, 14*14+3, num_points)
        
        out = self.actvn(self.lin1(features))
        #out = self.BN1(out)
        out = self.actvn(self.lin2(out))
        #out = self.BN2(out)
        #out = self.actvn(self.lin3(out))
        #out = self.actvn(self.lin4(out))
        out = torch.cat((out, features), axis=-1)        
        out = self.actvn(self.lin5(out))
        #out = self.BN5(out)
        #out = self.actvn(self.lin6(out))
        #out = self.BN6(out)
        out = self.actvn(self.lin7(out))
        #out = self.BN7(out)
        out = self.actvn(self.lin8(out))
        #out = self.BN8(out)
        sdf = self.lin_sdf(out)
        
        if self.hparams.fieldtype == 'sdf':
            sdf = nn.Tanh()(sdf)

        #rgb = self.embedding_rgb(torch.cat((batch['points'], sdf), axis=-1))
        rgb = self.embedding_sdf(sdf)
        rgb = torch.cat((features, rgb), axis=-1)
        rgb = self.actvn(self.lin9(rgb))
        #rgb = self.BN9(out)
        rgb = nn.Sigmoid()(self.lin_rgb(out).squeeze(-1))

        return sdf.squeeze(-1), rgb

    """def init_camera(self, device=None):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        
        batchsize = 12
                      
        lastpt = 360 - 360/batchsize - 180
        azim = torch.linspace(-180, lastpt, batchsize)
        R, T = look_at_view_transform(1.2, 20, azim)

        cameras = [FoVPerspectiveCameras(device=device, R=R[i].unsqueeze(0), T=T[i].unsqueeze(0)) for i in range(batchsize)]

        return cameras"""
        
class Conv3dFeatureExtractor(LightningModule):
    def __init__(self, hparams, f1, f2, f3, f4):
        super(Conv3dFeatureExtractor, self).__init__()
        if hparams.voxel_type == 'colored_density': f0 = 4
        else: f0 = 1
        self.num_voxels = hparams.num_voxels
        channels_out = self.num_voxels // (2**4)
        if self.num_voxels > 32:
            channels_out = self.num_voxels // (2**4)
            self.conv1 = nn.Conv3d(f0, f1, 3, padding=1)
            self.conv1_1 = nn.Conv3d(f1, f1, 3, padding=1)
            self.conv2 = nn.Conv3d(f1, f2, 3, padding=1)
            self.conv2_2 = nn.Conv3d(f2, f2, 3, padding=1)
            self.conv3 = nn.Conv3d(f2, f3, 3, padding=1)
            self.conv3_3 = nn.Conv3d(f3, f3, 3, padding=1)
            self.conv4 = nn.Conv3d(f3, f4, 3, padding=1)
            self.conv4_4 = nn.Conv3d(f4, f4, 3, padding=1)
        else:
            channels_out = self.num_voxels // (2**3)
            self.conv1 = nn.Conv3d(f0, f2, 3, padding=1)
            self.conv1_1 = nn.Conv3d(f2, f2, 3, padding=1)
            self.conv2 = nn.Conv3d(f2, f3, 3, padding=1)
            self.conv2_2 = nn.Conv3d(f3, f3, 3, padding=1)
            self.conv3 = nn.Conv3d(f3, f4, 3, padding=1)
            self.conv3_3 = nn.Conv3d(f4, f4, 3, padding=1)
            self.conv4 = nn.Conv3d(f3, f4, 3, padding=1)
            self.conv4_4 = nn.Conv3d(f4, f4, 3, padding=1)

        #self.pool = nn.MaxPool3d(2)
        self.pool = nn.AvgPool3d(2)
        self.actvn = nn.ReLU()
        self.actvn_out = nn.Linear(channels_out**3, 1)

        #layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.actvn_out]
        #for layer in layers:
            #torch.nn.utils.weight_norm(layer)
    
    def forward(self, voxels):
        out = self.pool((self.actvn(self.conv1(voxels))))
        out = self.pool((self.actvn(self.conv2(out))))
        out = self.pool((self.actvn(self.conv3(out))))
        if self.num_voxels > 32:
            out = self.pool((self.actvn(self.conv4(out))))

        out = torch.flatten(out, start_dim=2)

        out = self.actvn_out(out) #-> 64 features
        return out


class Conv3d_multiscale_FeatureExtractor(LightningModule):
    def __init__(self, hparams, f1, f2, f3, f4):
        super(Conv3d_multiscale_FeatureExtractor, self).__init__()
        if hparams.voxel_type == 'colored_density': f0 = 4
        else: f0 = 1
        self.conv1 = nn.Conv3d(f0, f1, 3, padding=1)
        self.conv1_1 = nn.Conv3d(f1, f1, 3, padding=1)
        self.conv2 = nn.Conv3d(f1, f2, 3, padding=1)
        self.conv2_2 = nn.Conv3d(f2, f2, 3, padding=1)
        self.conv3 = nn.Conv3d(f2, f3, 3, padding=1)
        self.conv3_3 = nn.Conv3d(f3, f3, 3, padding=1)
        #self.conv4 = nn.Conv3d(f3, f4, 3, padding=1)
        #self.conv4_4 = nn.Conv3d(f4, f4, 3, padding=1)
        #self.global_out = nn.Conv3d(f3, 1, 3, padding=1)

        self.reduce_1 = torch.nn.Conv1d(f1, 24, 1)
        self.reduce_2 = torch.nn.Conv1d(f2, 36, 1)
        self.reduce_3 = torch.nn.Conv1d(f3, 64, 1)

        self.BN1 = torch.nn.BatchNorm3d(f1)
        self.BN2 = torch.nn.BatchNorm3d(f2)
        self.BN3 = torch.nn.BatchNorm3d(f3)
        self.BN4 = torch.nn.BatchNorm3d(f4)
        

        self.pool = nn.MaxPool3d(2)
        #self.pool = nn.AvgPool3d(2)
        self.actvn = nn.ReLU()

        layers = [self.conv1, self.conv1_1, self.conv2, self.conv2_2, self.conv3, self.conv3_3, self.reduce_1, self.reduce_2, self.reduce_3] #, self.reduce_4
        for layer in layers:
            init_weights_relu(layer)
            layer = torch.nn.utils.weight_norm(layer)

    
    def forward(self, points, voxels): #vox in (say 32) f_n --> 4,16,32,64,128 --> 225 or 228 total
        p = torch.zeros_like(points)
        p[:, :, 0], p[:, :, 1], p[:, :, 2] = [2 * points[:, :, 2], 2 * points[:, :, 1], 2 * points[:, :, 0]]
        p = p.unsqueeze(1).unsqueeze(1)
        
        feature_0 = F.grid_sample(voxels, p, align_corners=True)  # out : (B,C (of x), 1,1,sample_num) // f_0 -> 4
        out = self.actvn(self.conv1(voxels))
        out = self.BN1(self.actvn(self.conv1_1(out)))
        feature_1 = F.grid_sample(out, p, align_corners=True) # f1 : 16
        
        out = self.pool(out)
        out = self.actvn(self.conv2(out))
        out = self.BN2(self.actvn(self.conv2_2(out)))
        feature_2 = F.grid_sample(out, p, align_corners=True) # f2 : 32

        out = self.pool(out)
        out = self.actvn(self.conv3(out))
        out = self.BN3(self.actvn(self.conv3_3(out)))
        feature_3 = F.grid_sample(out, p, align_corners=True) # f3 : 64

        #out = self.pool(out)
        #out = self.actvn(self.conv3(out))
        #out = self.BN4(self.actvn(self.conv4_4(out)))

        #out = self.pool((self.actvn(self.conv4(out))))
        #feature_4 = F.grid_sample(out, p, align_corners=True) # f4 : 128
        # here every channel corresponds to one feature.
        
        feature_0 = feature_0.squeeze(-2).squeeze(-2) #4
        feature_1 = feature_1.squeeze(-2).squeeze(-2)
        feature_1 = self.actvn(self.reduce_1(feature_1)) #8
        feature_2 = feature_2.squeeze(-2).squeeze(-2)
        feature_2 = self.actvn(self.reduce_2(feature_2)) #16
        feature_3 = feature_3.squeeze(-2).squeeze(-2)
        feature_3 = self.actvn(self.reduce_3(feature_3)) #32

        #feature_4 = feature_4.squeeze(-2).squeeze(-2)
        #feature_4 = self.reduce_4(feature_4) #68
        #--> 128 features per point

        #out = self.actvn(self.global_out(out)) #B, 1, 4,4,4
        #out = out.reshape(-1, 64, 1)
        #out = out.expand(feature_0.shape[0], 64, feature_0.shape[-1])
        
        features = torch.cat((feature_0, feature_1, feature_2, feature_3), dim=1)  # (B, features, 1,1,sample_num) #, feature_4
        #shape = features.shape
        #features = torch.reshape(features, 
        #                        (shape[0], shape[1] * shape[3], shape[4]))
        features = features.transpose(-1,-2) # features: (B, num_points, features)
        return features

class PreTrainedFeatureExtractor2D_projective(LightningModule):

    # All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), 
    # where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using 
    # mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

    def __init__(self, freeze_pretrained=None):
        super(PreTrainedFeatureExtractor2D_projective, self, ).__init__()
        model = 'resnet'
        if model == 'restnest':
            print('model is resneSt')
            self.pretrained = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50_fast_2s1x64d', pretrained=True)
            #self.layers = torch.nn.Sequential(*(list(torch.hub.load('zhanghang1989/ResNeSt', 'resnest50_fast_2s1x64d', pretrained=True).children())[:-1]))
            
            self.layer1 = torch.nn.Sequential(*(list(self.pretrained.children())[:4]))
            self.layer2 = torch.nn.Sequential(*(list(self.pretrained.children())[4:5]))
            self.layer3 = torch.nn.Sequential(*(list(self.pretrained.children())[5:6]))
            self.layer4 = torch.nn.Sequential(*(list(self.pretrained.children())[6:7]))
            self.layer5 = torch.nn.Sequential(*(list(self.pretrained.children())[7:-1]))
            #self.decoder1 = torch.nn.Linear(3+64+256+512+1024+2048, 256)        
            layers = [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5]
            #target is 3+8+16+32+64+128 = 251
            #5 extra conv1ds are needed
            # reshape 
            # In N,C,L // OUT N C L
            # N, C, numpoints
            self.reduce_1 = torch.nn.Conv1d(64, 8, 1)
            self.reduce_2 = torch.nn.Conv1d(256, 16, 1)
            self.reduce_3 = torch.nn.Conv1d(512, 32, 1)
            self.reduce_4 = torch.nn.Conv1d(1024, 64, 1)
            self.reduce_5 = torch.nn.Conv1d(2048, 133, 1) #133, 64, 32, 16, 8
            self.global_feat_out = torch.nn.Conv1d(2048, 256, 1)
        else:
            print('model is resnet50')
            self.pretrained = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)            
            self.layer1 = torch.nn.Sequential(*(list(self.pretrained.children())[:4]))
            self.layer2 = torch.nn.Sequential(*(list(self.pretrained.children())[4:5]))
            self.layer3 = torch.nn.Sequential(*(list(self.pretrained.children())[5:6]))
            self.layer4 = torch.nn.Sequential(*(list(self.pretrained.children())[6:7]))
            self.layer5 = torch.nn.Sequential(*(list(self.pretrained.children())[7:-1]))
            layers = [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5]
            #target is 3+8+16+32+64+128 = 251
            #5 extra conv1ds are needed
            # reshape 
            # In N,C,L // OUT N C L
            # N, C, numpoints
            self.reduce_1 = torch.nn.Conv1d(64, 8, 1)
            self.reduce_2 = torch.nn.Conv1d(256, 16, 1)
            self.reduce_3 = torch.nn.Conv1d(512, 32, 1)
            self.reduce_4 = torch.nn.Conv1d(1024, 64, 1)
            self.reduce_5 = torch.nn.Conv1d(2048, 133, 1) #133, 64, 32, 16, 8
            self.global_feat_out = torch.nn.Linear(2048, 256)
        
        self.BN1 = torch.nn.BatchNorm2d(64)
        self.BN2 = torch.nn.BatchNorm2d(256)
        self.BN3 = torch.nn.BatchNorm2d(512)
        self.BN4 = torch.nn.BatchNorm2d(1024)
        self.BN5 = torch.nn.BatchNorm2d(2048)
        self.actvn = torch.nn.ReLU()

        
        feat_layers = [self.reduce_1,  self.reduce_2,  self.reduce_3,  self.reduce_4,  self.reduce_5, self.global_feat_out]
        for layer in feat_layers:
            init_weights_relu(layer)
            #layer = torch.nn.utils.weight_norm(layer)

        if freeze_pretrained:
            print(f'freezing layers up to layer: {freeze_pretrained}')
            for layer in layers[freeze_pretrained:]:
                #don't unfreeze BatchNorm
                if isinstance(layer, nn.BatchNorm2d):
                    for child in layer.children():
                        for param in child.parameters():
                            param.requires_grad = False

            #only retrain highlevel representations
            for layer in layers[freeze_pretrained:]:
                for child in layer.children():
                    for param in child.parameters():
                        param.requires_grad = False
            

    def forward(self, points, images, camera):

        R, T = camera[...,:3,:3], camera[...,:3,3]
        points = rotate_world_to_view(points, R, T)
        projected_points = project_3d_to_2d_gridsample(points, 245, 112, 112) #could load/use an intrinsics file here

        projected_points = projected_points.unsqueeze(1) #points are now x,y within (-1, 1)
        feature_0 = F.grid_sample(images, projected_points) #(BS, 3, 1, 1) features: #BS,C, 1,numpoints -> 2, 3, 1, numpoints
        
        net = self.BN1(self.layer1(images))
        feature_1 = F.grid_sample(net, projected_points, align_corners=True) #(BS, 64, 1, numpoints)
        net = self.BN2(self.layer2(net)) 
        feature_2 = F.grid_sample(net, projected_points, align_corners=True) #(BS, 256, 1, numpoints)
        net = self.BN3(self.layer3(net))
        feature_3 = F.grid_sample(net, projected_points, align_corners=True) #(BS, 512, 1, numpoints)
        net = self.BN4(self.layer4(net))
        feature_4 = F.grid_sample(net, projected_points, align_corners=True) #(BS, 1024, 1, numpoints)
        net = self.BN5(self.layer5(net))
        feature_5 = F.grid_sample(net, projected_points, align_corners=True) #(BS, 2048, 1, numpoints)

        #net = net.reshape(points.shape[0], 2048)
        #.view(net.shape[0],-1, 1, 1)
        #net = torch.flatten(net, start_dim=1)
        #net = self.actvn(self.global_feat_out(net))

        #net = net.unsqueeze(2)#10,256,1
        #net = net.expand(points.shape[0], -1, points.shape[1])


        #print(feature_0.shape, '\n')
        #print(feature_1.shape, '\n')
        feature_0 = feature_0.squeeze(-2)
        feature_1 = feature_1.squeeze(-2)
        feature_2 = feature_2.squeeze(-2)
        feature_3 = feature_3.squeeze(-2)
        feature_4 = feature_4.squeeze(-2)
        feature_5 = feature_5.squeeze(-2)

        feature_1 = self.reduce_1(feature_1)
        feature_2 = self.reduce_2(feature_2)
        feature_3 = self.reduce_3(feature_3)
        feature_4 = self.reduce_4(feature_4)
        feature_5 = self.reduce_5(feature_5)

        feature_1 = self.actvn(feature_1)
        feature_2 = self.actvn(feature_2)
        feature_3 = self.actvn(feature_3)
        feature_4 = self.actvn(feature_4)
        feature_5 = self.actvn(feature_5)
        #net : BS, num_points, 256
        
        #feature_1 = self.BN1(self.actvn(feature_1))
        #feature_2 = self.BN2(self.actvn(feature_2))
        #feature_3 = self.BN3(self.actvn(feature_3))
        #feature_4 = self.BN4(self.actvn(feature_4))
        #feature_5 = self.BN5(self.actvn(feature_5))


        features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5), dim=1)
        features = features.transpose(1,-1)#.squeeze(2)
        #features = self.decoder1(features)# (B, sample_num, features)
        return features #(B, sample_num, features)



class PreTrainedFeatureExtractor2D_2(LightningModule):

    # All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), 
    # where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using 
    # mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

    def __init__(self):
        super(PreTrainedFeatureExtractor2D_2, self).__init__()
        self.pretrained = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_swsl')
        self.layers = torch.nn.Sequential(*(list(self.pretrained.children())[:-1]))
        """
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = False
        """
        self.fc_decoder = torch.nn.Linear(2048, 768)
        self.fc_decoder = torch.nn.utils.weight_norm(self.fc_decoder)

    def forward(self, points, image, voxels):
        net = self.layers(image)
        net = self.fc_decoder(net.reshape(image.shape[0], 1, 2048))
        return net.transpose(-2,-1)

class PreTrainedFeatureExtractor2D(LightningModule):

    # All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), 
    # where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using 
    # mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

    def __init__(self):
        super(PreTrainedFeatureExtractor2D, self).__init__()
        self.layers = torch.nn.Sequential(*(list(torch.hub.load('zhanghang1989/ResNeSt', 'resnest50_fast_2s1x64d', pretrained=True).children())[:-1]))
        """
        for child in self.layers.children():
            for param in child.parameters():
                param.requires_grad = False
        """
        self.fc_decoder = torch.nn.Linear(2048, 256)
        self.fc_decoder = torch.nn.utils.weight_norm(self.fc_decoder)

    def forward(self, points, image, voxels):
        net = self.layers(image)
        net = self.fc_decoder(net.reshape(image.shape[0], 1, 2048))
        return net.transpose(-2,-1)


class IFNetFeatureExtractor(LightningModule):

    def __init__(self, f1, f2, f3, f4):
        super(IFNetFeatureExtractor, self).__init__()

        self.conv_1 = nn.Conv3d(1, f1, 3, padding=1)  # out: 8
        self.conv_1_1 = nn.Conv3d(f1, f2, 3, padding=1)  # out: 8
        self.conv_2 = nn.Conv3d(f2, f3, 3, padding=1)  # out: 4
        self.conv_2_1 = nn.Conv3d(f3, f4, 3, padding=1)  # out: 4
        self.conv_3 = nn.Conv3d(f4, f4, 3, padding=1)  # out: 2
        self.conv_3_1 = nn.Conv3d(f4, f4, 3, padding=1)  # out: 2
        self.actvn = nn.ReLU()
        self.maxpool = nn.MaxPool3d(2)

        self.conv1_1_bn = nn.BatchNorm3d(f2)
        self.conv2_1_bn = nn.BatchNorm3d(f4)
        self.conv3_1_bn = nn.BatchNorm3d(f4)

        displacment = 0.035
        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)

        self.displacments = torch.Tensor(displacments)

    def forward(self, points, voxels):
        p = points
        
        p= 2 * points
        p = p.unsqueeze(1).unsqueeze(1)
        #p = torch.cat([p + d for d in self.displacments.to(p.device)], dim=2)  # (B,1,7,num_samples,3)
        feature_0 = F.grid_sample(voxels, p, align_corners=True)  # out : (B,C (of x), 1,1,sample_num)

        net = self.actvn(self.conv_1(voxels))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        feature_1 = F.grid_sample(net, p, align_corners=True)  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        feature_2 = F.grid_sample(net, p, align_corners=True)
        net = self.maxpool(net)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        feature_3 = F.grid_sample(net, p, align_corners=True)

        # here every channel corresponse to one feature.

        features = torch.cat((feature_0, feature_1, feature_2, feature_3), dim=1)  # (B, features, 1,7,sample_num)
        return features

def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


##This is the simple PointNet encoder from Mescheder et al. OccNet
# Slightly adapted to allow for embedded points
class SimplePointnet(nn.Module):
    ''' PointNet-based encoder network.
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''
    #bs, num_points, 2*embedding_size
    def __init__(self, c_dim=128, dim=3+512, hidden_dim=256):
        super().__init__()
        self.c_dim = c_dim

        self.fc_0 = nn.Linear(dim, hidden_dim)
        self.fc_1 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, 128)

        self.actvn = nn.ReLU()
        self.pool = maxpool

        for layer in [self.fc_0,self.fc_1,self.fc_2, self.fc_3, self.fc_c]:
            init_weights_relu(layer)
            layer = torch.nn.utils.weight_norm(layer)
        #self.fc_c = init_weights_symmetric(self.fc_c)
        #self.fc_c = torch.nn.utils.weight_norm(self.fc_c)

    def forward(self, p):
        #batch_size, T, D = p.size()
        # output size: B x T X F
        net = self.fc_0(self.actvn(p))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_1(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_2(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_3(self.actvn(net))

        # Recude to  B x F
        net = self.pool(net, dim=1)
        c = self.fc_c(self.actvn(net))

        return c



def project_3d_to_2d_gridsample(points, f, cx, cy):
    u = -f * points[...,0]/points[...,2] / cx
    v = -f * points[...,1]/points[...,2] / cy
    return torch.stack((u,v), axis=-1)

def rotate_world_to_view(points, R, T):
    viewpoints = (R @ points.transpose(2,1)).transpose(2,1)
    viewpoints = viewpoints + T.unsqueeze(1)
    #viewpoints = (R_b2CV @ viewpoints.transpose()).transpose()
    return viewpoints


class GaussFourierEmbedding(LightningModule):
    def __init__(self, num_input_channels, mapping_size=256, scale=12):
        super().__init__()
        if mapping_size!=0:
            self._B = torch.randn((mapping_size, num_input_channels), device='cuda:0') * scale
            self.register_buffer('B', self._B)
        else: self._B = None
        
    def forward(self, x):        
        if self._B is None: return x
        x = torch.cat([torch.sin((2.*math.pi*x) @ self._B.T), torch.cos((2.*math.pi*x) @ self._B.T)], axis=-1)    
        return x



def make_3d_grid(bb_min, bb_max, shape, res_increase = args.inf_res):
    size = shape[0] * shape[1] * shape[2] * res_increase**3
    pxs = torch.linspace(bb_min[0], bb_max[0], res_increase*shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], res_increase*shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], res_increase*shape[2])

    pxs = pxs.view(-1, 1, 1).expand(*shape*res_increase).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape*res_increase).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape*res_increase).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)
    return p



def evaluate_network_on_grid(network, batch, resolution, res_increase = args.inf_res):

    points_batch_size = batch['points'].shape[1] * batch['points'].shape[0] #num_points * batch_size
    pointsf = make_3d_grid(
        (-0.5,)*3, (0.5,)*3, resolution, res_increase
    )
    p_split = torch.split(pointsf, points_batch_size)
    probe = {}
    probe['image'] = batch['image'][0].unsqueeze(0)
    probe['voxels'] = batch['voxels'][0].unsqueeze(0)
    probe['camera'] = batch['camera'][0].unsqueeze(0)
    #depth points and prepare points dummies
    values = []
    for pi in p_split:
        probe['points'] = pi.unsqueeze(0).to(batch['image'].device)
        with torch.no_grad():

            out, _ = network(probe)
            if network.hparams.fieldtype == 'occupancy':
                out = torch.sigmoid(out)
            #subselect only first n points that we used originally         

        values.append(out.squeeze(0).detach().cpu())
    value = torch.cat(values, dim=0).numpy()
    value_grid = value.reshape(res_increase*resolution[0], res_increase*resolution[1], res_increase*resolution[2], 1)
    return value_grid


def evaluate_network_rgb(network, batch, vertices):
    probe = {}
    probe['image'] = batch['image'][0].unsqueeze(0)
    probe['voxels'] = batch['voxels'][0].unsqueeze(0)
    probe['camera'] = batch['camera'][0].unsqueeze(0)
    #depth points and prepare points dummies
    
    points = torch.from_numpy(vertices)

    points_batch_size = batch['points'].shape[1] * batch['points'].shape[0] #num_points * batch_size
    p_split = torch.split(points, points_batch_size)
    rgbs = []
    for pi in p_split:
        with torch.no_grad():
            probe['points'] = pi.to(batch['image'].device).unsqueeze(0).to(batch['points'].dtype)
            _, out = network(probe)
            #subselect only first n points that we used originally  
            rgbs.append(out.squeeze(0).detach().cpu())
    rgb = torch.cat(rgbs, dim=0).numpy()
    return rgb


def implicit_to_mesh(network, batch, resolution, threshold_p, output_path, res_increase=args.inf_res):
    value_grid = evaluate_network_on_grid(network, batch, resolution, res_increase)
    #limit to n verts?  
    vertices, faces = visualize_sdf(value_grid, output_path, level=threshold_p, rgb=True)
    rgb = evaluate_network_rgb(network, batch, vertices)
    make_col_mesh(vertices, faces, rgb, output_path)
    #make mesh out of rgb, vertices, triangles

def implicit_rgb_only(network, batch, obj_path, output_path):
    
    mesh = trimesh.load(obj_path, file_type='obj',force='mesh')
    vertices = mesh.vertices
    faces = mesh.faces
    rgb = evaluate_network_rgb(network, batch, vertices)
    make_col_mesh(vertices, faces, rgb, output_path)
"""
def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0][:, -3:]
    return points_grad
"""
#
# SAL Geometric init proposed by Atzmon et al. 2020 
def sal_init(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            std = np.sqrt(2) / np.sqrt(_calculate_correct_fan(m.weight, 'fan_out'))

            with torch.no_grad():
                m.weight.normal_(0., std)
        if hasattr(m, 'bias'):
            m.bias.data.fill_(0.0)


def sal_init_last_layer(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            with torch.no_grad():
                torch.nn.init.normal_(m.weight, mean=np.sqrt(np.pi) / np.sqrt(_calculate_correct_fan(m.weight, 'fan_in')), std=0.00001)
        if hasattr(m, 'bias'):
            m.bias.data.fill_(-0.1)


def init_weights_relu(m):
    if hasattr(m, 'weight'):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    if hasattr(m, 'bias'):
        m.bias.data.fill_(0.)

def init_weights_symmetric(m):
    if hasattr(m, 'weight'):
        torch.nn.init.xavier_normal_(m.weight)
    if hasattr(m, 'bias'):
        m.bias.data.fill_(0.)