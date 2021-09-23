from data_processing.data_processing import reproject_image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
)

from torchvision.models.resnet import resnext50_32x4d

from util import arguments
from util.visualize import visualize_sdf, make_col_mesh

from pytorch_lightning import LightningModule

import math
import trimesh
import numpy as np



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
            self.feature_extractor = Conv3dFeatureExtractor(16,32,64,128)
            feature_size = point_embedding_size + 128
        
        if self.hparams.encoder == 'conv2d_pretrained': 
            self.feature_extractor = PreTrainedFeatureExtractor2D()
            feature_size = point_embedding_size + 256

        if self.hparams.encoder == 'ifnet':
            self.feature_extractor = IFNetFeatureExtractor(32, 64, 128, 128)
            feature_size = (1 + 64 + 128 + 128) * 7

        if self.hparams.encoder == 'conv2d_pretrained_projective':
            self.feature_extractor = PreTrainedFeatureExtractor2D_projective(self.freeze_pretrained)
             
            feature_size = 256 + point_embedding_size

        if self.hparams.encoder == 'hybrid':
            self.feature_extractor_2d = PreTrainedFeatureExtractor2D_projective(self.freeze_pretrained)
            self.feature_extractor_3d = Conv3dFeatureExtractor(16,32,64,128)
            feature_size = point_embedding_size + 128 + 256

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
        layers = [self.lin1, self.lin2, self.lin5, self.lin7, self.lin8, self.lin9, self.lin_sdf, self.lin_rgb]
        for layer in layers:
            torch.nn.utils.weight_norm(layer)
        # Linear: 
        # Input(N, *, H_in) - H_in: input features.
        # Output(N, *, H_out)
        self.actvn = nn.ReLU()
        # consider using nn.ReLU6()

    def forward(self, batch):
        
        voxels = batch['voxels']
        image = batch['image']

        #This should rotate points to view
        #R, T = batch['camera'][...,:3,:3], batch['camera'][...,:3,3]
        #points = batch['points'].clone().detach()
        #batch['points'] = rotate_world_to_view(points, R, T)

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
        
        elif self.hparams.encoder == 'hybrid':
            features_2d = self.feature_extractor_2d(batch['points'], image, batch['camera'])  #(B, num_points, features) 
            points = self.embedding(batch['points'])
            features_3d = self.feature_extractor_3d(voxels)
            features_3d = features_3d.transpose(-1,-2)
            features_3d = features_3d.expand(-1, points.shape[1], -1)
            features = torch.cat((features_2d, features_3d, points), axis=-1) #(bs, num_points, features_2d(256) + features_3d(128) + 2*embedding_size)

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
        out = self.actvn(self.lin2(out))
        #out = self.actvn(self.lin3(out))
        #out = self.actvn(self.lin4(out))
        out = torch.cat((out, features), axis=-1)
        out = self.actvn(self.lin5(out))
        #out = self.actvn(self.lin6(out))
        out = self.actvn(self.lin7(out))
        out = self.actvn(self.lin8(out))
        sdf = self.lin_sdf(out)
        
        if self.hparams.fieldtype == 'sdf':
            sdf = nn.Tanh()(sdf)

        #rgb = self.embedding_rgb(torch.cat((batch['points'], sdf), axis=-1))
        rgb = self.embedding_sdf(sdf)
        rgb = torch.cat((features, rgb), axis=-1)
        rgb = self.actvn(self.lin9(rgb))
        rgb = nn.Sigmoid()(self.lin_rgb(out).squeeze())

        return sdf.squeeze(-1), rgb

    def init_camera(self, device=None):
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

        return cameras
        
class Conv3dFeatureExtractor(LightningModule):
    def __init__(self, f1, f2, f3, f4):
        super(Conv3dFeatureExtractor, self).__init__()

        self.conv1 = nn.Conv3d(1, f1, 3, padding=1)
        self.conv1_1 = nn.Conv3d(f1, f1, 3, padding=1)
        self.conv2 = nn.Conv3d(f1, f2, 3, padding=1)
        self.conv2_2 = nn.Conv3d(f2, f2, 3, padding=1)
        self.conv3 = nn.Conv3d(f2, f3, 3, padding=1)
        self.conv3_3 = nn.Conv3d(f3, f3, 3, padding=1)
        self.conv4 = nn.Conv3d(f3, f4, 3, padding=1)
        self.conv4_4 = nn.Conv3d(f4, f4, 3, padding=1)

        #self.pool = nn.MaxPool3d(2)
        self.pool = nn.AvgPool3d(2)
        self.actvn = nn.ReLU()
        self.actvn_out = nn.Linear(8*8*8, 1)

        layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.actvn_out]
        for layer in layers:
            torch.nn.utils.weight_norm(layer)

        """
        self.conv1_bn = nn.BatchNorm3d(f1)
        self.conv2_bn = nn.BatchNorm3d(f2)
        self.conv3_bn = nn.BatchNorm3d(f3)
        self.conv4_bn = nn.BatchNorm3d(f4)
        """
    
    def forward(self, voxels):
        """
        out = self.maxpool(self.conv1_bn(self.actvn(self.conv1(voxels)))) #in 64³, out:32³
        out = self.maxpool(self.conv2_bn(self.actvn(self.conv2(out)))) #out 16³
        out = self.maxpool(self.conv3_bn(self.actvn(self.conv3(out)))) #out 8³
        out = self.maxpool(self.conv4_bn(self.actvn(self.conv4(out)))) #out 4³ x 128 = 64² features
        
        out = self.actvn(self.conv1(voxels)) #in 64³, out:32³
        out = self.pool((self.actvn(self.conv1_1(out))))

        out = self.actvn(self.conv2(out)) #out 16³
        out = self.pool((self.actvn(self.conv2_2(out))))

        out = self.actvn(self.conv3(out)) #out 8³
        out = self.pool((self.actvn(self.conv3_3(out))))

        out = self.actvn(self.conv4(out)) #out 4³ x 128 = 64² features
        out = self.pool((self.actvn(self.conv4_4(out))))
        """
        out = self.pool((self.actvn(self.conv1(voxels))))
        out = self.pool((self.actvn(self.conv2(out))))
        out = self.pool((self.actvn(self.conv3(out))))
        out = self.pool((self.actvn(self.conv4(out))))

        out = torch.flatten(out, start_dim=2)
        out = self.actvn_out(out) #-> 64 features
        return out


class Conv3d_multiscale_FeatureExtractor(LightningModule):
    def __init__(self, f1, f2, f3, f4): #(16,32,64,128)
        super(Conv3dFeatureExtractor, self).__init__()

        self.conv1 = nn.Conv3d(1, f1, 3, padding=1)
        self.conv2 = nn.Conv3d(f1, f2, 3, padding=1)
        self.conv3 = nn.Conv3d(f2, f3, 3, padding=1)
        self.conv4 = nn.Conv3d(f3, f4, 3, padding=1)

        self.maxpool = nn.MaxPool3d(2)
        self.actvn = nn.ReLU()
        self.actvn_out = nn.Linear(4*4*4, 1, 1)

        self.conv1_bn = nn.BatchNorm3d(f1)
        self.conv2_bn = nn.BatchNorm3d(f2)
        self.conv3_bn = nn.BatchNorm3d(f3)
        self.conv4_bn = nn.BatchNorm3d(f4)

    
    def forward(self, points, image, voxels):
        
        out = self.maxpool(self.conv1_bn(self.actvn(self.conv1(voxels)))) #in 64³, out:32³
        features_0 = out #(32³ x 16)
        out = self.maxpool(self.conv2_bn(self.actvn(self.conv2(out)))) #out 16³
        features_1 = out #(16³ x 32)
        out = self.maxpool(self.conv3_bn(self.actvn(self.conv3(out)))) #out 8³
        features_2 = out #(8³ x 64)
        out = self.maxpool(self.conv4_bn(self.actvn(self.conv4(out)))) #out 4³ x 128 = 64² features
        features_3 = out #(4³ x 128)

        out = torch.flatten(out, start_dim=2)
        out = self.actvn_out(out) #-> 64 features
        return out         


class PreTrainedFeatureExtractor2D_projective(LightningModule):

    # All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), 
    # where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using 
    # mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

    def __init__(self, freeze_pretrained=None):
        super(PreTrainedFeatureExtractor2D_projective, self, ).__init__()
        self.pretrained = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50_fast_2s1x64d', pretrained=True)
        #self.layers = torch.nn.Sequential(*(list(torch.hub.load('zhanghang1989/ResNeSt', 'resnest50_fast_2s1x64d', pretrained=True).children())[:-1]))
        
        self.layer1 = torch.nn.Sequential(*(list(self.pretrained.children())[:4]))
        self.layer2 = torch.nn.Sequential(*(list(self.pretrained.children())[4:5]))
        self.layer3 = torch.nn.Sequential(*(list(self.pretrained.children())[5:6]))
        self.layer4 = torch.nn.Sequential(*(list(self.pretrained.children())[6:7]))
        self.layer5 = torch.nn.Sequential(*(list(self.pretrained.children())[7:-1]))
        self.decoder1 = torch.nn.Linear(3+64+256+512+1024+2048, 256)        
        layers = [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5]
        
        if freeze_pretrained:
            for layer in layers:
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
        feature_0 = F.grid_sample(images, projected_points) #(BS, 3, 1, 1) features: #BS,C, 1,numpoints -> 2, 256, 1, numpoints
        net = self.layer1(images)
        feature_1 = F.grid_sample(net, projected_points) #(BS, 64, 1, numpoints)
        net = self.layer2(net) 
        feature_2 = F.grid_sample(net, projected_points) #(BS, 256, 1, numpoints)
        net = self.layer3(net)
        feature_3 = F.grid_sample(net, projected_points) #(BS, 512, 1, numpoints)
        net = self.layer4(net)
        feature_4 = F.grid_sample(net, projected_points) #(BS, 1024, 1, numpoints)
        net = self.layer5(net)
        feature_5 = F.grid_sample(net.view(net.shape[0],-1, 1, 1), projected_points) #(BS, 2048, 1, numpoints)
        features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5), dim=1)
        features = features.transpose(1,-1).squeeze(2)
        features = self.decoder1(features)# (B, sample_num, features)
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
        torch.nn.utils.weight_norm(self.fc_decoder)

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
        torch.nn.utils.weight_norm(self.fc_decoder)

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
        #self.a = torch.ones_like(self._B[:,0])
        
    def forward(self, x):
        
        #with torch.no_grad():
        if self._B is None: return x
        #x = torch.cat([a * torch.sin((2.*torch.pi*x) @ self._B.T), a * torch.cos((2.*torch.pi*x) @ self._B.T)], axis=-1) / torch.linalg.norm(a)
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

    points_batch_size = 16384
    pointsf = make_3d_grid(
        (-0.5,)*3, (0.5,)*3, resolution, res_increase
    )
    p_split = torch.split(pointsf, points_batch_size)
    probe = batch
    probe['image'] = batch['image'][0].unsqueeze(0)
    probe['voxels'] = batch['voxels'][0].unsqueeze(0)
    probe['camera'] = batch['camera'][0].unsqueeze(0)
    values = []
    for pi in p_split:
        probe['points'] = pi.unsqueeze(0).to(batch['image'].device)
        with torch.no_grad():

            out, _ = network(probe)
            if network.hparams.fieldtype == 'occupancy':
                out = torch.sigmoid(out)         

        values.append(out.squeeze(0).detach().cpu())
    value = torch.cat(values, dim=0).numpy()
    value_grid = value.reshape(res_increase*resolution[0], res_increase*resolution[1], res_increase*resolution[2], 1)
    return value_grid


def evaluate_network_rgb(network, batch, vertices):

    probe = batch
    probe['image'] = batch['image'][0].unsqueeze(0)
    probe['voxels'] = batch['voxels'][0].unsqueeze(0)
    probe['camera'] = batch['camera'][0].unsqueeze(0)
    
    points = torch.from_numpy(vertices)

    points_batch_size = 16384 #55000 #num_points * batch_size
    p_split = torch.split(points, points_batch_size)
    rgbs = []
    for pi in p_split:
        with torch.no_grad():
            probe['points'] = pi.to(batch['image'].device).unsqueeze(0).to(batch['points'].dtype)
            _, out = network(probe)
            rgbs.append(out.detach().cpu())
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

#
# SAL Geometric init proposed by Atzmon et al. 2020 
def sal_init(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            std = np.sqrt(2) / np.sqrt(nn.init_calculate_correct_fan(m.weight, 'fan_out'))

            with torch.no_grad():
                m.weight.normal_(0., std)
        if hasattr(m, 'bias'):
            m.bias.data.fill_(0.0)

#last layer
def sal_init_last_layer(m):
    if hasattr(m, 'weight'):
        val = np.sqrt(np.pi) / np.sqrt(nn.init_calculate_correct_fan(m.weight, 'fan_in'))
        with torch.no_grad():
            m.weight.fill_(val)
    if hasattr(m, 'bias'):
        m.bias.data.fill_(0.0)

#### render-implicit-surface with 1 gradient step
# reproject depth map +/- epsilon
# evaluate sdf on grid around points. Worst case 7 points need to be evaluated (p, p+x, p-x, p+z, p-z, p+y, p-y)
# evaluate gradient-field via finite differences, from each point pi, take 1 step
# step: p_i_surface = p_i + sdf(p_i) * norm(grad(p_i, p_i+x ....))
# evaluate network on all points p_i_surface for RGB (and sdf for debugging)
# reproject pointcloud into image plane or render this pointcloud differentially
# l2 loss for all pixels that contain projected surface point
# for debugging: load network with checkpoint for forward passes etc.

def determine_implicit_surface(model, batch): #Could do this in forward pass already but don't have supervision
    #is cloning necessary
    probe = batch
    #if we keep track of indicies we might save reprojection and rendering, can just compare rgb vals
    # add x,y,z deviations with central difference theorem
    epsilon = 1e-6
    probe['points'] = shift_points(batch['depthpoints'], epsilon)
    sdf, _ = model(probe)
    # gradient 
    sdf_grad = finite_differences_gradient(sdf, epsilon)
    # take 1 gradient step, should land on surface
    num_points = probe['points'].shape[1]//4
    surface_points = probe['points'][:,:num_points] - sdf[:,:num_points,None] * sdf_grad
    probe['points'] = surface_points 
    #re-evaluate model for rgb and sdf on implicit surface
    sdf_surface, rgb_surface = model(probe)
    return sdf_surface, rgb_surface, sdf_grad #, subsample_indices

def finite_differences_gradient(points, epsilon = 1e-6):
    with torch.no_grad():
        #structure is points(0:num_points --> base points, num_points:2*num_points --> x_shifted points etc.)
        num_points = points.shape[1] // 4
        grad_x =(points[...,:num_points] - points[...,num_points:2*num_points]) / epsilon
        grad_y =(points[...,:num_points] - points[...,2*num_points:3*num_points]) / epsilon
        grad_z =(points[...,:num_points] - points[...,3*num_points:]) / epsilon
        gradxyz = torch.cat((grad_x.unsqueeze(-1), grad_y.unsqueeze(-1), grad_z.unsqueeze(-1)), axis=-1)
    return gradxyz

def shift_points(points, epsilon):
    with torch.no_grad():
        #structure is points(0:num_points --> base points, num_points:2*num_points --> x_shifted points etc.)
        x_shifted = points + torch.tensor([epsilon,0,0], device=points.device)[None,None,:]
        y_shifted = points + torch.tensor([0,epsilon,0], device=points.device)[None,None,:]
        z_shifted = points + torch.tensor([0,0,epsilon], device=points.device)[None,None,:]
        shifted_points = torch.cat((points, x_shifted, y_shifted, z_shifted), axis=1)
    return shifted_points

