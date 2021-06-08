import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import numpy as np
import pytorch3d as pt3d

from util import arguments
from util.visualize_rgb import visualize_implicit_rgb
from util.visualize import visualize_sdf

args = arguments.parse_arguments()

## Highlevel: 
## Some (pretrained, multiscale) 2D feature extraction on images
## Some MLP conditioned on (coordinate + multiscale-, possibly projected- latent code) generates (SDF/UDF, Color)
## How loss? Want to regularize SDF via Sitzmann/Lipman [Normals, Eikonal, Zero surface]
## Try regress an image via mlp first?  
## [Image --> CNN --> Multiscale features --> [F_i , x_i] --> MLP --> RGB --> (3x224x224) image --> L2 loss on Minibatch of pixels]

class Network(nn.Module):

    def __init__(self, hidden_dim=512):
        super(Network, self).__init__()

        #MultiScale Feature Encoder
        #self.feature_extractor = PreTrainedFeatureExtractor(32, 64, 128, 128)
        self.feature_extractor = FeatureExtractor(16, 32, 64, 128)
        feature_size = (1 + 3 + 32 + 64 + 128)

        self.fc_0 = nn.Conv1d(feature_size, hidden_dim, 1)
        self.fc_1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_3 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_4 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_5 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        #self.fc_out_rgb = nn.Conv1d(hidden_dim, 3, 1)
        self.fc_out_sdf = nn.Conv1d(hidden_dim, 1, 1)
        
        self.actvn = nn.ReLU()
        #self.actvn_out = nn.Sigmoid()

    def forward(self, batch):

        features = self.feature_extractor(batch)
        shape = features.shape

        features = torch.reshape(features,(shape[0], shape[1], shape[3]))

        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        net = self.actvn(self.fc_3(net))
        net = self.actvn(self.fc_4(net))
        net = self.actvn(self.fc_5(net))

        #net_rgb = self.fc_out_rgb(net)
        net_sdf = self.fc_out_sdf(net)
        #print('net rgb,sdf shape: ',net_rgb.shape, '\n', net_sdf.shape)

        #out = torch.cat([net_sdf,net_rgb],axis=1)
        #out = out.squeeze(1).transpose(2,1) # --> (BS, num_points, channels)
        
        out = net_sdf
        #print('out_shape:',out.shape)

        return out

class SimpleImplicitScene(nn.Module):

    def __init__(self, coord_embedding=256, hidden_dim=128):
        super(SimpleImplicitScene, self).__init__()

        self.conv_1 = nn.Conv2d(coord_embedding, hidden_dim, kernel_size=1, padding=0)
        self.conv_2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, padding=0)
        self.conv_3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, padding=0)
        self.conv_out = nn.Conv2d(hidden_dim, 3, kernel_size=1, padding=0)
        self.actvn = nn.ReLU()
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.actvn_out = nn.Sigmoid()

    def forward(self, batch):
        net = self.bn(self.actvn(self.conv_1(batch['points'])))
        net = self.bn(self.actvn(self.conv_2(net)))
        net = self.bn(self.actvn(self.conv_3(net)))
        out = self.actvn_out(self.conv_out(net))
        return out

class SimpleNetwork(nn.Module):

    def __init__(self, hparams, hidden_dim=512):
        super(SimpleNetwork, self).__init__()
        self.hparams = hparams
        ## clamp logic & actvn
        self.feature_extractor = NonProjectiveFeatureExtractor(16, 32, 64, 128, 256)
        feature_size = (3)

        self.fc_0 = nn.Conv1d(feature_size, hidden_dim, 1)
        self.fc_1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_3 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_4 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_5 = nn.Conv1d(hidden_dim, hidden_dim, 1)

        self.lin1 = nn.Linear(feature_size, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, hidden_dim)
        self.lin4 = nn.Linear(hidden_dim, hidden_dim)
        self.lin5 = nn.Linear(hidden_dim, 1)
        #self.fc_out_rgb = nn.Conv1d(hidden_dim, 3, 1)
        self.fc_out_sdf = nn.Conv1d(hidden_dim, 1, 1)

        self.actvn = nn.ReLU()
        # consider using nn.ReLU6()
        
        if self.hparams.clamp == 0:
            self.actvn_out = nn.Sigmoid()
        elif self.hparams.clamp > 0:
            self.actvn_out = nn.Tanh()
        else:
            self.actvn_out = nn.Sigmoid()

    def forward(self, batch):

        features = self.feature_extractor(batch) #(bs, 1, 28*28)
        features = features.transpose(2,1) #(bs,28*28,1)
        features = features.expand((-1, -1, batch['points'].shape[1])) #(bs, num_points, 28*28)
        points = batch['points']

        """features = torch.cat((features, points), axis=1) #(bs, 28*28+3, num_points)

        net = self.actvn(self.fc_0(points))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        net = self.actvn(self.fc_3(net))
        net = self.actvn(self.fc_4(net))
        net = self.actvn(self.fc_5(net))

        #net_rgb = self.fc_out_rgb(net)
        net_sdf = self.fc_out_sdf(net)
        #print('net rgb,sdf shape: ',net_rgb.shape, '\n', net_sdf.shape)

        #out = torch.cat([net_sdf,net_rgb],axis=1)
        #out = out.squeeze(1).transpose(2,1) # --> (BS, num_points, channels)
        
        out = net_sdf.squeeze()
        out = self.sigm(out)"""

        out = self.actvn(self.lin1(points))
        out = self.actvn(self.lin2(out))
        out = self.actvn(self.lin3(out))
        out = self.actvn(self.lin4(out))
        out = self.actvn_out(self.lin5(out)).squeeze()

        return out
        


class NonProjectiveFeatureExtractor(nn.Module):
    
    def __init__(self, f1, f2, f3, f4, f5):
        super(NonProjectiveFeatureExtractor, self).__init__()
        
        self.conv_1 = nn.Conv2d(3, f1, 3, padding=1)  # out: (224, f1) , f = 16, 32, 64, 128
        self.conv_1_1 = nn.Conv2d(f1, f2, 3, padding=1)  # out: (224, f2)
        self.conv_2 = nn.Conv2d(f2, f2, 3, padding=1)  # out: (112, f3)
        self.conv_2_1 = nn.Conv2d(f2, f3, 3, padding=1)  # out: (112, f4)
        self.conv_3 = nn.Conv2d(f3, f3, 3, padding=1)  # out: (64, f4)
        self.conv_3_1 = nn.Conv2d(f3, f4, 3, padding=1)  # out: (64, f4)
        self.conv_4 = nn.Conv2d(f4, f4, 3, padding=1)  # out: (64, f4)
        self.conv_4_1 = nn.Conv2d(f4, f5, 3, padding=1)  # out: (64, f4)
        self.conv_out = nn.Conv1d(f5, 1, 1) #out: 

        self.actvn = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)

        self.conv1_1_bn = nn.BatchNorm2d(f2)
        self.conv2_1_bn = nn.BatchNorm2d(f3)
        self.conv3_1_bn = nn.BatchNorm2d(f4)
        self.conv4_1_bn = nn.BatchNorm2d(f5)

    def forward(self, batch):
        net = self.actvn(self.conv_1(batch['target'])) # (bs,16,224,224)
        net = self.actvn(self.conv_1_1(net)) #(bs,32,224,224)
        net = self.conv1_1_bn(net)
        net = self.maxpool(net)#(bs,32,112,112)
        net = self.actvn(self.conv_2(net))#(bs,32,112,112)
        net = self.actvn(self.conv_2_1(net))#(bs,64,112,112)
        net = self.conv2_1_bn(net)
        net = self.maxpool(net)#(bs,64,56,56)
        net = self.actvn(self.conv_3(net))#(bs,64,56,56)
        net = self.actvn(self.conv_3_1(net))#(bs,128,56,56)
        net = self.conv3_1_bn(net)
        net = self.maxpool(net)#(bs,128,28,28)
        net = self.actvn(self.conv_4(net))#(bs, 256, 28,28)
        net = self.actvn(self.conv_4_1(net))#(bs, 256, 28,28)
        net = self.conv4_1_bn(net)#(bs, 256, 28,28)
        net = torch.flatten(net, start_dim=2)
        net = self.conv_out(net) #(bs,256,28*28)    
        return net #(bs,1,784)
        


class FeatureExtractor(nn.Module):

    def __init__(self, f1, f2, f3, f4):
        super(FeatureExtractor, self).__init__()        

        #2d convs, 3 input channels, 3x224 in (1 for now)
        self.convRGB = nn.Conv2d(3,1,1) #flatten RGB to feature map
        self.conv_1 = nn.Conv2d(3, f1, 3, padding=1)  # out: (224, f1) , f = 16, 32, 64, 128
        self.conv_1_1 = nn.Conv2d(f1, f2, 3, padding=1)  # out: (224, f2)
        self.conv_2 = nn.Conv2d(f2, f2, 3, padding=1)  # out: (112, f3)
        self.conv_2_1 = nn.Conv2d(f2, f3, 3, padding=1)  # out: (112, f4)
        self.conv_3 = nn.Conv2d(f3, f3, 3, padding=1)  # out: (64, f4)
        self.conv_3_1 = nn.Conv2d(f3, f4, 3, padding=1)  # out: (64, f4)

        self.actvn = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)

        #get rid of batchnorm maybe
        self.conv1_1_bn = nn.BatchNorm2d(f2)
        self.conv2_1_bn = nn.BatchNorm2d(f3)
        self.conv3_1_bn = nn.BatchNorm2d(f4)

        #2D-displacements could be useful, change coords
        """
        displacment = 0.035
        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)

        self.displacments = torch.Tensor(displacments)"""

    def forward(self, batch):
        x = batch['target']
        p = batch['points']
        #3d case
        #bs, h, w, px = points.shape
        #p = (points * 2 - 1).reshape(bs, 1, h*w, 2)

        #3d -> 2d
        #print(x.shape)
        #points = points[..., :2] #'orthogonal projection'
        
        bs, hw, px = p.shape
        p = (p * 2).unsqueeze(1)#.unsqueeze(1)
        p_z = p[..., 2]
        p = p[..., :2]
        #p = project_perspective(p)
        #p = torch.cat([p + d for d in self.displacments.to(p.device)], dim=2)  # (B,1,7,num_samples,3)
        #x = self.convRGB(x) #1x1 conv to make picture 1 channel
        #print(p.shape, '\n', x.shape)
        feature_0 = F.grid_sample(x, p)  # out : (B,C (of x), 1,1,sample_num)

        net = self.actvn(self.conv_1(x))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        feature_1 = F.grid_sample(net, p)  # out : (B,C (of x), 1,1,sample_num)

        net = self.maxpool(net)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        feature_2 = F.grid_sample(net, p)

        net = self.maxpool(net)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        feature_3 = F.grid_sample(net, p)

        # here every channel corresponse to one feature.

        features = torch.cat((p_z.unsqueeze(2), feature_0, feature_1, feature_2, feature_3), dim=1)  # (B, features, 1,7,sample_num)
        return features

class PreTrainedFeatureExtractor(nn.Module):

    # All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), 
    # where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using 
    # mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

    def __init__(self, f1, f2, f3, f4):
        super(PreTrainedFeatureExtractor, self).__init__()
        resnet18 = models.resnet18(pretrained=True)

    def forward(self, x):
        p = torch.resnet18(x)


class GaussianFourierFeatureTransform(torch.nn.Module):
    #Given an input of size [batches, num_input_channels, width, height],
    #returns a tensor of size [batches, mapping_size*2, width, height].

    def __init__(self, num_input_channels, mapping_size=256, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = torch.randn((num_input_channels, mapping_size)) * scale

    def forward(self, x):
        assert x.dim() == 4, 'Expected 4D input (got {}D input)'.format(x.dim())

        batches, channels, width, height = x.shape

        assert channels == self._num_input_channels,\
            "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)

        # Make shape compatible for matmul with _B.
        # From [B, C, W, H] to [(B*W*H), C].
        x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)

        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = x.view(batches, width, height, self._mapping_size)
        # From [B, W, H, C] to [B, C, W, H]
        x = x.permute(0, 3, 1, 2)

        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)


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

def make_2d_grid(bb_min, bb_max, shape):
    size = shape[0] * shape[1]
    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])

    pxs = pxs.view(-1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys], dim=1)
    return p

def evaluate_network_on_2d_grid(network, x, resolution):
    points_batch_size = args.num_points * args.batch_size #55000 #num_points * batch_size
    pointsf = make_2d_grid(
        (-1,)*2, (1,)*2, resolution
    )
    p_split = torch.split(pointsf, points_batch_size)
    values = []
    for pi in p_split:
        pi = pi.unsqueeze(0).to(x.device)
        with torch.no_grad():
            sdf, rgb = network(x, pi)[:,0], network(x, pi)[:,1:]
        values.append(sdf.squeeze(0).detach().cpu())
    value = torch.cat(values, dim=0).numpy()
    value_grid = value.reshape(3, resolution[0], resolution[1])
    return value_grid

def evaluate_network_on_grid(network, batch, resolution, res_increase = args.inf_res):
    points_batch_size = 65536 #55000 #num_points * batch_size
    pointsf = make_3d_grid(
        (-0.5,)*3, (0.5,)*3, resolution, res_increase
    )
    p_split = torch.split(pointsf, points_batch_size)
    probe = {}
    probe['target'] = batch['target'][0].unsqueeze(0)
    values = []
    for pi in p_split:
        probe['points'] = pi.unsqueeze(0).to(batch['target'].device)
        with torch.no_grad():
            out = network(probe)         

        values.append(out.squeeze(0).detach().cpu())
    value = torch.cat(values, dim=0).numpy()
    value_grid = value.reshape(res_increase*resolution[0], res_increase*resolution[1], res_increase*resolution[2], 1)
    #print(value_grid)
    return value_grid


def implicit_to_mesh(network, batch, resolution, threshold_p, output_path, res_increase=args.inf_res):
    #print(threshold_p)
    value_grid = evaluate_network_on_grid(network, batch, resolution, res_increase)  
    visualize_sdf(value_grid, output_path, level=threshold_p, rgb=True)

def implicit_to_img(network, x, resolution, output_path):
    value_grid = evaluate_network_on_2d_grid(network, x, resolution)
    visualize_implicit_rgb(value_grid, output_path)

def project_perspective(points):
    points[...,0] = points[...,0] / -points[...,2]
    points[...,1] = points[...,1] / -points[...,2]
    return points[..., :2]