from torch.utils.data import Dataset
from torchvision import transforms
import torch
from pathlib import Path
import numpy as np
from numpy.linalg import inv
import pyexr
from skimage.color import rgba2rgb
from data_processing.data_processing import reproject_image, generate_frustum, generate_frustum_volume
import fast_histogram

torch.manual_seed(17)

class ImplicitDataset(Dataset):

    def __init__(self, split, dataset_path, splitsdir, hparams=None, transform = None):
        self.rng = np.random.default_rng()
        if hparams.precision == 32: self.precision = np.float32
        else: self.precision = np.float16
        hparams.precision == np.float32
        self.hparams = hparams
        self.intrinsic = np.array(  
            [[245       ,   0.       ,   122,  0.],
            [  0.       , 245        ,   122,  0.],
            [  0.       ,   0.       ,   1. ,  0.],
            [  0.       ,   0.       ,   0. ,  1.]])
        self.transform = transform
        self.clamp = self.hparams.clamp
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.splitsdir = splitsdir
        self.split_shapes = [x.strip() for x in (Path("../data/splits") / splitsdir / f"{split}.txt").read_text().split("\n") if x.strip() != ""]
        self.data = [x for x in self.split_shapes]
        #self.data = self.data * (400 if ('overfit' in splitsdir) and split == 'train' else 1)
        # feature extractor hparam, only batch relevant info

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sample_folder = Path(item)

        sample_idx = str(np.random.randint(0,30)*12).zfill(3)
        points, rgb, sdf = self.prepare_points(sample_folder, pad_with_normals=True, num_points=self.hparams.num_points)
        #surface_points, surface_rgb, surface_sdf = self.prepare_points(sample_folder, itemname='surface_points_blender.npz', num_points = self.hparams.num_surface_points)
        #surface_points, _, _ = self.prepare_points(sample_folder, itemname='ptc_surface.npy', num_points = self.hparams.num_surface_points)
        

        #load image and depth map
        image = pyexr.read(str(sample_folder / f'_r_{sample_idx}.exr'))
        image[...,:3] = image[...,:3]/3
        depth = pyexr.read(str(sample_folder / f'_r_{sample_idx}_depth0001.exr'))[...,0]

        camera = np.loadtxt(sample_folder / f'RT{sample_idx}.txt').astype(self.precision)
        #fx = 245, fy=245, cx = 112, cy = 112
        
        ud_flip = np.array([[1, 0, 0],[0,-1,0],[0,0,1]], dtype=self.precision)
        lr_flip = np.array([[-1, 0, 0],[0,1,0],[0,0,1]], dtype=self.precision)
        #apply transforms for train:
        if ('train' in self.split) and (self.hparams.aug_threshhold is not None):
            
            #Generate random number to decide for each individual augmentation
            p = np.random.rand(10)
            threshhold = self.hparams.aug_threshhold
            
            #random backgrounds
            background = np.array([1,1,1]) #white
            if ('Background' in self.transform) and (p[0] >= threshhold):
                background = p[:3]

            #create mask for pixels vs background
            mask = image[...,3].copy().flatten()

            #Remove alpha channel
            image = rgba2rgb(image, background = background)

            #center and 0 std:            
            #image = whiten(image)
            
            #Mask out white pixels and change non-background pixels
            if ('Brightness' in self.transform) and (p[1] >= threshhold):
                brightness = np.random.uniform(low=0.8, high=1.2, size=1)
                image = image.reshape(-1,3)
                image[mask == 1, :] *= brightness
                image = image.reshape(224, 224, 3)
                rgb *= brightness
                if surface_rgb is not None: surface_rgb *= brightness
            
            if ('ColorJitter' in self.transform) and (p[2] >= threshhold):
                jitter = np.random.uniform(low=-0.1, high=0.1, size=(3))
                image = image.reshape(-1,3)
                image[mask == 1, :] += jitter
                image = image.reshape(224, 224, 3)
                rgb +=  jitter
                if surface_rgb is not None: surface_rgb +=  jitter

            if ('ColorPermute' in self.transform) and (p[3] >= threshhold):
                roll_int = np.random.randint(1,3)
                image = np.roll(image, roll_int, axis=-1)
                rgb = torch.roll(rgb, roll_int, dims=-1)     
                if surface_rgb is not None: surface_rgb = torch.roll(surface_rgb, roll_int, dims=-1)     

            if ('HorizontalFlip' in self.transform) and (p[4] >= threshhold):
                image = np.ascontiguousarray(np.fliplr(image))
                camera[:3,:3] = lr_flip @ camera[:3,:3]  # ORDER MATTERS
                depth = np.ascontiguousarray(np.fliplr(depth))

            if ('VerticalFlip' in self.transform) and (p[5] >= threshhold):
                image = np.ascontiguousarray(np.flipud(image))
                camera[:3,:3] = ud_flip @ camera[:3,:3]
                depth = np.ascontiguousarray(np.flipud(depth))
            
        if image.shape[-1] == 4: 
            image = rgba2rgb(image)
            #image = whiten(image)
        #image[...,:3] = (image[...,:3]-image[...,:3].min())/(image[...,:3].max()-image[...,:3].min())
        if self.hparams.encoder not in ('conv3d', 'ifnet', 'hybrid', 'hybrid_ifnet', 'hybrid_depthproject', 'hybrid_surface'):
            voxels = np.empty(1)
            """
            #use ground truth pre-computed voxel grid
        elif self.hparams.encoder in ('hybrid_depthproject'):
            try:
                voxels = np.load(sample_folder / 'voxels_fixed.npy').astype(np.float32)
            except FileNotFoundError:
                voxels = np.load(sample_folder / 'voxels.npy').astype(np.float32)
            #if self.hparams.fieldtype == 'occupancy':
            occvoxel = voxels.copy()
            occvoxel[voxels <= 0] = 1
            occvoxel[voxels > 0] = 0
            voxels = occvoxel
            voxels = voxels[np.newaxis, :]
            """

        else:  
            num_voxels = self.hparams.num_voxels
            voxel_size = 1/num_voxels + 1e-6
            if self.hparams.num_surface_points > 0:
                # using chibane surface-voxels (dense = 3k, sparse = 300)
                # for surface reconstruction experiments, points are in canonic space            
                xyz, rgbdots, _ = self.prepare_points(sample_folder, itemname='surface_points_blender.npz', num_points = 100000)
                pts_grid = np.round((xyz + 0.5) * num_voxels).transpose(1,0)
        
            # get camera space to grid space transform  
            else:
                # depth from camera to grid space
                rgbxyz = reproject_image(image, depth, RT=camera, f=245, cx=112, cy=112)
                # subselect 3k surface points
                #n = rgbxyz.shape[0]
                #rnd_indexes = self.rng.permutation(n)[:3000]
                rgbxyz = rgbxyz#[rnd_indexes]
                rgbdots = rgbxyz[...,3:]
                xyz = rgbxyz[...,:3]            
                intrinsic_inv = (inv(self.intrinsic))
                frustum = generate_frustum([244, 244], intrinsic_inv, -0.5, 0.5)
                _, camera2frustum = generate_frustum_volume(frustum, voxel_size)
                coords = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=-1)
                pts_grid = (camera2frustum @ coords.transpose(1,0))[:3, :].transpose(1,0).reshape(-1, 3) #depth_points_in_gridspace
                pts_grid += 0.5                      
            
            if self.hparams.voxel_type == 'colored_density':
                #normalized density voxelgrid with mean-average color of points inside
                voxels_occ = fast_histogram.histogramdd(pts_grid, bins=num_voxels, range=[[0, num_voxels], [0, num_voxels], [0, num_voxels]])
                voxels_r = fast_histogram.histogramdd(pts_grid, bins=num_voxels, range=[[0, num_voxels], [0, num_voxels], [0, num_voxels]], weights=rgbdots[:,0])
                voxels_g = fast_histogram.histogramdd(pts_grid, bins=num_voxels, range=[[0, num_voxels], [0, num_voxels], [0, num_voxels]], weights=rgbdots[:,1])
                voxels_b = fast_histogram.histogramdd(pts_grid, bins=num_voxels, range=[[0, num_voxels], [0, num_voxels], [0, num_voxels]], weights=rgbdots[:,2])
                voxels = np.stack((voxels_occ, voxels_r, voxels_g, voxels_b),axis=0)
                voxels[1:, voxels[0] != 0] /= voxels[0, voxels[0] != 0]
                voxels[0] /= np.nanmax(voxels[0])
            elif self.hparams.voxel_type == 'density':
                #normalized density voxelgrid
                voxels = fast_histogram.histogramdd(pts_grid, bins=num_voxels, range=[[0, num_voxels], [0, num_voxels], [0, num_voxels]])
                voxels /= np.nanmax(voxels)
                voxels = voxels[np.newaxis, :]
            elif self.hparams.voxel_type == 'occupancy':
                #occuapncy voxelgrid (0,1)
                voxels = fast_histogram.histogramdd(pts_grid, bins=num_voxels, range=[[0, num_voxels], [0, num_voxels], [0, num_voxels]])
                voxels[voxels > 0] = 1
                voxels = voxels[np.newaxis,:].reshape((1, num_voxels, num_voxels, num_voxels))
            
            #for pointnet encoder - use pseudovoxels (= points) to keep same namespace
            if self.hparams.encoder in ('hybrid_surface'):
                n = rgbxyz.shape[0]
                rnd_indexes = self.rng.permutation(n)[:3000]
                voxels = rgbxyz[rnd_indexes,:]
             
            
        #Do these transformations regardless of train/val/test --> ([0,1] normalization, axis permutation)
        xyz_to_blender = np.array([[1, 0, 0],[0,0,-1],[0,1,0]], dtype=self.precision)
        camera[:3,:3] = camera[:3,:3] @ xyz_to_blender #bring it from blender space to pytorch3d space (z into image, righthanded)

        image = torch.from_numpy(image).permute(2, 0, 1)
        camera = torch.from_numpy(camera)
        voxels = torch.from_numpy(voxels.astype(self.precision))
        #rgb = torch.cat((rgb, surface_rgb), dim=0)
        #sdf = torch.cat((sdf, surface_sdf), dim=0)
        #points = torch.cat((points, surface_points), dim=0)
        mesh_path = str(sample_folder / item / 'disn_mesh.obj')
        #normalize RGB-Pointclouds to 0, 1
        #rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())         
        
        #if self.hparams.encoder in ('conv2d_pretrained', 'conv2d_pretrained_projective', 'hybrid', 'hybrid_ifnet', 'hybrid_depthproject') and self.hparams.freeze_pretrained is not None:
        #    image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)

        return {
            'name': item,
            'camera': camera,
            'points': points,
            'rgb': rgb,
            'sdf': sdf,
            'image': image,
            'voxels': voxels,
            'sample_idx': sample_idx,
            #'surface_points': surface_points,
                    }

    def prepare_points(self, sample_folder, itemname = "fixed_point_samples_blender3.npz", num_points = None, pad_with_normals=False):
        if num_points is None:
            num_points = self.hparams.num_points

        points, rgb, sdf, normals = [], [], [], []
        
        sample_points = np.load(sample_folder / itemname)
        if type(sample_points) is not np.ndarray:
            if 'train' not in self.split: num_points *= 3
            subsample_indices = np.random.randint(0, sample_points['points'].shape[0], num_points)

            points = sample_points['points'][subsample_indices]
            rgb = sample_points['rgb'][subsample_indices]
            sdf = sample_points[f'{self.hparams.fieldtype}'][subsample_indices]
            

        else: #this is a hacky way to make it work for surface points, should have saved them as npz
            subsample_indices = np.random.randint(0, sample_points.shape[0], num_points)
            points = sample_points[subsample_indices]
            rgb = np.arange(3)
            sdf = np.arange(1)     

        """
        if pad_with_normals and ('train' in self.split):       
            #change original input too? points = point+epsilon
            normals = sample_points['normals'][subsample_indices]    
            normals = normals/(np.linalg.norm(normals,axis=1, keepdims=True))
            displacements_n = np.random.uniform(low=0.5, high=0.9, size=(points.shape[0],1)) 
            displacements_e = np.random.beta(a=2, b=2, size=(points.shape[0], 1))

            points_n2 = points.copy() - displacements_n*((sdf[:,np.newaxis]))*normals
            sdf_n2 = sdf.copy() - sdf.copy()*displacements_n.squeeze()

            points = points - 0.1*displacements_e*((sdf[:,np.newaxis]))*normals
            sdf = sdf - sdf*0.1*displacements_e.squeeze()

            points = np.concatenate([points,points_n2], axis=0)
            sdf = np.concatenate([sdf,sdf_n2])
            rgb = np.concatenate([rgb,rgb], axis=0)
        """
        
        points = torch.from_numpy(points.astype(self.precision)).reshape(-1,3)
        rgb = torch.from_numpy(rgb.astype(self.precision)).reshape(-1,3)
        sdf = torch.from_numpy(sdf.astype(self.precision))           

        return points, rgb, sdf


    def rotate_world_to_view(self,points, R, T):
        xyz_to_blender = np.array([[1, 0, 0],[0,0,-1],[0,1,0]])
        R_b2CV = np.array([[-1,0,0],[0,-1,0],[0,0,1]], dtype=self.precision)
        
        viewpoints = (R @ xyz_to_blender @ points.transpose(1,0)).transpose(1,0) + T
        viewpoints = (R_b2CV @ viewpoints.transpose()).transpose()
        return viewpoints
    

def whiten(image):
    image = torch.from_numpy(image).permute(2, 0, 1)
    
    #statistics of blender_copy_fixed 'train' split
    ## images mean and std
    #image = transforms.Normalize((0.11082486,0.09608462,0.09117579), (0.2149361,0.19069198,0.18358029))(image)

    ## points mean and std
    #image = transforms.Normalize((0.36911592, 0.31920254, 0.30284569), (0.23451681, 0.21427469, 0.20948004))(image)

    #chibane / image / mean and std
    image = transforms.Normalize((0.10992143, 0.09546989, 0.09096114), (0.21359299, 0.18972058, 0.18306873))(image)
    image = image.permute(1, 2, 0).numpy()
    return image

if __name__ == "__main__":
    from util import arguments
    _args = arguments.parse_arguments()
    dataset = ImplicitDataset("train", "../data", "overfit", _args)
    datatest = dataset[0]
    print(dataset[0])
    print(dataset[0]['sdf'].shape)
