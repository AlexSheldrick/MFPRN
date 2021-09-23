from torch.utils.data import Dataset
from torchvision import transforms
import torch
from pathlib import Path
import numpy as np
import pyexr
from skimage.color import rgba2rgb
from data_processing.data_processing import reproject_image, generate_frustum, generate_frustum_volume

torch.manual_seed(17)

class ImplicitDataset(Dataset):

    def __init__(self, split, dataset_path, splitsdir, hparams=None, transform = None):
        
        self.hparams = hparams
        self.intrinsic = torch.tensor(  
            [[245       ,   0.       ,   122,  0.],
            [  0.       , 245        ,   122,  0.],
            [  0.       ,   0.       ,   1. ,  0.],
            [  0.       ,   0.       ,   0. ,  1.]]
                                    )
        self.transform = transform
        self.clamp = self.hparams.clamp
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.splitsdir = splitsdir
        self.split_shapes = [x.strip() for x in (Path("../data/splits") / splitsdir / f"{split}.txt").read_text().split("\n") if x.strip() != ""]
        self.data = [x for x in self.split_shapes]
        self.data = self.data * (400 if ('overfit' in splitsdir) and split == 'train' else 1)
        # feature extractor hparam, only batch relevant info

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sample_folder = Path(item)

        sample_idx = str(np.random.randint(0,30)*12).zfill(3)
        
        #image = torch.tensor(get_image(sample_folder / f'rgb{sample_idx}.png')).permute(2, 0, 1) # (224xH,224xW,3xC) --> (3C, 224xH, 224xW)
        #depth = torch.tensor(get_image(sample_folder / f'depth{sample_idx}.png')).unsqueeze(2).permute(2, 0, 1) # (224xH,224xW,1xC) --> (1C, 224xH, 224xW)

        points, rgb, sdf = self.prepare_points(sample_folder)
        
        #only send depth if necessary
        depth = torch.empty((1,1))

        #load image and depth map
        image = pyexr.read(str(sample_folder / f'_r_{sample_idx}.exr'))
        depth = pyexr.read(str(sample_folder / f'_r_{sample_idx}_depth0001.exr'))[...,0]
        
        #deprecated, using dynamic voxel grid now
        """#only send voxels if necessary
        voxels = torch.empty((1,1))
        
        voxels = np.load(sample_folder / 'voxels.npy').astype(np.float32)
        if self.hparams.fieldtype == 'occupancy':
            occvoxel = voxels.copy()
            occvoxel[voxels <= 0] = 1
            occvoxel[voxels > 0] = 0
            voxels = occvoxel"""

        camera = np.loadtxt(sample_folder / f'RT{sample_idx}.txt').astype(np.float32)
        
        
        ud_flip = np.array([[1, 0, 0],[0,-1,0],[0,0,1]], dtype=np.float32)
        lr_flip = np.array([[-1, 0, 0],[0,1,0],[0,0,1]], dtype=np.float32)
        
        #fx = 245, fy=245, cx = 112, cy = 112

        #apply transforms for train:
        if 'train' in self.split:
            
            #Generate random number to decide for each individual augmentation
            p = torch.rand(10)
            threshhold = self.hparams.aug_threshhold
            
            #random backgrounds
            white = np.array([1,1,1])
            background = white
            if ('Background' in self.transform) and (p[0] >= threshhold):
                background =  p[:3]

            #Remove alpha channel
            image = rgba2rgb(image, background = background)
            
            #Mask out white pixels and change non-white pixels
            if ('Brightness' in self.transform) and (p[1] >= threshhold):
                brightness = np.random.uniform(low=0.8, high=1.2, size=1)
                image[np.all(image != white, axis=2)] *= brightness
                rgb *= brightness
            
            if ('ColorJitter' in self.transform) and (p[2] >= threshhold):
                jitter = np.random.uniform(low=-0.1, high=0.1, size=(3))
                image[np.all(image != white, axis=2)] += jitter
                rgb +=  jitter

            if ('ColorPermute' in self.transform) and (p[3] >= threshhold):
                image2 = image.copy()
                rands = np.random.permutation([0,1,2])

                image[...,rands[0]] = image2[...,rands[2]]
                image[...,rands[1]] = image2[...,rands[0]]
                image[...,rands[2]] = image2[...,rands[1]]

                rgb2 = rgb.clone().detach()
                rgb[...,rands[0]] = rgb2[...,rands[2]]
                rgb[...,rands[1]] = rgb2[...,rands[0]]
                rgb[...,rands[2]] = rgb2[...,rands[1]]            

            #do voxels need to be flipped??
            if ('HorizontalFlip' in self.transform) and (p[4] >= threshhold):
                image = np.ascontiguousarray(np.fliplr(image))
                camera[:3,:3] = lr_flip @ camera[:3,:3]  # ORDER MATTERS
                depth = np.ascontiguousarray(np.fliplr(depth))

            if ('VerticalFlip' in self.transform) and (p[5] >= threshhold):
                image = np.ascontiguousarray(np.flipud(image))
                camera[:3,:3] = ud_flip @ camera[:3,:3]
                depth = np.ascontiguousarray(np.flipud(depth))

            
            if ('Rotation' in self.transform) and (p[6] >= threshhold): #rotation around viewing direction == +z
                pass
            
            #hard augmentations
            #render different model positions (R,T) -> have to shift and rotate model by R,T
            #pixel distance? Voxel distance?
            
        if image.shape[-1] == 4: image = rgba2rgb(image)
        image[...,:3] = (image[...,:3]-image[...,:3].min())/(image[...,:3].max()-image[...,:3].min())

        # dynamic voxel grid:
        # load image & depthmap
        # reproject points
        # frustrum??

        # get camera space to grid space transform
        intrinsic_inv = torch.inverse(self.intrinsic)
        frustum = generate_frustum([244, 244], intrinsic_inv, -0.5, 0.5)
        dims, camera2frustum = generate_frustum_volume(frustum, 0.01) #voxel size?

        # depth from camera to grid space
        xyz = reproject_image(image, depth, RT=camera, f=245, cx=112, cy=112)[...,:3]

        depth_idx = np.random.randint(0, xyz.shape[0], 2000) #if we keep track of indicies we might save reprojection and rendering, can just compare rgb vals
        depthpoints = xyz[depth_idx].reshape(depth_idx.shape[-1],3)

        #keep track of filtered pixels
        rgb_idx = np.arange(depth.shape[0]*depth.shape[1])
        rgb_idx = rgb_idx[np.nonzero(depth.flatten() < 65500)]
        #the rgb pixels corresponding to subsampled & projected points
        depth_idx = rgb_idx[depth_idx]
    
        coords = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=-1)
        depth_in_gridspace = (camera2frustum @ coords.transpose(1,0))[:3, :].transpose(1,0).reshape(-1, 3)

        voxels = np.zeros((128,)*3)
        to_int = lambda x: np.round(x).astype(np.int32)
        
        #turn this into a density grid of image points
        voxels[to_int(depth_in_gridspace[:, 0]), to_int(depth_in_gridspace[:, 1]), to_int(depth_in_gridspace[:, 2])] += 1
        voxels /= np.nanmax(voxels)

        #Do these transformations regardless of train/val/test --> ([0,1] normalization, axis permutation)
        xyz_to_blender = np.array([[1, 0, 0],[0,0,-1],[0,1,0]], dtype=np.float32)
        camera[:3,:3] = camera[:3,:3] @ xyz_to_blender #bring it from blender space to pytorch3d space (z into image, righthanded)

        
        #we no longer need depth
        #depth = torch.from_numpy(depth)
        depth = torch.empty(1)
        image = torch.from_numpy(image).permute(2, 0, 1)
        camera = torch.from_numpy(camera)
        voxels = torch.from_numpy(voxels.astype(np.float32)).unsqueeze(0)
        depth_idx = torch.from_numpy(depth_idx).to(torch.long)
        depthpoints = torch.from_numpy(depthpoints.astype(np.float32))

        #normalize RGB-Pointclouds to 0, 1
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())         

        if self.hparams.encoder not in ('conv3d', 'ifnet', 'hybrid', 'hybrid_ifnet'):
            voxels = torch.empty(1)

        if self.hparams.encoder in ('conv2d_pretrained', 'conv2d_pretrained_projective', 'hybrid', 'hybrid_ifnet') and self.hparams.freeze_pretrained is not None:
            image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)

        return {
            'name': item,
            'camera': camera,
            'points': points,
            'rgb': rgb,
            'sdf': sdf,
            'depth': depth,
            'image': image,
            'voxels': voxels,
            'sample_idx': sample_idx,
            'depthpoints': depthpoints, 
            'depth_idx': depth_idx
                    }

    def prepare_points(self, sample_folder):
        
        points, rgb, sdf = [], [], []
        
        sample_points = np.load(sample_folder / "point_samples_blender.npz")

        subsample_indices = np.random.randint(0, sample_points['points'].shape[0], self.hparams.num_points)

        points = sample_points['points'][subsample_indices]
        rgb = sample_points['rgb'][subsample_indices]
        sdf = sample_points[f'{self.hparams.fieldtype}'][subsample_indices]

        points = torch.from_numpy(points.astype(np.float32)).reshape(-1,3)
        rgb = torch.from_numpy(rgb.astype(np.float32)).reshape(-1,3)
        sdf = torch.from_numpy(sdf.astype(np.float32))   

        return points, rgb, sdf


def rotate_world_to_view(points, R, T):
    xyz_to_blender = np.array([[1, 0, 0],[0,0,-1],[0,1,0]])
    R_b2CV = np.array([[-1,0,0],[0,-1,0],[0,0,1]], dtype=np.float32)
    
    viewpoints = (R @ xyz_to_blender @ points.transpose(1,0)).transpose(1,0) + T
    viewpoints = (R_b2CV @ viewpoints.transpose()).transpose()
    return viewpoints

if __name__ == "__main__":
    from util import arguments
    _args = arguments.parse_arguments()
    dataset = ImplicitDataset("train", "../data", "overfit", _args)
    datatest = dataset[0]
    print(dataset[0])
    print(dataset[0]['sdf'].shape)
