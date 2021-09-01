from torch.utils.data import Dataset
from torchvision import transforms
import torch
from pathlib import Path
import numpy as np
import pyexr
from skimage.color import rgba2rgb

class ImplicitDataset(Dataset):

    def __init__(self, split, dataset_path, splitsdir, hparams=None, transform = None):
        
        self.hparams = hparams
        self.clamp = self.hparams.clamp
        self.transform = transform
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

        #convert RGBA to RGB image with random background
        image = pyexr.read(str(sample_folder / f'_r_{sample_idx}.exr'))
        background = np.random.rand(3,)
        image = rgba2rgb(image, background = background)
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = (image-image.min())/(image.max()-image.min())
        
        #depth = pyexr_array[4,...].unsqueeze(0)
        #depth = torch.from_numpy(pyexr.read(str(sample_folder / f'_r_{sample_idx}_depth0001.exr'))).permute(2, 0, 1)[0,...].unsqueeze(0)

        #only send voxels if necessary
        voxels = torch.empty((1,1))
        
        if self.hparams.encoder in ('conv3d', 'ifnet', 'hybrid', 'hybrid_ifnet'):
            voxels = torch.from_numpy(np.load(sample_folder / 'voxels.npy').astype(np.float32)).unsqueeze(0)
            if self.hparams.fieldtype == 'occupancy':
                occvoxel = voxels.clone().detach()
                occvoxel[voxels <= 0] = 1
                occvoxel[voxels > 0] = 0
                voxels = occvoxel
        
        if self.hparams.encoder in ('conv2d_pretrained', 'conv2d_pretrained_projective', 'hybrid', 'hybrid_ifnet'):
            image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)

        camera = torch.from_numpy(np.loadtxt(sample_folder / f'RT{sample_idx}.txt').astype(np.float32))
        xyz_to_blender = torch.tensor([[1, 0, 0],[0,0,-1],[0,1,0]], dtype=torch.float32)
        camera[:3,:3] = camera[:3,:3] @ xyz_to_blender
        #fx = 245, fy=245, cx = 112, cy = 112

        return {
            'name': item,
            'camera': camera,
            'points': points,
            'rgb': rgb,
            'sdf': sdf,
            'depth': depth,
            'image': image,
            'voxels':voxels,
            'sample_idx':sample_idx
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

        #normalize RGB to 0, 1
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())    

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
