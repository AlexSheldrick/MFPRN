from torch.utils.data import Dataset
from torchvision import transforms
import torch
from pathlib import Path
import numpy as np

import imageio
#from .model.model import GaussianFourierFeatureTransform


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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sample_folder = Path(self.dataset_path) / "processed" / self.splitsdir / item
        #path = sample_folder / "cat2_scaled.jpg"

        #this can be replaced by openexr 4 channel RGBD readout
        """myrand = np.random.rand(8,224,224,4) #8 batches of 224x224 RGBD images
        myrand_split_1, myrand_split_2 = np.split(myrand, [3], axis=3)
        print(myrand_split_1.shape, myrand_split_2.shape)""" ##split_1 is rgb and split_2 is d
        target = torch.tensor(get_image(sample_folder / 'rgb.png')).permute(2, 0, 1) # (224xH,224xW,3xC) --> (3C, 224xH, 224xW)
        depth = torch.tensor(get_image(sample_folder / 'depth.png')).unsqueeze(2).permute(2, 0, 1) # (224xH,224xW,1xC) --> (1C, 224xH, 224xW)
        

        points, rgb, sdf = self.prepare_points(sample_folder)
        #points = pixels_to_points(target)
        #rgb, sdf = [], []
        #depth = []        
        
        # points are strictly dependant of batch_size via matrix reshape. Could precompute?`This is for 1`
        #points = torch.from_numpy(np.load(sample_folder / "x_ff.npy")).squeeze(0) #(1, 2, 256, 10)

        #points = GaussianFourierFeatureTransform(2, 128, 10)(points)

        return { #rgb, depth, points
            'name': item,
            'points': points,
            'rgb': rgb,
            'sdf': sdf,
            'depth': depth,
            'target': target
                    }

    def prepare_points(self, sample_folder):
        
        points, rgb, sdf = [], [], []
        
        sample_points = np.load(sample_folder / "point_samples.npz")

        subsample_indices = np.random.randint(0, sample_points['points'].shape[0], self.hparams.num_points)

        points = sample_points['points'][subsample_indices]
        rgb = sample_points['rgb'][subsample_indices]
        sdf = sample_points['sdf'][subsample_indices]

        points = torch.from_numpy(points.astype(np.float32)).reshape(-1,3)
        rgb = torch.from_numpy(rgb.astype(np.float32)).reshape(-1,3)        
        sdf = torch.from_numpy(sdf.astype(np.float32))

        #to truncate sdf to a treshold
        if self.clamp > 0:
            sdf[sdf > self.clamp] = self.clamp
            sdf[sdf < -self.clamp] = -self.clamp

        #this turns sdf into occupancy
        elif self.clamp == 0:
            sdf[sdf > 0] = 0
            sdf[sdf < 0] = 1

        #this sets it to UDF with 0.2 clamping
        elif self.clamp  == -1:
            sdf[sdf < 0] = -sdf[sdf<0]
            sdf[sdf > 0.2] = 0.2            


        return points, rgb, sdf

def get_image(path):
    img = imageio.imread(path)
    if img.ndim > 2:
        img = img[..., :3] / 255. #normalized to [0,1] (224xH,224xW,3xC) images
    else: img = img / 255.

    return img.astype(np.float32)

def pixels_to_points(image):
    points = np.linspace(0, 1, image.shape[2], endpoint=False)
    xy_grid = np.stack(np.meshgrid(points, points), -1)
    xy_grid = torch.tensor(xy_grid).float().contiguous() # shape: 224,224,2 .permute(2, 0, 1)
    return xy_grid


if __name__ == "__main__":
    from util import arguments
    _args = arguments.parse_arguments()
    dataset = ImplicitDataset("train", "../data", "overfit", _args)
    datatest = dataset[0]
    print(dataset[0])
    print(dataset[0]['sdf'].shape)
