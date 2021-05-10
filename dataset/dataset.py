from torch.utils.data import Dataset
from torchvision import transforms
import torch
from pathlib import Path
import numpy as np

import imageio
from .model.model import GaussianFourierFeatureTransform


class ImplicitDataset(Dataset):

    def __init__(self, split, dataset_path, splitsdir, hparams=None, transform = None):
        self.hparams = hparams
        self.transform = transform
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.splitsdir = splitsdir
        self.split_shapes = [x.strip() for x in (Path("data/splits") / splitsdir / f"{split}.txt").read_text().split("\n") if x.strip() != ""]
        self.data = [x for x in self.split_shapes]
        self.data = self.data * (400 if ('overfit' in splitsdir) and split == 'train' else 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sample_folder = Path(self.dataset_path) / "processed" / self.splitsdir / item
        #path = sample_folder / "cat2_scaled.jpg"

        target = torch.tensor(get_image(sample_folder / 'rgb.png')).permute(2, 0, 1) # (224xH,224xW,3xC) --> (3C, 224xH, 224xW)
        depth = torch.tensor(get_image(sample_folder / 'depth.png')).unsqueeze(2).permute(2, 0, 1) # (224xH,224xW,1xC) --> (1C, 224xH, 224xW)
        points, rgb, sdf = self.prepare_points(sample_folder)

        points = pixels_to_points(target)
        rgb, sdf = [], []
        depth = []
        
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
        
        points = []
        rgb = []
        sdf = []

        for sample in ['surface', 'sampled']:
            
            sample_points = np.load(sample_folder / f"pc_{sample}.npz")

            sampling_points = sample_points['points']
            sampling_rgb = sample_points['rgb']
            sampling_sdf = sample_points['sdf']

            subsample_indices = np.random.randint(0, sampling_points.shape[0], self.hparams.num_points)

            np.append(points, sampling_points[subsample_indices])
            np.append(rgb, sampling_rgb[subsample_indices])
            np.append(sdf, sampling_sdf[subsample_indices])

        points = torch.from_numpy(np.array(points, dtype='float32'))
        rgb = torch.from_numpy(np.array(points, dtype='float32'))
        sdf = torch.from_numpy(np.array(points, dtype='float32'))

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
    dataset = ImplicitDataset("train", "data", "overfit")
    datatest = dataset[0]
    print(dataset[0])
