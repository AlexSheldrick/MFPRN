from torch.utils.data import Dataset
from torchvision import transforms
import torch
from pathlib import Path
import numpy as np

import imageio
from model.model import GaussianFourierFeatureTransform


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
        path = sample_folder / "cat2_scaled.jpg"

        target = torch.tensor(get_image(path)).permute(2, 0, 1) # (224xH,224xW,3xC) --> (3C, 224xH, 224xW)
        points = pixels_to_points(target)
        
        # points are strictly dependant of batch_size via matrix reshape. Could precompute?`This is for 1`
        #points = torch.from_numpy(np.load(sample_folder / "x_ff.npy")).squeeze(0) #(1, 2, 256, 10)

        #points = GaussianFourierFeatureTransform(2, 128, 10)(points)

        return {
            'name': item,
            'points': points,
            'target': target
                    }


def get_image(path):
    img = imageio.imread(path)[..., :3] / 255. #normalized to [0,1] (224xH,224xW,3xC) images
    return img.astype(np.float32)

def pixels_to_points(image):
    points = np.linspace(0, 1, image.shape[2], endpoint=False)
    xy_grid = np.stack(np.meshgrid(points, points), -1)
    xy_grid = torch.tensor(xy_grid).float().contiguous() # shape: 224,224,2 .permute(2, 0, 1)
    return xy_grid


if __name__ == "__main__":
    dataset = ImplicitDataset("train", "data", "overfit")
    print(dataset[0])
