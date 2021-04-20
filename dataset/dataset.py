from torch.utils.data import Dataset
from torchvision import transforms
import torch
from pathlib import Path
import numpy as np
from PIL import Image


class ImplicitDataset(Dataset):

    def __init__(self, split, dataset_path, splitsdir, hparams, transform = None):
        self.hparams = hparams
        self.transform = transform
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.splitsdir = splitsdir
        self.split_shapes = [x.strip() for x in (Path("data/splits") / splitsdir / f"{split}.txt").read_text().split("\n") if x.strip() != ""]
        self.data = [x for x in self.split_shapes]
        self.data = self.data * (50 if ('overfit' in splitsdir) and split == 'train' else 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sample_folder = Path(self.dataset_path) / "processed" / self.splitsdir / item
        sample_input = np.array(Image.open(sample_folder / "0000.jpg"))#.convert("RGB")).transpose((2, 0, 1))
        sample_input = torch.from_numpy(sample_input)
        sample_target = sample_input

        return {
            'name': item,
            'input': (sample_input).unsqueeze(0),
            'target': (sample_target).unsqueeze(0)
        }


if __name__ == "__main__":
    dataset = ImplicitDataset("train", "data", "overfit")
    print(dataset[0])
