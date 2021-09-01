from pathlib import Path
import glob
import imageio
import numpy as np
import torch

def list_model_ids(inpath='../data/processed', shapenet_class='car', save=True):
    inpath = Path(inpath)
    inpath = inpath / f'{shapenet_class}'
    names = sorted(glob.glob(str(inpath / '*' / 'name.txt')))
    model_dict = {}
    for idx, name in enumerate(names):
        with open(name, 'r') as f:
            text = f.readlines()[0]
            name = text.replace('/models/model_normalized.obj','').split('/')[-1]
            model_dict[idx] = name

    if save:
        with open(str(inpath / 'model_ids.txt'), 'w') as out:
            for key in model_dict.keys():
                out.writelines(str(key)+' '+str(model_dict[key]) + '\n')
    else: return model_dict

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