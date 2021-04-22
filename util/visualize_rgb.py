import imageio
import numpy as np
import torch

def visualize_implicit_rgb(value_grid, output_path):
    img = tensor_to_numpy(value_grid)
    img = imageio.imwrite(str(output_path), img)

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor * 256
    tensor[tensor > 255] = 255
    tensor[tensor < 0] = 0
    tensor = tensor.type(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return tensor