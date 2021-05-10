import mcubes as mc
import numpy as np
import trimesh
from PIL import Image
from pathlib import Path
import pyexr
import torch


def rgba_to_rgb():
    """There is an algorithm for this (from this wikipedia link):
    https://en.wikipedia.org/wiki/Alpha_compositing

    Normalise the RGBA values so that they're all between 0 and 1 - just divide each value by 255 to do this. We'll call the result Source.
    Normalise also the matte colour (black, white whatever). We'll call the result BGColor Note - if the background colour is also transparent, then you'll have to recurse the process for that first (again, choosing a matte) to get the source RGB for this operation.

    Now, the conversion is defined as (in complete psuedo code here!):

    Source => Target = (BGColor + Source) =
    Target.R = ((1 - Source.A) * BGColor.R) + (Source.A * Source.R)
    Target.G = ((1 - Source.A) * BGColor.G) + (Source.A * Source.G)
    Target.B = ((1 - Source.A) * BGColor.B) + (Source.A * Source.B)

    To get the final 0-255 values for Target you simply multiply all the normalised values back up by 255, making sure you cap at 255 if any of the combined values exceed 1.0 (this is over-exposure and there are more complex algorithms dealing with this that involve whole-image processing etc.).

    EDIT: In your question you said you want a white background - in that case just fix BGColor to 255,255,255."""

def to_point_list(s):
    return np.concatenate([c[:, np.newaxis] for c in np.where(s >= 0.5)], axis=1)


def visualize_point_list(grid, output_path):
    f = open(output_path, "w")
    for i in range(grid.shape[0]):
        x, y, z = grid[i, 0], grid[i, 1], grid[i, 2]
        c = [1, 1, 1]
        f.write('v %f %f %f %f %f %f\n' % (x + 0.5, y + 0.5, z + 0.5, c[0], c[1], c[2]))
    f.close()

def visualize_rgb_pointcloud(point_cloud, output_path):
    c = np.zeros_like(point_cloud[:,:3])
    print(c.shape)
    with open(output_path, "w") as f:        
        if point_cloud.shape[1] == 6:
            c = point_cloud[:, 3:]
        for i in range(point_cloud.shape[0]):          
            x, y, z = point_cloud[i, 0], point_cloud[i, 1], point_cloud[i, 2]
            f.write('v %f %f %f %f %f %f\n' % (x, y ,z, c[i,0], c[i,1], c[i,2]))


def visualize_sdf(sdf, output_path, level=0.75):
    vertices, triangles = mc.marching_cubes(sdf.astype(float), level)
    mc.export_obj(vertices, triangles, output_path)


def visualize_grid(grid, output_path):
    point_list = to_point_list(grid)
    if point_list.shape[0] > 0:
        base_mesh = trimesh.voxel.ops.multibox(centers=point_list, pitch=1)
        base_mesh.export(output_path)
        
def visualize_depthmap(depthmap, output_path, flip=False):
    if isinstance(depthmap, np.ndarray):
        depthmap = depthmap.squeeze()

    elif isinstance(depthmap, torch.Tensor):
        depthmap = depthmap.squeeze().cpu().numpy()

    else:
        raise NotImplementedError

    if flip:
        depthmap = np.flip(depthmap, axis=1)
    rescaled = (255.0 / depthmap.max() * (depthmap - depthmap.min())).astype(np.uint8)
    im = Image.fromarray(rescaled)
    im.save(str(output_path) +'.png')
    pyexr.write(str(output_path) +'.exr', depthmap)

def visualize_implicit_rgb(value_grid, output_path):
    #rescaled = (255.0 / value_grid.max() * (value_grid - value_grid.min())).astype(np.uint8)
    img = Image.fromarray(value_grid.numpy(), 'RGB')
    im = Image.fromarray(img)
    im.save(str(output_path) +'.png')

def scale(path):
    dims = (139, 104, 112)
    mesh = trimesh.load(path, process=False)
    total_size = np.array(dims)
    #mesh.apply_translation(-np.array(dims)/2)
    mesh.apply_scale(1 / total_size)
    new_path = str(path)[:-4] + "_scaled.obj"
    print(new_path)
    mesh.export(new_path)

if __name__ == "__main__":
    path = Path("/home/alex/Documents/ifnet_scenes-main/ifnet_scenes/data/visualizations/overfit/00000")
    path = path / "mesh.obj"
    scale(path)