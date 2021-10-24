import mcubes as mc
import numpy as np
from pytorch3d.renderer.materials import Materials
import trimesh
from PIL import Image
from pathlib import Path
import pyexr
import torch
from util.load_obj import read_obj
import imageio
from skimage import img_as_ubyte
import pyexr

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.datasets import (ShapeNetCore)
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights,
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesVertex,
    HardFlatShader
)

def render_mesh(obj_filename, outpath=None, viewangle = None, cull_backfaces = False, elevation=None, start=None):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    verts, faces = read_obj(obj_filename)
    verts = torch.from_numpy(verts)
    faces = torch.from_numpy(faces).unsqueeze(0).to(device)

    verts_cv = verts.detach().clone()
    
    verts_cv[...,0] = -verts[...,0]
    verts_cv[...,2] = -verts[...,2]

    verts = verts_cv.unsqueeze(0).to(device)

    verts, rgb = torch.split(verts, 3, dim=-1)
    
    textures = TexturesVertex(verts_features=rgb)
    mesh = Meshes(verts, faces, textures)

    if viewangle is not None and elevation is None:
        batchsize=4
        elevation = np.arctan(0.6/1)*180/np.pi
    
        lastpt = 360 - 360/batchsize - 60
        azim = torch.linspace(-60, lastpt, batchsize)

        R, T = look_at_view_transform(1, 20, azim)
        #30.963756 is atan(0.6/1.0) in ° (which correlates to the angle from Blender rendering)
        Rview, Tview = look_at_view_transform(1, elevation, viewangle)
        R, T = torch.cat((Rview, R), axis=0),  torch.cat((Tview, T), axis=0)
        batchsize += 1
    
    elif elevation is not None:
        steps=30
        batchsize=1
        lastpt = 360/steps*(batchsize*(1+start)-1)
        azim = torch.linspace(360*batchsize*(start)/steps, lastpt, batchsize)
        R, T = look_at_view_transform(1, elevation, azim)

    meshes = mesh.extend(batchsize)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    raster_settings = RasterizationSettings(
        image_size=224, 
        blur_radius=0.0001, 
        faces_per_pixel=3,
        cull_backfaces = False, 
        max_faces_per_bin=300000,
        perspective_correct=True,        
    )
    #lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    #this supposedly fixes triangular artifacts
    location = torch.tensor([0.0, 0.0, -3.0])#.repeat(batch_size,1)
    location = R @ location

    lights = PointLights(
        device=device, 
        location=location,
        ambient_color=((1, 1, 1),), 
        diffuse_color=((0, 0, 0),),
        specular_color=((0, 0, 0),),
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )

    materials = Materials(
        device=device,
        specular_color=[[1.0, 1.0, 1.0]],
        shininess=100.0
    )
    images = renderer(meshes, lights=lights, materials=materials, cameras=cameras)
    
    if outpath is not None:
        if start is None:
            for i in range(batchsize):
                visualize_implicit_rgb(images[i], outpath+str(i))
        else:
            visualize_implicit_rgb(images[0], outpath+str(start))

    return images

#for pixelwise loss:
#render model_prediction under equal circumstances


def render_shapenet(shapenet_id_dict, outpath='../data/processed/car', idx=0, batchsize = 12, distance=1.2, verbose=True):
    #shapenet_idx is a tuple (idx, id)
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    #device = torch.device("cpu")
    SHAPENET_PATH = "../data/ShapeNetCore.v2"
    shapenet_dataset = ShapeNetCore(SHAPENET_PATH, version = 2, synsets=['02958343'])

    #render for every mesh
    ids = shapenet_id_dict[int(idx)]
    
    if verbose: print('rendering:',idx, ids, datetime.datetime.now())
    ids = [ids] * batchsize
    
    lastpt = 360 - 360/batchsize - 180
    azim = torch.linspace(-180, lastpt, batchsize)
    R, T = look_at_view_transform(distance, 20, azim)

    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    raster_settings = RasterizationSettings(
        image_size=224, 
        blur_radius=0.0001, 
        faces_per_pixel=3,
        cull_backfaces = False, 
        max_faces_per_bin=300000,
        perspective_correct=True,        
    )

    location = torch.tensor([0.0, 0.0, -3.0])#.repeat(batch_size,1)
    location = R @ location
    lights = PointLights(
        device=device, 
        location=location,
        ambient_color=((1, 1, 1),), 
        diffuse_color=((0, 0, 0),),
        specular_color=((0, 0, 0),),
    )
    
    #this line uses a modified MeshRenderer (replaced MeshRenderer with MeshRendererWithFragments in ShapeNet Dataset Base Class)
    images, fragments = shapenet_dataset.render(
        model_ids=ids,
        device=device,
        cameras=cameras,
        raster_settings=raster_settings,
        lights=lights,
        shader_type=HardFlatShader,
    
    )

    z = fragments.zbuf[...,0]
    z = z.unsqueeze(-1)

    if outpath is not None:
        
        out_path = Path(outpath) / str(idx).zfill(5)
        

        images_exr = torch.cat((images, z), dim=-1)
        z_img = (z - z.min()) / (z.max() - z.min())*2-1

        for j in range(batchsize):
            pyexr.write(str(out_path / f'rgb{str(j).zfill(2)}.exr'), images_exr[j].squeeze().cpu().numpy(), ['R','G','B','A','D'])

            
            imageio.imwrite(str(out_path / f'rgb{str(j).zfill(2)}.png'), img_as_ubyte(images[j,:,:,:].cpu()))
            imageio.imwrite(str(out_path / f'depth{str(j).zfill(2)}.png'), img_as_ubyte(z_img[j,:,:,:].cpu()))
    else:
        return images, z


def visualize_implicit_rgb(value_grid, output_path):
    img = tensor_to_numpy(value_grid)
    img = imageio.imwrite(str(output_path)+'.png', img[...,:3])

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        tensor = tensor * 256
        tensor[tensor > 255] = 255
        tensor[tensor < 0] = 0
        tensor = tensor.type(torch.uint8)
        if tensor.size(-1) > 4: tensor = tensor.permute(1, 2, 0)
        tensor = tensor.detach().cpu().numpy()
    return tensor


def rgba_to_rgb(source):

    #target = source.copy()
    #target[0] = ((1-))

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
        f.write('v %f %f %f %f %f %f\n' % (x, y, z, c[0], c[1], c[2]))
    f.close()

def visualize_rgb_pointcloud(point_cloud, output_path):
    c = np.zeros_like(point_cloud[:,:3])
    #print(c.shape)
    with open(output_path, "w") as f:        
        if point_cloud.shape[1] == 6:
            c = point_cloud[:, 3:]
        for i in range(point_cloud.shape[0]):          
            x, y, z = point_cloud[i, 0], point_cloud[i, 1], point_cloud[i, 2]
            f.write('v %f %f %f %f %f %f\n' % (x, y ,z, c[i,0], c[i,1], c[i,2]))


def visualize_sdf(sdf, output_path, level=0.75, rgb=False, export=False):
    if rgb:
        rgb = sdf[...,1:]
        sdf = sdf[...,0].squeeze()
    vertices, triangles = mc.marching_cubes(sdf.astype(float), level)
    vertices /= sdf.shape[0]
    vertices -= 0.5
    
    if export: 
        mc.export_obj(vertices, triangles, output_path)
        print(vertices.shape, sdf.shape) #could use rgb here, define and export a pointcloud

    return vertices, triangles

def make_col_mesh(verts, faces, rgb, outpath):
    if verts.size > 0:         
        #min max norm might not give authentic colors
        #rgb = ((rgb - rgb.min())/(rgb.max() - rgb.min())*255).astype(np.uint8)
        
        if np.all(rgb>=0) and np.all(rgb<=1): rgb = (255*(rgb)).astype(np.uint8) #this if rgb €[0,1] --> #from (-1,1) to (0,1)
        elif np.all(rgb >= -1) and np.all(rgb<= 1): rgb = (255*(rgb/2 + 0.5)).astype(np.uint8)  #this if rgb €[-1,1] --> from (-1,1) to (0,1)

        else: 
            print('vertex colors out of range')
            rgb = ((rgb - rgb.min())/(rgb.max() - rgb.min())*255).astype(np.uint8)
        mymesh = trimesh.Trimesh(vertices=verts, vertex_colors=rgb, faces=faces)
        print(verts.shape, faces.shape, rgb.shape)
        with open(outpath, 'w') as f:
            f.writelines(trimesh.exchange.obj.export_obj(mymesh))

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


def scale(path):
    dims = (139, 104, 112)
    mesh = trimesh.load(path, process=False)
    total_size = np.array(dims)
    #mesh.apply_translation(-np.array(dims)/2)
    mesh.apply_scale(1 / total_size)
    new_path = str(path)[:-4] + "_scaled.obj"
    #print(new_path)
    mesh.export(new_path)

if __name__ == "__main__":
    from util.bookkeeping import list_model_ids
    import datetime
    import warnings
    warnings.filterwarnings('ignore')
    id_dict = list_model_ids(save=False)
    offset = 3471
    offset = [i for i in range(offset, len(id_dict.keys()))]
    offset = [3]

    print('renderings remaining:',len(id_dict.keys())-offset[0])
    for i in offset:
        try: render_shapenet(id_dict, idx=i, outpath='../data/processed/car')
        except IndexError:
            print(i,'Index Error')
        except RuntimeError:
            print(i,'Memory Error')