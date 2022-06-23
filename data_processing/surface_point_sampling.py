from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from pytorch3d.io.obj_io import load_obj, load_objs_as_meshes
import mesh_to_sdf
import datetime
import glob

import torch

import torch.nn as nn
from pathlib import Path
import trimesh
import numpy as np
import igl

def sample_surface_mesh_pytorch3d(inpath, num_points = 100000):
    try:
        mesh = load_objs_as_meshes([inpath]).to('cuda:0')
        mesh
        ptc, normals = sample_points_from_meshes(mesh, num_samples = num_points, return_normals=True)
        ptc = ptc.squeeze().cpu().numpy().astype(np.float32)
        outpath = inpath.replace('disn_mesh.obj', 'ptc_surface_v2')
        np.savez(outpath, points=ptc, normals=normals)
    except ValueError:
        print('ERROR:',inpath.split('/')[-2])

def fix_points(path):
    #path = path.replace('disn_mesh.obj', '')
    print(f'{datetime.datetime.now()}, reading {path}')
    path = path.replace('/fixed_point_samples_blender.npz','')
    pts_npz = np.load(path+'/fixed_point_samples_blender.npz')
    car = trimesh.load(path+'/disn_mesh.obj')    
    filename_out = str(path +'/fixed_point_samples_blender2.npz')

    points = pts_npz['points']
    rgb = pts_npz['rgb']
    
    sdf = mesh_to_sdf.mesh_to_sdf(car, points)
    #_,sdf = compute_sdf_from_mesh
    occupancy = np.zeros_like(sdf)
    occupancy[sdf <= 0] = 1
    occupancy[sdf > 0] = 0
    
    np.savez(filename_out, points=points, rgb=rgb, sdf=sdf, occupancy=occupancy)

def fix_sdf_with_normals(path):
    print(f'{datetime.datetime.now()}, reading {path}')
    path = path.replace('/fixed_point_samples_blender.npz','')
    pts_npz = np.load(path+'/fixed_point_samples_blender2.npz')
    mesh = trimesh.load(path+'/disn_mesh.obj')    
    
    filename = str(path +'/fixed_point_samples_blender3.npz')
    points = pts_npz['points']
    rgb = pts_npz['rgb']
    
    sdf, I, C, N = igl.signed_distance(points, mesh.vertices, mesh.faces, return_normals=True)

    surf_pts = points - sdf.reshape(sdf.shape[0],1)*N/np.linalg.norm(N,axis=1).reshape(N.shape[0],1)
    surf_sdf, _, _ = igl.signed_distance(surf_pts, mesh.vertices, mesh.faces, return_normals=False)
    
    
    sdf = sdf[np.abs(surf_sdf) < 1e-4]
    points = points[np.abs(surf_sdf) < 1e-4]
    N = N[np.abs(surf_sdf) < 1e-4] 
    rgb = rgb[np.abs(surf_sdf) < 1e-4]    
    
    #visualize if sdf & pseudo-normals were correct
    #surf_pts = surf_pts[np.abs(surf_sdf) < 1e-4]      
    #visualize_rgb_pointcloud(np.concatenate([surf_pts, rgb], axis=1), 'test_surfpcloud.obj')
    
    np.savez(filename, points=points, rgb=rgb, sdf=sdf, normals=N)
    
    #return sdf, N, points, rgb

if __name__ == '__main__':
    #do this with argparse
    from util.arguments import argparse    
    #import mesh_to_sdf
    import warnings
    from tqdm import tqdm
    from joblib import Parallel, delayed
    from pathlib import Path
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(
        description="""Creates pointcloud samples with 45parts sigma stddev,
                        45parts 10*sigma stddev and 10 parts sqrt(2) uniform sampling.
                        UDF and SDF are truncated to truncate value"""
    )
    #parser.add_argument('--inpath', type=str, default='../data/ShapeNetCore.v2')
    parser.add_argument('--inpath', type=str, default='../data/blender')
    parser.add_argument('--outpath', type=str, default='../data/blender')
    parser.add_argument('--num_points', type=int, default=100000)
    parser.add_argument('--category', type=str, default='car')
    parser.add_argument('--offset', type=int, default=0)

    args = parser.parse_args()

    inpath = Path(args.inpath)
    outpath = Path(args.outpath)
    category = args.category
    offset = args.offset
    num_points = args.num_points

    outpath.mkdir(exist_ok=True, parents=True)
    print('processing')
    #fix_sampling_sdf(offset=offset) 
    #
    #offset = [i for i in range(3400,3500)]
    #render_blender('a','b')    
    #offset = [i for i in range(10)]
    #offset = 2625
    
    #names = sorted(glob.glob(str(f'{inpath}/car/*/disn_mesh.obj')))
    #names = [f.replace('/disn_mesh.obj', '') for f in names]
    names = sorted(glob.glob(str(f'{inpath}/car/*/fixed_point_samples_blender.npz')))
    #names = [f.replace('/surface_points_blender.npz', '') for f in names]
    #names = names[:3]
    #offset=[2134, 2152, 2161, 2188, 2143, 2169, 2178, 2124]
    if isinstance(offset, int):
        names = names[offset:]
        print(len(names),'samples remaining')
    elif isinstance(offset, list):
        print(len(names) - offset[0],'samples remaining') 
        names = [names[i] for i in offset]
        offset = offset[0]
    print(len(names), names[0],names[-1])
    #Parallel(n_jobs=4)(delayed(sample_surface_mesh_pytorch3d)(i) for i in tqdm(names))
    #Parallel(n_jobs=4)(delayed(fix_points)(i) for i in tqdm(names))
    Parallel(n_jobs=6)(delayed(fix_sdf_with_normals)(i) for i in tqdm(names))
    