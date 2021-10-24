from typing import List
import numpy as np
import igl
import os
import subprocess
try:
    from subprocess import DEVNULL  # Python 3.
except ImportError:
    DEVNULL = open(os.devnull, 'wb')

import datetime

import trimesh
from plyfile import PlyData
from mesh_to_sdf import sample_sdf_near_surface, mesh_to_voxels, mesh_to_sdf
from pathlib import Path
from sklearn.neighbors import KDTree
import glob
import pyexr
from skimage.io import imsave
from skimage import img_as_ubyte
from skimage.color import rgba2rgb
import math


#open all the relevant paths (glog, pathlib)
# for each mesh: load (trimesh), translate, scale (-> save T), sample sdf, set RGB=0, save points
# for each mesh: load (cc), apply T, sample RGB, set sdf=0, save points, save mesh
def add_noise(points, sigma):
    n = len(points)
    
    #randomize which index gets which noise added, but leave order unchanged
    rndindx = np.random.permutation(np.array(range(n)))
    fine = int(0.495*n)
    coarse = fine+int(0.5*n)
    a,b,c = np.split(rndindx, [fine, coarse])

    #points[o] += 0
    points[a] += np.random.normal(0, 0.003, size=(points[:fine].shape))
    points[b] += np.random.normal(0, 0.02, size=(points[fine:coarse].shape))
    points[c] += np.random.normal(0, 0.08, size=(points[coarse:].shape))
    random_points = np.random.uniform(-1, 1, size=(points[coarse:].shape))
    return points, random_points

def reprojected_sampling(inpath, outpath, verbose=False, num_points=100000, sigma=0.003, truncate=0.1, voxel_resolution=64):
    pointcloud = trimesh.load(inpath)
    #points from reprojection (~400k)
    #if points < threshold return error
    #subsample 10k points (sdf=0)


def process_sample_v3(inpath, outpath, verbose=False, num_points=100000, sigma=0.003, truncate=1.0, voxel_resolution=64):
    if verbose: print(inpath,'\n',outpath)
    name = str(inpath).split('/')[-1]

    mesh, transform_matrix = load_and_normalize_mesh(inpath) 
    transform_txt = str(outpath / 'transform.txt')
    np.savetxt(transform_txt, transform_matrix) 

    if verbose: print(f'starting job {name}')
    points, rgb, _ = process_sample_rgb(inpath, outpath, num_points)

    noisy_points, points_rnd = add_noise(points, sigma)
    noisy_rgb = copy_colors_nearest_neighbor(points, noisy_points, rgb, k=5)
    rgb_rnd = copy_colors_nearest_neighbor(points, points_rnd, rgb, k=5)

    points = np.append(noisy_points, points_rnd).reshape(-1,3)
    rgb = np.append(noisy_rgb,rgb_rnd).reshape(-1,3)

    if np.any( mesh.centroid >0.02 ): 
        print(f'model not centered')
        print(f'{mesh.centroid = }')
        print(f'{mesh.extents = }')
    #get SDF with backup method if first method fails
    sdf = mesh_to_sdf(mesh, points)
    if np.count_nonzero(sdf < 0) < 40000:
        print('not enough inside points for camera, trying sdf_to_mesh:depth')
        sdf_2 = mesh_to_sdf(mesh, points, sign_method='depth')
        if np.count_nonzero(sdf < sdf_2): sdf = sdf_2
        else: pass

    #points_outside = points[points > mesh.bounds[1] or points < mesh.bounds[0]]
    #set all sdf for outside points to |sdf|
    sdf[np.any(points > mesh.bounds[1],axis=1)] = np.abs(sdf[np.any(points > mesh.bounds[1],axis=1)])
    sdf[np.any(points < mesh.bounds[0],axis=1)] = np.abs(sdf[np.any(points < mesh.bounds[0],axis=1)])
    
    sdf[sdf > truncate] = truncate
    sdf[sdf < -truncate] = -truncate

    udf = np.absolute(sdf)
    
    #this turns sdf into occupancy
    occupancy = np.zeros_like(sdf)
    occupancy[sdf <= 0] = 1
    occupancy[sdf > 0] = 0    

    filename = str(outpath / 'point_samples_fix')
    np.savez(filename, points=points, rgb=rgb, sdf=sdf, occupancy=occupancy, udf=udf)

    if voxel_resolution > 0:
        if verbose: print(f'voxelizing')
        vox = mesh_to_voxels(trimesh.load(inpath), voxel_resolution=voxel_resolution)
        vox /= 2 #bring it from -1,1 mesh to -0.5, 0.5 mesh values
        vox[vox < - truncate] = -truncate
        vox[vox > truncate] = truncate
        np.save(str(outpath / 'voxels'), vox)
          
    if verbose: print(f'finished job {name}')
    return points, rgb, occupancy

def process_sample_v2(inpath, outpath, verbose=False, num_points=100000, sigma=0.003, truncate=1.0, voxel_resolution=64):
    if verbose: print(inpath,'\n',outpath)
    name = str(inpath).split('/')[-1]
    
    if verbose: print(f'starting job {name}')
    points, rgb, _ = process_sample_rgb(inpath, outpath, num_points)
    
    points, points_rnd = add_noise(points, sigma)
    rgb_rnd = copy_colors_nearest_neighbor(points, points_rnd, rgb)

    mesh, transform_matrix = load_and_normalize_mesh(inpath) 
    transform_txt = str(outpath / 'transform.txt')
    np.savetxt(transform_txt, transform_matrix) 
    
    verts = mesh.vertices
    faces = mesh.faces

    points = np.append(points, points_rnd).reshape(-1,3)
    rgb = np.append(rgb,rgb_rnd).reshape(-1,3)
    sdf, _, _ = igl.signed_distance(points, verts, faces)
    
    filename = str(outpath / 'point_samples_new')

    sdf[sdf > truncate] = truncate
    sdf[sdf < -truncate] = -truncate
    
    #this turns sdf into occupancy
    occupancy = np.zeros_like(sdf)
    occupancy[sdf <= 0] = 1
    occupancy[sdf > 0] = 0

    udf = sdf.copy()
    udf[sdf < 0] = -sdf[sdf<0]
    udf[udf > truncate] = truncate

    np.savez(filename, points=points, rgb=rgb, sdf=sdf, occupancy=occupancy, udf=udf)

    if voxel_resolution > 0:
        if verbose: print(f'voxelizing')
        vox = mesh_to_voxels(trimesh.load(inpath), voxel_resolution=voxel_resolution)
        vox /= 2 #bring it from -1,1 mesh to -0.5, 0.5 mesh values
        vox[vox < - truncate] = -truncate
        vox[vox > truncate] = truncate
        np.save(str(outpath / 'voxels'), vox)
          
    if verbose: print(f'finished job {name}')
    return points, rgb, occupancy

def process_sample(inpath, outpath, verbose=False, num_points=100000, fraction=45/50, sigma=0.003, truncate=1.0, voxel_resolution=64):
    if verbose: print(inpath,'\n',outpath)
    name = str(inpath).split('/')[-1]
    if verbose: print(f'starting sdf-job {name}')
    points, sdf = process_sample_sdf(inpath, outpath, num_points, fraction, sigma)

    if verbose: print(f'starting rgb-job {name}')
    points_2, rgb_2, sdf_2 = process_sample_rgb(inpath, outpath, num_points)
    rgb = copy_colors_nearest_neighbor(points_2, points, rgb_2)

    points = np.append(points,points_2).reshape(-1,3)
    rgb = np.append(rgb,rgb_2).reshape(-1,3)
    sdf = np.append(sdf,sdf_2)

    filename = str(outpath / 'point_samples')

    sdf[sdf > truncate] = truncate
    sdf[sdf < -truncate] = -truncate
    
    #this turns sdf into occupancy
    occupancy = np.zeros_like(sdf)
    occupancy[sdf <= 0] = 1
    occupancy[sdf > 0] = 0

    udf = sdf.copy()
    udf[sdf < 0] = -sdf[sdf<0]
    udf[udf > truncate] = truncate

    np.savez(filename, points=points, rgb=rgb, sdf=sdf, occupancy=occupancy, udf=udf)

    if verbose: print(f'voxelizing')
    vox = mesh_to_voxels(trimesh.load(inpath), voxel_resolution=voxel_resolution)
    vox /= 2 #bring it from -1,1 mesh to -0.5, 0.5 mesh values
    vox[vox < - truncate] = -truncate
    vox[vox > truncate] = truncate
    np.save(str(outpath / 'voxels'), vox)

    if verbose: print(f'finished job {name}')
    return points, rgb, occupancy


def process_sample_sdf(inpath, outpath, num_points, fraction, sigma):
    
    mesh, transform_matrix = load_and_normalize_mesh(inpath)        
    points, sdf = compute_sdf_from_mesh(mesh, num_points, fraction, sigma)

    transform_txt = str(outpath / 'transform.txt')
    np.savetxt(transform_txt, transform_matrix) 
    
    return points, sdf

def copy_colors_nearest_neighbor(col_points, nocol_points, rgb_col, k=2):
    tree = KDTree(col_points, leaf_size=35)  
    dist, ind = tree.query(nocol_points, k)
    rgb_nocol = np.zeros_like(nocol_points)
    
    #linear
    #for i in range(k):
    #    rgb_nocol += rgb_col[ind[:,i]]
    #rgb_nocol = np.round(rgb_nocol/k)

    #quadratic --> looks better? https://sighack.com/post/averaging-rgb-colors-the-right-way
    rgb_nocol += np.sum(rgb_col[ind]*rgb_col[ind],axis=1)
    rgb_nocol = np.sqrt( rgb_nocol / k)
    return rgb_nocol 


def fix_surface_points(path):
    #path = path.replace('disn_mesh.obj', '')
    print(f'{datetime.datetime.now()}, reading {path}')
    pts_npz = np.load(path+'/surface_points_blender.npz')
    car = trimesh.load(path+'/disn_mesh.obj')    
    filename = str(path +'/surface_points_blender.npz')

    points = pts_npz['points']
    rgb = pts_npz['rgb']
    
    sdf = mesh_to_sdf.mesh_to_sdf(car, points)
    occupancy = np.zeros_like(sdf)
    occupancy[sdf <= 0] = 1
    occupancy[sdf > 0] = 0
    
    np.savez(filename, points=points, rgb=rgb, sdf=sdf, occupancy=occupancy)

def process_sample_rgb(inpath, outpath, num_points):  
    cc_path = cc_extract_colored_pc(inpath, outpath, num_points)
    points, rgb, sdf = points_from_ply_xyzrgb(cc_path)
    return points, rgb, sdf

def read_names(inpath=None):
    if inpath is None: inpath = '../data/ShapeNetCore.v2/names.txt'
    id_dict = {}
    with open(inpath, 'r') as f:
        textbody = f.readlines()
        for line in textbody:
            line = line.strip()
            id, name = line[:8], line[9:]
            id_dict[f'{name}'] = id
    return id_dict

#helpers
def cc_extract_colored_pc(filepath, outpath, num_points):
    transform = str(outpath / 'transform.txt')
    output_mesh = str(outpath / 'norm.obj')
    output_ply = str(outpath / 'norm.ply')
    subprocess.run(['CloudCompare','-SILENT','-o',filepath,'-AUTO_SAVE','OFF','-APPLY_TRANS',transform,'-SAMPLE_MESH','POINTS',f'{num_points}','-C_EXPORT_FMT','PLY','-M_EXPORT_FMT','OBJ','-SAVE_CLOUDS', 'FILE',output_ply,'-SAVE_MESHES','FILE',output_mesh], stdout=DEVNULL, stderr=DEVNULL)#, capture_output=True)#,    ])
    return output_ply

def cc_norm_mesh(filepath, outpath):
    transform = str(outpath / 'transform.txt')
    output_mesh = str(outpath / 'norm.obj')
    subprocess.run(['CloudCompare','-SILENT','-o',filepath,'-AUTO_SAVE','OFF','-APPLY_TRANS',transform,'-M_EXPORT_FMT','OBJ','-SAVE_MESHES','FILE',output_mesh], stdout=DEVNULL, stderr=DEVNULL)#, capture_output=True)#,    ])
    

def points_from_ply_xyzrgb(filename, out_path=None):
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        points = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        rgb = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        
        points[:,0] = plydata['vertex'].data['x']
        points[:,1] = plydata['vertex'].data['y']
        points[:,2] = plydata['vertex'].data['z']
        
        rgb[:,0] = plydata['vertex'].data['red']
        rgb[:,1] = plydata['vertex'].data['green']
        rgb[:,2] = plydata['vertex'].data['blue']

        sdf=np.zeros((points.shape[0],1))

        if out_path is not None:
            np.savez(out_path, points=points, rgb=rgb, sdf=sdf)
        return points, rgb, sdf


def compute_sdf_from_mesh(mesh, num_points, fraction, sigma):
    points, sdf = sample_sdf_near_surface(mesh, number_of_points=num_points, sign_method='depth', surface_point_method='scan', sphere_radius=np.sqrt(2), surface_sample_frac=fraction, sigma=sigma)
    translation, scale = compute_unit_sphere_transform(mesh)
    points = (points / scale) - translation
    sdf /= scale
    return points, sdf


def compute_unit_sphere_transform(mesh):
    """
    returns translation and scale, which is applied to meshes before computing their SDF cloud
    """
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("The mesh parameter must be a trimesh mesh.")
     
    # the transformation applied by mesh_to_sdf.scale_to_unit_sphere(mesh)
    translation = -mesh.bounding_box.centroid
    scale = 1 / np.max(np.linalg.norm(mesh.vertices + translation, axis=1))
    return translation, scale


def load_and_normalize_mesh(filepath):
    mesh = trimesh.load_mesh(filepath)
    
    normalizing = 1/(mesh.extents).max()
    centering = -mesh.centroid
    
    # 4x4 transformation matrix in homogeneous coordinates
    T = np.eye(4)*normalizing
    T[:3,3] = centering*normalizing
    T[3,3] = 1

    mesh.apply_transform(T)
    return mesh, T


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


def category_processor_blender(category, inpath='../data/processed', outpath='../data/processed', offset=0, sigma=0.003, truncate=0.2, num_points=100000, visualize=False, voxel_resolution=64):
    inpath = Path(inpath)
    #outpath = Path(outpath)

    #id_dict = read_names()
    #cat_id = id_dict[category]
    
    #names = sorted(glob.glob(str(f'{inpath}/{cat_id}/*/models/model_normalized.obj')))
    #names = sorted(glob.glob(str(f'{inpath}/car/*/disn_mesh.obj')))
    #print(names)
    item = f'{inpath}/disn_mesh.obj'
    # item = f'{inpath}/disn_mesh.obj'
    out_folder = inpath
    print(f'processing {item}, {datetime.datetime.now()}')
    
    #for i, item in enumerate(names):
    #print(i+offset, datetime.datetime.now())
    #out_folder = Path(outpath / category / str(i+offset).zfill(5))
    out_folder.mkdir(exist_ok=True, parents=True)
    #with open(out_folder / 'name.txt', 'w') as f:
    #    f.write(item)
    
    #item = rel-path to norm.obj
    #points, rgb, occupancy = process_sample_blender(item, out_folder, sigma=sigma, truncate=truncate, num_points=num_points, voxel_resolution=voxel_resolution)
    try:
        points, rgb, occupancy = pointcloud_fixed(item, out_folder, sigma=sigma, truncate=truncate, num_points=num_points, voxel_resolution=voxel_resolution)
        if visualize:
            pointcloud = np.concatenate((points, rgb), axis=-1)
            visualize_rgb_pointcloud(pointcloud[occupancy == 1], out_folder/'fixed_pointcloud_in.obj')
            visualize_rgb_pointcloud(pointcloud[occupancy == 0], out_folder/'fixed_pointcloud_out.obj')
    except:
        print(f'{inpath} has not compiled')
    


def process_sample_blender(item, out_folder, sigma, truncate, num_points, voxel_resolution):
    mesh = trimesh.load(item)

    images_path = str(out_folder).split('/')[:-1]
    images_path = '/'.join(images_path) + '/'

    #load mesh in trimesh, normalize it, save it, render it
    mesh, transform_matrix = load_and_normalize_mesh(item) 
    transform_txt = str(out_folder / 'transform.txt')
    np.savetxt(transform_txt, transform_matrix) 

    cc_norm_mesh(item, out_folder)
    render_blender(str(out_folder)+'/norm.obj', out_folder)
    unproj_pts = reproject_blender(out_folder)

    rndarray = np.random.randint(0, unproj_pts.shape[0], size=(num_points,))
    partial_unproj_points = unproj_pts[rndarray]

    opoints, orgb = partial_unproj_points[...,:3], partial_unproj_points[...,3:]
    newpts, newptsrnd = add_noise(opoints, sigma)

    noisy_rgb = copy_colors_nearest_neighbor(opoints, newpts, orgb, k=5)
    rgb_rnd = copy_colors_nearest_neighbor(opoints, newptsrnd, orgb, k=5)

    pts = np.append(newpts, newptsrnd).reshape(-1,3).astype(np.float32)
    rgb = np.append(noisy_rgb,rgb_rnd).reshape(-1,3).astype(np.float32)

    sdf = mesh_to_sdf(mesh, pts)
    if np.count_nonzero(sdf < 0) < 20000:
        print('not enough inside points for camera, trying sdf_to_mesh:depth')
        sdf_2 = mesh_to_sdf(mesh, pts, sign_method='depth')
        if np.count_nonzero(sdf < sdf_2): sdf = sdf_2
        else: pass

    if np.any(mesh.centroid > 0.1):
        print(f'possibly erroneous mesh, {item} is not centered')
    #points_outside = points[points > mesh.bounds[1] or points < mesh.bounds[0]]
    #set all sdf for outside points to |sdf|
    sdf[np.any(pts > mesh.bounds[1],axis=1)] = np.abs(sdf[np.any(pts > mesh.bounds[1],axis=1)])
    sdf[np.any(pts < mesh.bounds[0],axis=1)] = np.abs(sdf[np.any(pts < mesh.bounds[0],axis=1)])

    sdf[sdf > truncate] = truncate
    sdf[sdf < -truncate] = -truncate

    udf = np.absolute(sdf)

    #this turns sdf into occupancy
    occupancy = np.zeros_like(sdf)
    occupancy[sdf <= 0] = 1
    occupancy[sdf > 0] = 0  

    filename = str(out_folder / 'point_samples_blender')
    np.savez(filename, points=pts, rgb=rgb, sdf=sdf.astype(np.float32), occupancy=occupancy.astype(np.float32), udf=udf.astype(np.float32))

    filename_exr = str(out_folder) + '/_r_336.exr'
    image = pyexr.read(filename_exr)
    image[...,:3] = (image[...,:3]-image[...,:3].min())/(image[...,:3].max()-image[...,:3].min())
    image = rgba2rgb(image)
    imsave(filename_exr.replace('exr','png'), img_as_ubyte(image))

    if voxel_resolution > 0:
        vox = mesh_to_voxels(mesh, voxel_resolution=voxel_resolution)
        vox /= 2 #bring it from -1,1 mesh to -0.5, 0.5 mesh values
        vox[vox < - truncate] = -truncate
        vox[vox > truncate] = truncate
        np.save(str(out_folder / 'voxels'), vox)
          
    return pts, rgb, occupancy

def render_blender(inpath, outpath, format='OPEN_EXR'):
    #inpath ='/media/alex/SSD Datastorage/data/processed/car/00000/norm.obj'
    #outpath = '../data/blender/car/'
    if format == 'OPEN_EXR':
        cdepth = 16
    else: cdepth = 8 
    subprocess.run(['blender','--background','--python','data_processing/render_blender.py','--','--output_folder',f'{outpath}',f'{inpath}','--format',f'{format}','--color_depth',f'{cdepth}'], stdout=DEVNULL, stderr=DEVNULL)#, capture_output=True)#,    ])

def reproject_image(image, depthmap, RT=None, f=245, cx = 112, cy = 112):
    #depth 65000 and the coord transform is only for blender images

    xyz = depth_to_camera(-depthmap, f, cx, cy).transpose(1,0)
    
    #filter non-surface points (depth > 65500)
    mask = depthmap.flatten() 
    xyz = xyz[np.nonzero(mask < 65500), :].reshape(-1,3)
    
    if RT is not None:
        R = RT[:3,:3]
        T = RT[:3, 3]
    
        #transform back into world coordinates from view
        xyz = (R.transpose(1,0) @ xyz.transpose(1,0)).transpose(1,0)
        xyz = xyz + (R.transpose(1,0) @ np.expand_dims(T,0).transpose(1,0)).squeeze()

        #Bring from blender coordinates to CV convention
        R_b2CV = np.array([[-1,0,0],[0,0,-1],[0,1,0]], dtype=np.float32)
        xyz = (R_b2CV @ xyz.transpose(1,0)).transpose(1,0).squeeze() 

    if image is not None:
        #add color to the pointcloud via image
        rgb_image = image.reshape(-1,3)
        rgb_image = rgb_image[np.nonzero(mask < 65500), :].reshape(-1,3)   
        xyz = np.concatenate((xyz, rgb_image), axis=-1)
    
    return xyz

def save_surface_points(inpath):
    print(inpath)
    unproj_pts = reproject_blender(inpath)
    points, rgb = unproj_pts[...,:3], unproj_pts[...,3:]
    pointcloud = np.concatenate((points, rgb), axis=-1)
    visualize_rgb_pointcloud(pointcloud, inpath+'/surface_pointcloud.obj')



def pointcloud_fixed(item, out_folder, sigma, truncate, num_points, voxel_resolution):
    mesh = trimesh.load(item)

    images_path = str(out_folder).split('/')[:-1]
    images_path = '/'.join(images_path) + '/'

    #load mesh in trimesh, normalize it, save it, render it
    unproj_pts = reproject_blender(out_folder)

    rndarray = np.random.randint(0, unproj_pts.shape[0], size=(num_points,))
    partial_unproj_points = unproj_pts[rndarray]

    opoints, orgb = partial_unproj_points[...,:3], partial_unproj_points[...,3:]
    
    #save the surface points
    osdf = mesh_to_sdf(mesh, opoints)
    filename = str(out_folder / 'surface_points_blender')
    occupancy = np.zeros_like(osdf)
    occupancy[osdf <= 0] = 1
    occupancy[osdf > 0] = 0  
    np.savez(filename, points=opoints, rgb=orgb, sdf=osdf.astype(np.float32), occupancy=occupancy.astype(np.float32))
    
    newpts, newptsrnd = add_noise(opoints, sigma)

    noisy_rgb = copy_colors_nearest_neighbor(opoints, newpts, orgb, k=5)
    rgb_rnd = copy_colors_nearest_neighbor(opoints, newptsrnd, orgb, k=5)

    pts = np.append(newpts, newptsrnd).reshape(-1,3).astype(np.float32)
    rgb = np.append(noisy_rgb,rgb_rnd).reshape(-1,3).astype(np.float32)

    sdf = mesh_to_sdf(mesh, pts)

    if np.count_nonzero(sdf < 0) < 20000:
        print('not enough inside points for camera, trying sdf_to_mesh:depth')
        sdf_2 = mesh_to_sdf(mesh, pts, sign_method='depth')
        if np.count_nonzero(sdf < sdf_2): sdf = sdf_2
        else: pass

    if np.any(np.abs(mesh.centroid) > 0.1):
        print(f'possibly erroneous mesh, {item} is not centered')
    #points_outside = points[points > mesh.bounds[1] or points < mesh.bounds[0]]
    #set all sdf for outside points to |sdf|
    sdf[np.any(pts > mesh.bounds[1],axis=1)] = np.abs(sdf[np.any(pts > mesh.bounds[1],axis=1)])
    sdf[np.any(pts < mesh.bounds[0],axis=1)] = np.abs(sdf[np.any(pts < mesh.bounds[0],axis=1)])

    sdf[sdf > truncate] = truncate
    sdf[sdf < -truncate] = -truncate

    #this turns sdf into occupancy
    occupancy = np.zeros_like(sdf)
    occupancy[sdf <= 0] = 1
    occupancy[sdf > 0] = 0  

    filename = str(out_folder / 'fixed_point_samples_blender')
    np.savez(filename, points=pts, rgb=rgb, sdf=sdf.astype(np.float32), occupancy=occupancy.astype(np.float32))

    filename_exr = str(out_folder) + '/_r_336.exr'
    image = pyexr.read(filename_exr)
    image[...,:3] = image[...,:3]/3
    image = rgba2rgb(image)
    #image = (image[...,:3]-image[...,:3].min())/(image[...,:3].max()-image[...,:3].min())    
    try:
        imsave(filename_exr.replace('exr','png'), img_as_ubyte(image))
    except ValueError:
        print(f'{item} threw an error while saving the image')

    if voxel_resolution > 0:
        vox = mesh_to_voxels(mesh, voxel_resolution=voxel_resolution)
        vox /= 2 #bring it from -1,1 mesh to -0.5, 0.5 mesh values
        vox[vox < - truncate] = -truncate
        vox[vox > truncate] = truncate
        np.save(str(out_folder / 'voxels_fixed'), vox)
          
    return pts, rgb, occupancy

def generate_frustum(image_size, intrinsic_inv, depth_min, depth_max):
    x = image_size[0]
    y = image_size[1]
    eight_points = np.array([[0 * depth_min, 0 * depth_min, depth_min, 1.0],
                             [0 * depth_min, y * depth_min, depth_min, 1.0],
                             [x * depth_min, y * depth_min, depth_min, 1.0],
                             [x * depth_min, 0 * depth_min, depth_min, 1.0],
                             [0 * depth_max, 0 * depth_max, depth_max, 1.0],
                             [0 * depth_max, y * depth_max, depth_max, 1.0],
                             [x * depth_max, y * depth_max, depth_max, 1.0],
                             [x * depth_max, 0 * depth_max, depth_max, 1.0]]).transpose()
    frustum = np.dot(intrinsic_inv, eight_points)
    frustum = frustum.transpose()
    return frustum[:, :3]

def generate_frustum_volume(frustum, voxelsize):
    maxx = np.max(frustum[:, 0]) / voxelsize
    maxy = np.max(frustum[:, 1]) / voxelsize
    maxz = np.max(frustum[:, 2]) / voxelsize
    minx = np.min(frustum[:, 0]) / voxelsize
    miny = np.min(frustum[:, 1]) / voxelsize
    minz = np.min(frustum[:, 2]) / voxelsize

    dimX = math.ceil(maxx - minx)
    dimY = math.ceil(maxy - miny)
    dimZ = math.ceil(maxz - minz)
    camera2frustum = np.array([[1.0 / voxelsize, 0, 0, -minx],
                               [0, 1.0 / voxelsize, 0, -miny],
                               [0, 0, 1.0 / voxelsize, -minz],
                               [0, 0, 0, 1.0]])

    return (dimX, dimY, dimZ), camera2frustum


def reproject_blender(inpath):
    #path = '/media/alex/SSD Datastorage/data/blender/00000/'
    path = str(inpath)
    idx = path.split('/')[-2]
    f = 245

    unproj_points = np.array([])

    for i in range(30):
        #load inputs per image
        rgb_img = pyexr.read(path+f'/_r_{str((i)*12).zfill(3)}.exr').astype(np.float32)
        rgb_img = rgb_img[...,:3]/3
        depthmap = pyexr.read(path+f'/_r_{str((i)*12).zfill(3)}_depth0001.exr').astype(np.float32)[...,0]
        RT = np.loadtxt(path+f'/RT{str((i)*12).zfill(3)}.txt').astype(np.float32)

        xyz = reproject_image(rgb_img, depthmap, RT=RT, f=245, cx = 112, cy = 112)

        unproj_points = np.append(unproj_points, xyz).reshape(-1,6)
    
    return unproj_points

def depth_to_camera(depth_map, f=245, cx=112, cy=112):
    v, u = np.meshgrid(np.arange(depth_map.shape[-2]), np.arange(depth_map.shape[-1]))
    X = -np.multiply(v, depth_map)/f + cx * depth_map/f 
    Y = -np.multiply(u, depth_map)/f + cy * depth_map/f 
    Z = depth_map
    return np.stack((X.flatten(), Y.flatten(), Z.flatten())) 

def check_remaining():
    g1 = sorted(glob.glob('../data/processed/car/*/point_samples_fix.npz'))
    g2 = sorted(glob.glob('../data/processed/car/*/rgb00.png'))
    g1b = ['/'.join(g.split('/')[:-1]) for g in g1]
    g2b = ['/'.join(g.split('/')[:-1]) for g in g2]
    g1c = set(g1b)
    g2c = set(g2b)
    remaining = sorted(list(g1c-g2c))
    print(len(remaining), remaining)


if __name__ == '__main__':
    #do this with argparse
    from util.arguments import argparse
    from util.visualize import visualize_rgb_pointcloud
    from Manifold_sample import cloud_compare_icp, read_and_apply_icp_norm
    
    from util.bookkeeping import list_model_ids
    from util.visualize import render_shapenet
    import mesh_to_sdf
    import warnings
    from tqdm import tqdm
    from joblib import Parallel, delayed
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(
        description="""Creates pointcloud samples with 45parts sigma stddev,
                        45parts 10*sigma stddev and 10 parts sqrt(2) uniform sampling.
                        UDF and SDF are truncated to truncate value"""
    )
    #parser.add_argument('--inpath', type=str, default='../data/ShapeNetCore.v2')
    parser.add_argument('--inpath', type=str, default='../data/blender')
    parser.add_argument('--outpath', type=str, default='../data/blender')
    parser.add_argument('--truncate', type=float, default=0.1)
    parser.add_argument('--sigma', type=float, default=0.003)
    parser.add_argument('--num_points', type=int, default=200000)
    parser.add_argument('--category', type=str, default='car')
    parser.add_argument('--offset', type=int, default=3500)
    parser.add_argument('--voxel_resolution', type=int, default=64)
    parser.add_argument('--visualize', type=bool, default=True)    

    args = parser.parse_args()

    inpath = Path(args.inpath)
    outpath = Path(args.outpath)
    truncate = args.truncate
    sigma = args.sigma
    category = args.category
    offset = args.offset
    num_points = args.num_points
    voxel_resolution = args.voxel_resolution
    visualize = args.visualize

    outpath.mkdir(exist_ok=True, parents=True)
    id_dict = list_model_ids(save=True)
    print('processing')
    #fix_sampling_sdf(offset=offset) 
    #
    #offset = [i for i in range(3400,3500)]
    #render_blender('a','b')    
    #offset = [i for i in range(10)]
    offset = 680
    #names = sorted(glob.glob(str(f'{inpath}/car/*/disn_mesh.obj')))
    #names = [f.replace('/disn_mesh.obj', '') for f in names]
    names = sorted(glob.glob(str(f'{inpath}/car/*/surface_points_blender.npz')))
    names = [f.replace('/surface_points_blender.npz', '') for f in names]
    
    
    out_folder = inpath
    if isinstance(offset, int):
        names = names[offset:]
        print(len(names),'samples remaining')
    elif isinstance(offset, list):
        print(len(names) - offset[0],'samples remaining') 
        names = [names[i] for i in offset]
        offset = offset[0]
    print(len(names), names[0],names[-1])
    #check_remaining()
    #Parallel(n_jobs=4)(delayed(category_processor_blender)(category, inpath = i, outpath=names, offset=offset, sigma=sigma, truncate=truncate, num_points=num_points, voxel_resolution=voxel_resolution, visualize=visualize) for i in tqdm(names))
    #Parallel(n_jobs=-1)(delayed(save_surface_points)(i)for i in tqdm(names))
    #Parallel(n_jobs=4)(delayed(category_processor_blender)(category=category, inpath=i, outpath=i, offset=offset, sigma=sigma, truncate=truncate, num_points=num_points, voxel_resolution=voxel_resolution, visualize=visualize) for i in tqdm(names))
    Parallel(n_jobs=8)(delayed(fix_surface_points)(i) for i in tqdm(names))