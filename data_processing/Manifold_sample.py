#Manifold-sample
import glob
import subprocess
import os
from pathlib import Path
import trimesh
import numpy as np
import time

try:
    from subprocess import DEVNULL  # Python 3.
except ImportError:
    DEVNULL = open(os.devnull, 'wb')


def convert_to_manifold(filepaths):
    for filepath in (filepaths):
        print(filepath)
        out_dir = filepath.replace('norm.obj', 'manifold_norm.obj')
        #print(out_dir)
        try:
            #subprocess.run(['../ManifoldPlus/build/manifold','--input',filepath, '--output', out_dir, "--depth", "9"], timeout=120, check=True, capture_output=True)
            subprocess.run(['../ManifoldPlus/build/manifold','--input',filepath, '--output', out_dir], timeout=120, check=True, capture_output=True)
        
        except subprocess.SubprocessError:
            print("{} failed to run",format(filepath))
            return False
    return True

def reveal_names(filepaths):
    names = []
    for filepath in (filepaths):
        filepath = filepath.replace('norm.obj', 'name.txt')
        with open(filepath, 'r') as f:
            original_path = f.readlines()[0]
            name = original_path.split('/')[4]
            #print(name)
        
        shapenet_name = Path(filepath.replace('name.txt', 'name/'+name))
        shapenet_name.mkdir(exist_ok=True, parents=True)
        #names.append(shapenet_name)
    #return names

def transform_preprocssed(in_path, disn_list):
    names = reveal_names(in_path)
    disn_names = glob.glob('/media/alex/SSD Datastorage/data/DISN SHapenet/02958343/*/isosurf.obj')
    for i, name in enumerate(names):
        pass
        #find name in disn_names
        #load disn mesh, transform it and save it in original folder
        #if it cant be found, return some error, append to some list

def scale_and_save_disn_meshes():
    disn_names = glob.glob('/media/alex/SSD Datastorage/data/DISN SHapenet/02958343/*/isosurf.obj')
    for meshname in disn_names:
        mesh = trimesh.load(meshname, process=False, force=True)
        mesh = transform_disn(mesh)
        findpath = meshname.split('/')[-2]
        outpath = glob.glob('../data/blender/car/*/name/'+findpath)[0].split('/')[:5]
        outpath = ('/').join(outpath)
        othermesh_path = outpath + '/norm.obj'
        othermesh = trimesh.load(othermesh_path, process=False)
        #print(othermesh.extents - mesh.extents)
        #print(np.any(othermesh.extents - mesh.extents > 0.01))
        trimesh.exchange.export.export_mesh(mesh, outpath+'/disn_mesh.obj')
        if not np.any(np.abs(othermesh.extents - mesh.extents) > 0.01):
            pass
        else:
            print(f'mesh:{othermesh_path.split("/")[-2]} does not fit')
            #trimesh.exchange.export.export_mesh(mesh, outpath+'/disn_mesh_err.obj')

def transform_disn(mesh):
    H_neg90_y = rotationmatrix_y(-np.pi/2)
    mesh.bounding_box.extents
    mesh.apply_transform(H_neg90_y)
    #print(1/mesh.bounding_box.extents.max())

    mesh.apply_translation(-mesh.centroid)
    mesh.apply_scale(1/mesh.bounding_box.extents.max())

    #print(mesh.bounding_box.extents, mesh.centroid)
    return mesh

def rotationmatrix_y(phi):
    R = np.array([[np.cos(phi), 0, np.sin(phi)],[0,1,0], [np.sin(phi), 0, np.cos(phi)]])
    H = np.zeros((4,4))
    H[:3,:3] = R
    H[3,3] = 1
    return H

def cloud_compare_icp(filepath, mode='mesh'):
    
    disn_obj = filepath+'disn_mesh.obj'
    if 'mesh' in mode:
        GT = filepath+'norm.obj'
        subprocess.run(['CloudCompare','-SILENT','-o',GT,'-o',disn_obj,'-AUTO_SAVE','OFF','-MATCH_CENTERS','-ICP',
        '-REFERENCE_IS_FIRST', '-ADJUST_SCALE', '-MIN_ERROR_DIFF', '1e-6','-M_EXPORT_FMT', 'OBJ', '-SAVE_MESHES', 
        'FILE', f'GT_dump.obj {filepath}disn_mesh.obj'], stdout=DEVNULL, stderr=DEVNULL, timeout=600, check=True)
    elif 'pc' in mode:
        GT = filepath+'surface_pointcloud.obj'
        subprocess.run(['CloudCompare','-SILENT','-o', GT,'-o',disn_obj,'-REFERENCE_IS_FIRST','-AUTO_SAVE','OFF','-ICP', '-ADJUST_SCALE',
        '-FARTHEST_REMOVAL','-OVERLAP', '100', '-RANDOM_SAMPLING_LIMIT', '100000', 
        '-MIN_ERROR_DIFF', '1e-6'], stdout=DEVNULL, stderr=DEVNULL, timeout=120, check=True, capture_output=True)
    
    
def read_and_apply_icp_norm(filepath):
    norm_path = sorted(glob.glob(filepath+'*_REGISTRATION_MATRIX_*'))
    #norm_matrix = np.loadtxt(norm_path[-1])
    print(norm_path[-1])
    mesh = trimesh.load(filepath+'disn_mesh.obj')
    #mesh.apply_transform(norm_matrix)
    other_mesh = trimesh.load(filepath+'norm.obj')
    if (np.any(np.abs(mesh.extents - other_mesh.extents) > 1e-2)):
        print(mesh.extents - other_mesh.extents)
        print(f'{filepath}-mesh possibly badly aligned')
    #mesh.export(filepath+'disn_mesh.obj')

def fix_meshes(filepath):
    #for filepath in filepaths:
    #print(filepath)
    try:
        cloud_compare_icp(filepath, 'mesh')
    except subprocess.SubprocessError as E:
        print(f"{filepath} failed to run: {E}")
        return False
    return True
    #read_and_apply_icp_norm(filepath)

def compare_meshes(filepath):
    #print(filepath)
    
    try:
        fix_meshes(filepath)
        mesh = trimesh.load(filepath+'disn_mesh.obj')
        other_mesh = trimesh.load(filepath+'norm.obj')
        if (np.any(np.abs(mesh.extents - other_mesh.extents) > 2e-2)):
            print(f'{filepath}-mesh possibly still misaligned: {np.abs((mesh.extents - other_mesh.extents)).max()}')
        else:
            print(f'successfully aligned {filepath}')
    except: 
        ValueError
        print(f'{filepath} throws value error')
    

if __name__ == '__main__':
    from tqdm import tqdm
    from joblib import Parallel, delayed
    
    #in_dir = '../data/blender/car/00000/'
    
    in_dir = '../data/blender/car/*/disn_mesh.obj'
    #in_dir = in_dir+'norm.obj'

    #print(in_dir)
    filepaths = sorted(glob.glob(in_dir))
    filepaths = [path.replace('disn_mesh.obj','') for path in filepaths][100:]
    print(len(filepaths), filepaths[0],filepaths[-1])
    #print(filepaths)
    #fix_meshes(filepaths)

    #convert_to_manifold(filepaths)
    #reveal_names(filepaths)
    #trimesh.load('name.obj', process=False)
    #Parallel(n_jobs=-1)(delayed(fix_meshes)(i) for i in tqdm(filepaths))
    Parallel(n_jobs=-1)(delayed(compare_meshes)(i) for i in tqdm(filepaths))