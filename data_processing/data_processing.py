import numpy as np
import subprocess
import trimesh
from plyfile import PlyData
from mesh_to_sdf import sample_sdf_near_surface
#import glob
#from ..util.arguments import *
from pathlib import Path

#open all the relevant paths (glog, pathlib)
# for each mesh: load (trimesh), translate, scale (-> save T), sample sdf, set RGB=0, save points
# for each mesh: load (cc), apply T, sample RGB, set sdf=0, save points, save mesh

def process_sample(inpath, outpath, verbose=False, num_points=100000, fraction=45/50, sigma=0.003):
    print(inpath,'\n',outpath)
    name = str(inpath).split('/')[-1]
    if verbose: print(f'starting sdf-job {name}')
    points, rgb, sdf = process_sample_sdf(inpath, outpath, num_points, fraction, sigma)

    if verbose: print(f'starting rgb-job {name}')
    points_2, rgb_2, sdf_2 = process_sample_rgb(inpath, outpath, num_points)

    points = np.append(points,points_2).reshape(-1,3)
    rgb = np.append(rgb,rgb_2).reshape(-1,3)
    sdf = np.append(sdf,sdf_2)

    filename = str(outpath / 'point_samples')
    np.savez(filename, points=points, rgb=rgb, sdf=sdf)

    if verbose: print(f'finished job {name}')


def process_sample_sdf(inpath, outpath, num_points, fraction, sigma):
    
    mesh, transform_matrix = load_and_normalize_mesh(inpath, outpath)        
    points, sdf = compute_sdf_from_mesh(mesh, num_points, fraction, sigma)
    rgb = np.zeros_like(points)

    transform_txt = str(outpath / 'transform.txt')
    np.savetxt(transform_txt, transform_matrix) 
    
    return points, rgb, sdf


def process_sample_rgb(inpath, outpath, num_points):  
    cc_path = cc_extract_colored_pc(inpath, outpath, num_points)
    points, rgb, sdf = points_from_ply_xyzrgb(cc_path)
    return points, rgb, sdf


#helpers
def cc_extract_colored_pc(filepath, outpath, num_points):
    #command = "CloudCompare -O "+filepath
    
    #os.system('cmd /k f{command}') #k to remain, c to close
    #result = subprocess.check_output(['sudo','service','mpd','restart'])
    # samples colored pointcloud from textured mesh and saves it as ply in ascii format (3xslower than binary, 3sec per mesh)
    transform = str(outpath / 'transform.txt')
    output_mesh = str(outpath / 'norm.obj')
    output_ply = str(outpath / 'norm.ply')
    subprocess.run(['CloudCompare','-SILENT','-o',filepath,'-AUTO_SAVE','OFF','-APPLY_TRANS',transform,'-SAMPLE_MESH','POINTS',f'{num_points}','-C_EXPORT_FMT','PLY','-M_EXPORT_FMT','OBJ','-SAVE_CLOUDS', 'FILE',output_ply,'-SAVE_MESHES','FILE',output_mesh])#, capture_output=True)#,    ])
    return output_ply
    #subprocess.run(['CloudCompare','-SILENT','-o',filepath,'-APPLY_TRANS',transform,'-SAMPLE_MESH','POINTS','100000','-C_EXPORT_FMT','PLY','-PLY_EXPORT_FMT','ASCII','-SAVE_CLOUDS', 'FILE',out_path])#, capture_output=True)#,    ])


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
    points, sdf = sample_sdf_near_surface(mesh, number_of_points=num_points, sign_method='normal', surface_point_method='scan', sphere_radius=np.sqrt(2), surface_sample_frac=fraction, sigma=sigma)
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


def load_and_normalize_mesh(filepath, out_path):
    mesh = trimesh.load_mesh(filepath)
    normalizing = 1/(mesh.extents).max()
    centering = -mesh.centroid
    mesh.apply_translation(centering)
    mesh.apply_scale(normalizing)

    # 4x4 transformation matrix in homogeneous coordinates
    T = np.eye(4)*normalizing
    T[:3,3] = centering*normalizing
    T[3,3] = 1
    return mesh, T


if __name__ == '__main__':
    inpath = Path('/media/alex/SSD Datastorage/data/processed/overfit/car/85f6145747a203becc08ff8f1f541268/models/model_normalized.obj')
    outpath = Path('test/police_car')
    outpath.mkdir(exist_ok=True, parents=True)
    print('processing')
    process_sample(inpath, outpath, verbose=True)