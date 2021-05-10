from pathlib import Path
import numpy as np
import subprocess
import trimesh
from plyfile import PlyData

from mesh_to_sdf import sample_sdf_near_surface
from util.visualize import visualize_rgb_pointcloud

def cc_extract_pc_sdf(filepath, out_path):
    mesh = trimesh.load_mesh(filepath)
    points, sdf = sample_sdf_near_surface(mesh, number_of_points=250000)
    translation, scale = compute_unit_sphere_transform(mesh)
    points = (points / scale) - translation
    sdf /= scale
    #visualize_rgb_pointcloud(points, 'points.obj')
    np.savez(out_path, points=points, rgb=np.zeros_like(points), sdf=sdf)
    #return points, sdf

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
        if out_path is not None:
            np.savez(out_path, points=points, rgb=rgb, sdf=np.zeros((points.shape[0],1)))
        #return points, rgb

def cc_extract_colored_pc(filepath, out_path):
    #command = "CloudCompare -O "+filepath
    
    #os.system('cmd /k f{command}') #k to remain, c to close
    #result = subprocess.check_output(['sudo','service','mpd','restart'])
    # samples colored pointcloud from textured mesh and saves it as ply in ascii format (3xslower than binary, 3sec per mesh)
    subprocess.run(['CloudCompare','-SILENT','-o',filepath,'-SAMPLE_MESH','POINTS','1000000','-C_EXPORT_FMT','PLY','-PLY_EXPORT_FMT','ASCII','-SAVE_CLOUDS', 'FILE',out_path])#, capture_output=True)#,    ])

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

if __name__ == "__main__":
    #path = 'data/model_normalized.obj'
    plyfile = 'T_pc_col.ply'
    #out_path = 'pc_sdf'
    ply_out_path = 'pc_surface'
    #cc_extract_pc_sdf(path, out_path)
    points_from_ply_xyzrgb(plyfile, ply_out_path)
    #cc_extract_colored_pc(path, out_path)