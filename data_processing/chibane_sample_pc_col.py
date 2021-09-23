import numpy as np
import trimesh
from glob import glob
import os
import multiprocessing as mp
from multiprocessing import Pool
import argparse
import traceback
import tqdm

def barycentric_coordinates(p, q, u, v):
    """
    Calculate barycentric coordinates of the given point
    :param p: a given point
    :param q: triangle vertex
    :param u: triangle vertex
    :param v: triangle vertex
    :return: 1X3 ndarray with the barycentric coordinates of p
    """
    v0 = u - q
    v1 = v - q
    v2 = p - q
    d00 = v0.dot(v0)
    d01 = v0.dot(v1)
    d11 = v1.dot(v1)
    d20 = v2.dot(v0)
    d21 = v2.dot(v1)
    denom = d00 * d11 - d01 * d01
    y = (d11 * d20 - d01 * d21) / denom
    z = (d00 * d21 - d01 * d20) / denom
    x = 1.0 - z - y
    return np.array([x, y, z])

def sample_colors(gt_mesh_path):
    try:
        path = os.path.normpath(gt_mesh_path)
        challange = path.split(os.sep)[-4]
        split = path.split(os.sep)[-3]
        gt_file_name = path.split(os.sep)[-2]
        full_file_name = path.split(os.sep)[-1][:-4]

        out_file = cfg['data_path'] + '/{}/{}/{}_color_samples{}_bbox{}.npz' \
            .format(split, gt_file_name, full_file_name, 200000, cfg['data_bounding_box_str'])

        if os.path.exists(out_file):
            print('File exists. Done.')
            return
        
        gt_mesh = (trimesh.load(gt_mesh_path))
        if isinstance(gt_mesh, trimesh.Scene):
            gt_mesh = gt_mesh.dump().sum()

        sample_points, face_idxs = gt_mesh.sample(200000, return_index = True)

        triangles = gt_mesh.triangles[face_idxs]
        face_vertices = gt_mesh.faces[face_idxs]
        faces_uvs = gt_mesh.visual.uv[face_vertices]

        q = triangles[:, 0]
        u = triangles[:, 1]
        v = triangles[:, 2]

        uvs = []

        for i, p in enumerate(sample_points):
            barycentric_weights = barycentric_coordinates(p, q[i], u[i], v[i])
            uv = np.average(faces_uvs[i], 0, barycentric_weights)
            uvs.append(uv)

        texture = gt_mesh.visual.material.image

        colors = trimesh.visual.color.uv_to_color(np.array(uvs), texture)

        #np.savez(out_file, points = sample_points, grid_coords = utils.to_grid_sample_coords(sample_points, bbox), colors = colors[:,:3])
    except Exception as err:
        print('Error with {}: {}'.format(out_file, traceback.format_exc()))
